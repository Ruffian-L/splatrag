use crate::ann::KeyedAnn;
use crate::basin_score::{
    basin_aware_final, basin_centroid_affinity, basin_peer_dense_support,
    local_semantic_neighborhood_boost, local_semantic_neighborhood_sum,
    peer_support_for_candidate,
};
use crate::config::RetrievalConfig;
use crate::embedding::{EmbeddingClient, matryoshka64};
use crate::geometry::{cosine as geom_cosine, HotState};
use crate::lexical::LexicalIndex;
use crate::qdrant::QdrantIndex;
use crate::record::{MemoryRecord, RecallContext, RecallFilters, RecallHit, ScoreBreakdown};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub struct RetrievalEngine<'a> {
    pub embedding: &'a EmbeddingClient,
    pub qdrant: &'a QdrantIndex,
    pub ann: &'a KeyedAnn,
    pub lexical: &'a LexicalIndex,
    pub config: &'a RetrievalConfig,
}

impl RetrievalEngine<'_> {
    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        filters: &RecallFilters,
        records: &HashMap<Uuid, MemoryRecord>,
        hot: &HotState,
    ) -> Result<Vec<RecallHit>> {
        let limit = limit.clamp(1, 100);
        let candidate_limit = (limit * self.config.candidate_multiplier).max(20);
        let lexical = self.lexical.search(query, candidate_limit)?;
        let full_query = self.embedding.embed(query).await?;
        let local_query = matryoshka64(&full_query)?;
        let local_dense = self.ann.search(&local_query, candidate_limit)?;
        let qdrant_dense = self.qdrant.search(&full_query, candidate_limit).await?;

        let mut bm25 = HashMap::new();
        let mut cosine_scores = HashMap::new();
        let mut candidate_ids = HashSet::new();
        for hit in lexical {
            candidate_ids.insert(hit.id);
            bm25.insert(hit.id, hit.score);
        }
        for hit in local_dense {
            candidate_ids.insert(hit.id);
            cosine_scores
                .entry(hit.id)
                .and_modify(|score: &mut f32| *score = score.max(1.0 - hit.distance))
                .or_insert(1.0 - hit.distance);
        }
        for hit in qdrant_dense {
            candidate_ids.insert(hit.id);
            cosine_scores
                .entry(hit.id)
                .and_modify(|score| *score = score.max(hit.score))
                .or_insert(hit.score);
        }

        let splats = hot.splat_map();
        let basins = hot.basin_map();

        // Basin expansion: geometry co-members of top dense hits join the pool.
        if self.config.basin_expand_max_size > 0 {
            let mut dense_ranked: Vec<(Uuid, f32)> =
                cosine_scores.iter().map(|(id, s)| (*id, *s)).collect();
            dense_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let seeds = self.config.basin_expand_seeds.max(1);
            let mut expanded_basins = HashSet::new();
            for (id, score) in dense_ranked.into_iter().take(seeds) {
                let Some(splat) = splats.get(&id) else {
                    continue;
                };
                let Some(basin_id) = splat.basin_id.as_deref() else {
                    continue;
                };
                if !expanded_basins.insert(basin_id.to_string()) {
                    continue;
                }
                let Some(basin) = basins.get(basin_id) else {
                    continue;
                };
                if basin.member_ids.len() > self.config.basin_expand_max_size {
                    continue;
                }
                // Inherited soft cosine so expanded peers are not orthogonal zeros.
                let inherited = (score * 0.85).max(0.05);
                for mid in &basin.member_ids {
                    if candidate_ids.insert(*mid) {
                        cosine_scores.entry(*mid).or_insert(inherited);
                    }
                }
            }
        }

        // HNSW graph expansion: neighbors of top dense *and* top BM25 hits
        // (local cluster walk — pulls both_miss golds that sit next to a hit in ANN).
        // Expanded nodes get *true* query–semantics cosine (not a fake inherit), so they
        // can compete in hybrid rank without the global "fill all missing cosines" regression.
        if self.config.basin_graph_expand > 0 {
            let seeds = self.config.basin_expand_seeds.max(1);
            let mut seed_list: Vec<Uuid> = Vec::new();
            let mut dense_ranked: Vec<(Uuid, f32)> =
                cosine_scores.iter().map(|(id, s)| (*id, *s)).collect();
            dense_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (id, _) in dense_ranked.into_iter().take(seeds) {
                seed_list.push(id);
            }
            let mut bm25_ranked: Vec<(Uuid, f32)> =
                bm25.iter().map(|(id, s)| (*id, *s)).collect();
            bm25_ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (id, _) in bm25_ranked.into_iter().take(seeds) {
                seed_list.push(id);
            }
            // Nearest-first graph expand (neighbors_for_key is distance-ordered).
            // Only *new* candidates get true query–semantics cosine. Filling cosine for
            // every BM25-only neighbor was measured as a regression (R 0.8518 → 0.842).
            let mut expanded = HashSet::new();
            for id in seed_list {
                let Ok(neighbors) = self.ann.neighbors_for_key(id) else {
                    continue;
                };
                for mid in neighbors
                    .into_iter()
                    .take(self.config.basin_graph_expand)
                {
                    if candidate_ids.insert(mid) {
                        expanded.insert(mid);
                    }
                }
            }
            for mid in expanded {
                if let Some(splat) = splats.get(&mid) {
                    let c = geom_cosine(&local_query, &splat.semantics).max(0.0);
                    cosine_scores
                        .entry(mid)
                        .and_modify(|s| *s = s.max(c))
                        .or_insert(c);
                }
            }
        }
        let mut sorted_cosine = cosine_scores.values().copied().collect::<Vec<_>>();
        sorted_cosine.sort_by(|a, b| b.total_cmp(a));
        let damping = homeostatic_damping(&sorted_cosine);
        let effective_radiance_weight = self.config.weight_radiance * damping;

        // Per-query rank lists for RRF (1-based ranks among candidates that have a score).
        let bm25_rank = rank_map(&bm25);
        let cosine_rank = rank_map(&cosine_scores);
        let bm25_norm = if self.config.normalize_scores {
            minmax_normalize(&bm25)
        } else {
            HashMap::new()
        };
        let cosine_norm = if self.config.normalize_scores {
            minmax_normalize(&cosine_scores)
        } else {
            HashMap::new()
        };
        let use_rrf = self.config.fusion.eq_ignore_ascii_case("rrf");
        let rrf_k = self.config.rrf_k.max(1.0);

        // Basin geometry for ranking (membership from hot field; no domain stamps).
        let id_to_basin: HashMap<Uuid, Option<String>> = splats
            .iter()
            .map(|(id, splat)| (*id, splat.basin_id.clone()))
            .collect();
        let peer_support = if self.config.weight_basin_peer != 0.0 {
            basin_peer_dense_support(&cosine_scores, &id_to_basin)
        } else {
            HashMap::new()
        };
        let id_to_semantics: HashMap<Uuid, Vec<f32>> = if self.config.weight_basin_centroid != 0.0
        {
            splats
                .iter()
                .map(|(id, splat)| (*id, splat.semantics.clone()))
                .collect()
        } else {
            HashMap::new()
        };
        // Cache centroid affinity per basin id for this query.
        let mut centroid_cache: HashMap<String, f32> = HashMap::new();
        if self.config.weight_basin_centroid != 0.0 {
            for basin in hot.basins.iter() {
                let aff = basin_centroid_affinity(
                    &local_query,
                    &basin.member_ids,
                    &id_to_semantics,
                );
                centroid_cache.insert(basin.id.clone(), aff);
            }
        }

        // Top dense hits with semantics for local soft-cluster boost.
        let dense_hits_sem: Vec<(f32, Vec<f32>)> = if self.config.weight_basin_local != 0.0 {
            let mut ranked: Vec<(Uuid, f32)> =
                cosine_scores.iter().map(|(id, s)| (*id, *s)).collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            ranked
                .into_iter()
                .take(24)
                .filter_map(|(id, score)| {
                    let sem = splats.get(&id)?.semantics.clone();
                    Some((score, sem))
                })
                .collect()
        } else {
            Vec::new()
        };

        let mut hits = Vec::new();
        for id in candidate_ids {
            let Some(memory) = records.get(&id) else {
                continue;
            };
            if !memory.matches(filters) {
                continue;
            }
            let splat = splats.get(&id);
            let basin_id = splat.and_then(|splat| splat.basin_id.clone());
            if let Some(required) = &filters.basin_id
                && basin_id.as_ref() != Some(required)
            {
                continue;
            }
            let radiance = splat
                .map(|splat| splat.radiance / (splat.radiance + 1.0))
                .unwrap_or(0.0);
            let bm25_score = bm25.get(&id).copied().unwrap_or(0.0);
            // 0.0 here means the ANN never returned this candidate — it was found by BM25 alone.
            //
            // That looks like a bug, and it was "fixed" once by computing the exact cosine from the
            // splat's semantics, on the reasoning that unmeasured is not the same as orthogonal.
            // Measured on SciFact against an identical store, that cost recall@10 0.8518 -> 0.8354
            // and nDCG@10 0.7257 -> 0.7142. The absence is *information*: the ANN declining to
            // return a candidate is evidence about it, and replacing that with a ~0.7 similarity
            // promotes weak lexical matches by ~7 points at weight_cosine 10.0.
            //
            // Leave it. If this is revisited, revisit it with a measurement and not an argument.
            let cosine = cosine_scores.get(&id).copied().unwrap_or(0.0);
            let base = if use_rrf {
                let mut s = 0.0;
                if let Some(r) = bm25_rank.get(&id) {
                    s += self.config.weight_bm25 / (rrf_k + *r as f32);
                }
                if let Some(r) = cosine_rank.get(&id) {
                    s += self.config.weight_cosine / (rrf_k + *r as f32);
                }
                if effective_radiance_weight != 0.0 {
                    s += radiance * effective_radiance_weight * 0.01;
                }
                s
            } else {
                let b = if self.config.normalize_scores {
                    bm25_norm.get(&id).copied().unwrap_or(0.0)
                } else {
                    bm25_score
                };
                let c = if self.config.normalize_scores {
                    cosine_norm.get(&id).copied().unwrap_or(0.0)
                } else {
                    cosine
                };
                b * self.config.weight_bm25
                    + c * self.config.weight_cosine
                    + radiance * effective_radiance_weight
            };
            let peer = peer_support_for_candidate(basin_id.as_deref(), &peer_support);
            let cent = basin_id
                .as_deref()
                .and_then(|b| centroid_cache.get(b).copied())
                .unwrap_or(0.0);
            let mut final_score = basin_aware_final(
                base,
                peer,
                cent,
                self.config.weight_basin_peer,
                self.config.weight_basin_centroid,
            );
            if self.config.weight_basin_local != 0.0 {
                if let Some(splat) = splat {
                    let local = if self.config.basin_local_sum {
                        local_semantic_neighborhood_sum(
                            &splat.semantics,
                            &dense_hits_sem,
                            self.config.basin_local_threshold,
                        )
                    } else {
                        local_semantic_neighborhood_boost(
                            &splat.semantics,
                            &dense_hits_sem,
                            self.config.basin_local_threshold,
                        )
                    };
                    final_score += self.config.weight_basin_local * local;
                }
            }
            let basin_label = basin_id
                .as_deref()
                .and_then(|id| basins.get(id))
                .map(|basin| basin.label.clone());
            hits.push(RecallHit {
                memory: memory.clone(),
                context: context_for(memory, records, 2),
                basin_id,
                basin_label,
                scores: ScoreBreakdown {
                    bm25: bm25_score,
                    cosine,
                    radiance,
                    radiance_weight: effective_radiance_weight,
                    final_score,
                },
            });
        }
        hits.sort_by(|a, b| {
            b.scores
                .final_score
                .total_cmp(&a.scores.final_score)
                .then_with(|| a.memory.id.cmp(&b.memory.id))
        });
        hits.truncate(limit);
        Ok(hits)
    }
}

fn rank_map(scores: &HashMap<Uuid, f32>) -> HashMap<Uuid, usize> {
    let mut items: Vec<(Uuid, f32)> = scores.iter().map(|(id, s)| (*id, *s)).collect();
    items.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items
        .into_iter()
        .enumerate()
        .map(|(i, (id, _))| (id, i + 1))
        .collect()
}

fn minmax_normalize(scores: &HashMap<Uuid, f32>) -> HashMap<Uuid, f32> {
    if scores.is_empty() {
        return HashMap::new();
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for v in scores.values() {
        min = min.min(*v);
        max = max.max(*v);
    }
    let span = (max - min).max(1e-9);
    scores
        .iter()
        .map(|(id, v)| (*id, (v - min) / span))
        .collect()
}

pub fn homeostatic_damping(cosine_scores: &[f32]) -> f32 {
    if cosine_scores.len() < 2 {
        return 0.95;
    }
    let sample = &cosine_scores[..cosine_scores.len().min(20)];
    let max_score = sample[0];
    let mean = sample.iter().sum::<f32>() / sample.len() as f32;
    let variance = sample
        .iter()
        .map(|score| (score - mean).powi(2))
        .sum::<f32>()
        / (sample.len() as f32 - 1.0).max(1.0);
    let std_dev = variance.sqrt();
    let adaptive_penalty = if (max_score > 0.75 && std_dev < 0.015) || std_dev > 0.05 {
        -0.01
    } else if std_dev <= 0.015 {
        -0.15
    } else {
        (-0.15 + (0.13 / 0.035) * (std_dev - 0.015)).clamp(-0.15, -0.02)
    };
    1.0 + adaptive_penalty
}

fn context_for(
    target: &MemoryRecord,
    records: &HashMap<Uuid, MemoryRecord>,
    radius: usize,
) -> RecallContext {
    let Some(conversation_id) = &target.conversation_id else {
        return RecallContext::default();
    };
    let mut conversation = records
        .values()
        .filter(|record| record.conversation_id.as_ref() == Some(conversation_id))
        .cloned()
        .collect::<Vec<_>>();
    conversation.sort_by(|a, b| {
        a.turn_index
            .cmp(&b.turn_index)
            .then_with(|| a.timestamp.cmp(&b.timestamp))
            .then_with(|| a.id.cmp(&b.id))
    });
    let Some(position) = conversation
        .iter()
        .position(|record| record.id == target.id)
    else {
        return RecallContext::default();
    };
    let start = position.saturating_sub(radius);
    let end = (position + radius + 1).min(conversation.len());
    RecallContext {
        before: conversation[start..position].to_vec(),
        after: conversation[position + 1..end].to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homeostasis_damps_flat_low_confidence_lists() {
        let flat = vec![0.4; 20];
        let clear = vec![0.95, 0.7, 0.5, 0.2];
        assert!(homeostatic_damping(&flat) < homeostatic_damping(&clear));
    }
}
