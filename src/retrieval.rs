use crate::ann::KeyedAnn;
use crate::basin_score::{
    basin_aware_final, basin_centroid_affinity, basin_peer_dense_support, local_boost_applies,
    local_semantic_neighborhood_boost, local_semantic_neighborhood_sum, peer_support_for_candidate,
};
use crate::config::RetrievalConfig;
use crate::embedding::{EmbeddingClient, matryoshka64};
use crate::geometry::{HotState, cosine as geom_cosine};
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
        let matching = matching_ids(records, filters);
        let restrict = filters_restrict(filters);
        let candidate_limit = candidate_limit_for(
            limit,
            self.config.candidate_multiplier,
            matching.len(),
            restrict,
        );
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
        // Date / speaker / conversation filters cannot wait for ANN+BM25 to
        // already contain the row. If the matching set is small, inject it.
        // Basin membership is hot-only (`matching_ids` cannot see it).
        if restrict && matching.len() <= FILTER_INJECT_CAP {
            candidate_ids.extend(matching.iter().copied());
        }
        candidate_ids.extend(basin_inject_ids(hot, filters));
        // Unique typos (`yesetday`, `ingestr`) can miss the stemmer dictionary on a
        // huge index even though the raw text is in cold. Substring-scan the rare
        // query against the record map and give those rows a BM25 rank.
        if rare_token_query(query) {
            for id in substring_ids(records, query, FILTER_INJECT_CAP) {
                candidate_ids.insert(id);
                bm25.entry(id).or_insert(RARE_TOKEN_BM25);
            }
        }
        // Marker substring injection is available (scientific_marker_tokens) but
        // boosting BM25-missing rows on this poisoned SciFact store added FPs and
        // lost freeze 0.8584/0.7295. Ranking, not pooling, is the remaining gap.
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
            let mut bm25_ranked: Vec<(Uuid, f32)> = bm25.iter().map(|(id, s)| (*id, *s)).collect();
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
                for mid in neighbors.into_iter().take(self.config.basin_graph_expand) {
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
        let id_to_semantics: HashMap<Uuid, Vec<f32>> = if self.config.weight_basin_centroid != 0.0 {
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
                let aff =
                    basin_centroid_affinity(&local_query, &basin.member_ids, &id_to_semantics);
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
                let mut s = rrf_score(
                    bm25_rank.get(&id).copied(),
                    cosine_rank.get(&id).copied(),
                    self.config.weight_bm25,
                    self.config.weight_cosine,
                    rrf_k,
                    self.config.rrf_mnz,
                );
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
            if self.config.weight_coverage != 0.0 {
                final_score +=
                    self.config.weight_coverage * query_term_coverage(query, &memory.text);
            }
            if self.config.weight_sentence != 0.0 {
                final_score +=
                    self.config.weight_sentence * max_sentence_coverage(query, &memory.text);
            }
            if self.config.weight_basin_local != 0.0
                && let Some(splat) = splat
            {
                let member_count = basin_id
                    .as_deref()
                    .and_then(|id| basins.get(id))
                    .map(|basin| basin.member_ids.len())
                    .unwrap_or(0);
                if local_boost_applies(member_count, self.config.basin_local_max_members) {
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
        hits = dedup_by_content_hash(hits);
        if self.config.rerank_pool > 1 {
            let pool = (limit.saturating_mul(self.config.rerank_pool)).max(limit);
            hits.truncate(pool);
            match self.config.rerank.as_str() {
                "bm25" => hits.sort_by(|a, b| {
                    b.scores
                        .bm25
                        .total_cmp(&a.scores.bm25)
                        .then_with(|| b.scores.final_score.total_cmp(&a.scores.final_score))
                        .then_with(|| a.memory.id.cmp(&b.memory.id))
                }),
                "cosine" => hits.sort_by(|a, b| {
                    b.scores
                        .cosine
                        .total_cmp(&a.scores.cosine)
                        .then_with(|| b.scores.final_score.total_cmp(&a.scores.final_score))
                        .then_with(|| a.memory.id.cmp(&b.memory.id))
                }),
                "full" => {
                    let texts: Vec<String> = hits
                        .iter()
                        .map(|hit| hit.memory.text.chars().take(1000).collect())
                        .collect();
                    let embeddings = self.embedding.embed_batch(&texts).await?;
                    let full_cos: Vec<f32> = embeddings
                        .iter()
                        .map(|emb| geom_cosine(&full_query, emb).max(0.0))
                        .collect();
                    let mut order: Vec<usize> = (0..full_cos.len()).collect();
                    order.sort_by(|&left, &right| {
                        full_cos[right]
                            .total_cmp(&full_cos[left])
                            .then_with(|| hits[left].memory.id.cmp(&hits[right].memory.id))
                    });
                    let mut rank_of = vec![0usize; order.len()];
                    for (rank, idx) in order.iter().enumerate() {
                        rank_of[*idx] = rank + 1;
                    }
                    let full_w = if self.config.weight_full != 0.0 {
                        self.config.weight_full
                    } else {
                        self.config.weight_cosine
                    };
                    for (idx, hit) in hits.iter_mut().enumerate() {
                        hit.scores.final_score +=
                            full_w / (self.config.rrf_k.max(1.0) + rank_of[idx] as f32);
                    }
                    hits.sort_by(|a, b| {
                        b.scores
                            .final_score
                            .total_cmp(&a.scores.final_score)
                            .then_with(|| a.memory.id.cmp(&b.memory.id))
                    });
                }
                _ => {}
            }
        }
        hits.truncate(limit);
        Ok(hits)
    }
}

/// Synthetic BM25 for a rare-token substring hit that Tantivy did not return.
pub const RARE_TOKEN_BM25: f32 = 80.0;
/// Boost added when a scientific marker token (PGE2, miR7a, IL-1) occurs in the text.
pub const MARKER_BM25_BOOST: f32 = 40.0;
/// Skip a marker that matches more rows than this — it is not a needle.
pub const MARKER_INJECT_CAP: usize = 80;

/// Inject every filter match when the matching set is this size or smaller.
pub const FILTER_INJECT_CAP: usize = 2048;
/// Floor on BM25/HNSW pool size when any recall filter is set.
pub const FILTER_CANDIDATE_FLOOR: usize = 500;

/// True when the caller asked for a facet, not an open search.
pub fn filters_restrict(filters: &RecallFilters) -> bool {
    !filters.domains.is_empty()
        || !filters.sources.is_empty()
        || !filters.speakers.is_empty()
        || !filters.models.is_empty()
        || filters.basin_id.is_some()
        || filters.conversation_id.is_some()
        || filters.after.is_some()
        || filters.before.is_some()
}

/// Memory ids that already satisfy the filters. Used to inject dated / conversation
/// rows that BM25 and HNSW would otherwise never retrieve.
///
/// Cold-only: `filters.basin_id` is ignored here (membership lives on hot).
pub fn matching_ids(records: &HashMap<Uuid, MemoryRecord>, filters: &RecallFilters) -> Vec<Uuid> {
    if !filters_restrict(filters) {
        return Vec::new();
    }
    records
        .iter()
        .filter(|(_, record)| record.matches(filters))
        .map(|(id, _)| *id)
        .collect()
}

/// Hot basin members for the same small-set inject as date/speaker.
/// Empty if no basin filter, missing basin, or member count above [`FILTER_INJECT_CAP`].
/// Does not write cosine (ANN-miss stays absent from the rank map).
pub fn basin_inject_ids(hot: &HotState, filters: &RecallFilters) -> Vec<Uuid> {
    let Some(basin_id) = filters.basin_id.as_deref() else {
        return Vec::new();
    };
    let basins = hot.basin_map();
    let Some(basin) = basins.get(basin_id) else {
        return Vec::new();
    };
    if basin.member_ids.len() > FILTER_INJECT_CAP {
        return Vec::new();
    }
    basin.member_ids.clone()
}

/// Tokens that BM25's stemmer will not treat as the needle: digits, greek, short hyphens.
#[allow(dead_code)]
fn marker_needles(marker: &str) -> Vec<String> {
    let mut needles = vec![marker.to_string()];
    let stripped = marker.replace('-', "");
    if stripped != marker {
        needles.push(stripped);
    }
    let ungreek = marker
        .replace('α', "alpha")
        .replace('β', "beta")
        .replace('μ', "mu")
        .replace('κ', "kappa");
    if ungreek != marker {
        needles.push(ungreek.replace('-', ""));
        needles.push(ungreek);
    }
    needles
}

pub fn scientific_marker_tokens(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in query.split_whitespace() {
        let trimmed: String = raw
            .chars()
            .skip_while(|c| {
                !c.is_alphanumeric() && *c != 'α' && *c != 'β' && *c != 'μ' && *c != 'κ'
            })
            .collect();
        let token = trimmed
            .trim_end_matches(|c: char| {
                !c.is_alphanumeric() && c != 'α' && c != 'β' && c != 'μ' && c != 'κ' && c != '-'
            })
            .trim_end_matches('-');
        if token.len() < 2 || token.len() > 24 {
            continue;
        }
        let lower = token.to_ascii_lowercase();
        let has_digit = lower.chars().any(|c| c.is_ascii_digit());
        let has_greek = token.chars().any(|c| !c.is_ascii());
        let has_hyphen = lower.contains('-') && lower.chars().any(|c| c.is_ascii_alphabetic());
        if !(has_digit || has_greek || (has_hyphen && lower.len() <= 16)) {
            continue;
        }
        if seen.insert(lower.clone()) {
            out.push(lower);
        }
    }
    out
}

pub fn rare_token_query(query: &str) -> bool {
    let trimmed = query.trim();
    if trimmed.len() < 5 || trimmed.len() > 48 {
        return false;
    }
    let words = trimmed.split_whitespace().count();
    words > 0 && words <= 2 && trimmed.chars().any(|c| c.is_alphabetic())
}

pub fn substring_ids(records: &HashMap<Uuid, MemoryRecord>, query: &str, cap: usize) -> Vec<Uuid> {
    let needle = query.trim().to_ascii_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut ids = Vec::new();
    for (id, record) in records {
        if record.text.to_ascii_lowercase().contains(&needle) {
            ids.push(*id);
            if ids.len() >= cap {
                break;
            }
        }
    }
    ids
}

pub fn candidate_limit_for(
    limit: usize,
    multiplier: usize,
    matching: usize,
    restrict: bool,
) -> usize {
    let base = (limit * multiplier).max(20);
    if !restrict {
        return base;
    }
    base.max(FILTER_CANDIDATE_FLOOR)
        .max(matching.min(FILTER_INJECT_CAP))
}

/// Keep the highest-ranked copy of identical text. Qdrant re-embeds and dual
/// ingest paths leave the same `content_sha256` under two memory ids.
fn dedup_by_content_hash(hits: Vec<RecallHit>) -> Vec<RecallHit> {
    let mut seen = HashSet::new();
    hits.into_iter()
        .filter(|hit| {
            let hash = hit.memory.content_sha256.as_str();
            hash.is_empty() || seen.insert(hash.to_string())
        })
        .collect()
}

/// Reciprocal-rank fusion of BM25 and dense ranks. `mnz` is CombMNZ: multiply by
/// the number of lists that ranked the candidate (1 or 2).
pub fn rrf_score(
    bm25_rank: Option<usize>,
    cosine_rank: Option<usize>,
    weight_bm25: f32,
    weight_cosine: f32,
    rrf_k: f32,
    mnz: bool,
) -> f32 {
    let k = rrf_k.max(1.0);
    let mut score = 0.0;
    let mut arms = 0u8;
    if let Some(rank) = bm25_rank {
        score += weight_bm25 / (k + rank as f32);
        arms += 1;
    }
    if let Some(rank) = cosine_rank {
        score += weight_cosine / (k + rank as f32);
        arms += 1;
    }
    if mnz {
        score *= arms.max(1) as f32;
    }
    score
}

/// Fraction of query content tokens that appear in the document, title-weighted.
/// First line is the SciFact title (and a reasonable title for other sources).
/// Not a qrel. Stopwords and tokens shorter than 3 are ignored.
pub fn query_term_coverage(query: &str, text: &str) -> f32 {
    let query_terms = coverage_tokens(query);
    if query_terms.is_empty() {
        return 0.0;
    }
    let title = text.split('\n').next().unwrap_or(text);
    let title_hit = overlap_fraction(&query_terms, title);
    let body_hit = overlap_fraction(&query_terms, text);
    0.65 * title_hit + 0.35 * body_hit
}

fn coverage_tokens(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|tok| tok.len() >= 3)
        .map(|tok| tok.to_ascii_lowercase())
        .filter(|tok| !COVERAGE_STOP.contains(&tok.as_str()))
        .collect()
}

/// Highest query-term overlap of any single sentence. A restated claim beats
/// the same tokens scattered across an abstract.
pub fn max_sentence_coverage(query: &str, text: &str) -> f32 {
    let query_terms = coverage_tokens(query);
    if query_terms.is_empty() {
        return 0.0;
    }
    text.split(|c: char| c == '.' || c == '?' || c == '!' || c == '\n')
        .map(|sentence| overlap_fraction(&query_terms, sentence))
        .fold(0.0_f32, f32::max)
}

fn overlap_fraction(query_terms: &HashSet<String>, text: &str) -> f32 {
    let doc = coverage_tokens(text);
    if query_terms.is_empty() {
        return 0.0;
    }
    let hit = query_terms.iter().filter(|t| doc.contains(*t)).count();
    hit as f32 / query_terms.len() as f32
}

const COVERAGE_STOP: &[&str] = &[
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
    "our", "out", "has", "have", "been", "from", "that", "this", "with", "were", "what", "when",
    "where", "who", "which", "why", "how", "did", "does", "could", "would", "should", "into",
    "than", "then", "them", "they", "their", "there",
];

pub fn rank_map(scores: &HashMap<Uuid, f32>) -> HashMap<Uuid, usize> {
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
    fn scientific_markers_extract_abbrev_needles_not_plain_english() {
        let markers = scientific_marker_tokens(
            "Aspirin inhibits the production of PGE2 and TNF-α via miR7a.",
        );
        assert!(markers.iter().any(|t| t.contains("pge2")), "{markers:?}");
        assert!(
            markers
                .iter()
                .any(|t| t.contains("mir7a") || t.contains("mir")),
            "{markers:?}"
        );
        assert!(
            scientific_marker_tokens("Birth weight is positively associated with breast cancer.")
                .is_empty()
        );
        let needles = marker_needles("tnf-α");
        assert!(needles.iter().any(|n| n.contains("alpha")), "{needles:?}");
    }

    #[test]
    fn combmnz_promotes_both_hit_over_single_arm_top() {
        // BM25-only rank 1 vs both-hit at ranks 15/15. Without MNZ the single arm wins.
        let k = 10.0;
        let single = rrf_score(Some(1), None, 1.3, 1.0, k, false);
        let both = rrf_score(Some(15), Some(15), 1.3, 1.0, k, false);
        assert!(single > both, "plain RRF: {single} vs {both}");
        let single_mnz = rrf_score(Some(1), None, 1.3, 1.0, k, true);
        let both_mnz = rrf_score(Some(15), Some(15), 1.3, 1.0, k, true);
        assert!(
            both_mnz > single_mnz,
            "CombMNZ should flip: both {both_mnz} vs single {single_mnz}"
        );
    }

    #[test]
    fn sentence_coverage_prefers_local_claim_not_scattered_terms() {
        let query = "Aspirin inhibits the production of PGE2.";
        let local =
            "Intro padding. Aspirin inhibits the production of PGE2 in culture. Methods follow.";
        let scattered = "Aspirin is a common analgesic used worldwide. Later we discuss production of lipids. PGE2 is mentioned in passing at the end.";
        let a = max_sentence_coverage(query, local);
        let b = max_sentence_coverage(query, scattered);
        assert!(a > b, "local {a} vs scattered {b}");
        assert!(
            a >= 0.8,
            "claim sentence should cover almost all query tokens, got {a}"
        );
    }

    #[test]
    fn term_coverage_prefers_title_overlap_not_qrels() {
        let query = "Birth-weight is positively associated with breast cancer.";
        let gold =
            "Birth-weight is positively associated with breast cancer.\nA cohort of 12000 women.";
        let decoy = "Urban slang: cancer as in killer beat, associated with the streets.";
        let g = query_term_coverage(query, gold);
        let d = query_term_coverage(query, decoy);
        assert!(g > d, "title-weighted coverage gold {g} vs decoy {d}");
        assert!(
            g > 0.8,
            "claim tokens should mostly hit the matching title, got {g}"
        );
    }

    #[test]
    fn homeostasis_damps_flat_low_confidence_lists() {
        let flat = vec![0.4; 20];
        let clear = vec![0.95, 0.7, 0.5, 0.2];
        assert!(homeostatic_damping(&flat) < homeostatic_damping(&clear));
    }

    #[test]
    fn date_and_speaker_filters_inject_the_small_match_set() {
        use chrono::{TimeZone, Utc};
        let mut dated = MemoryRecord::new("cli-session", "today/1", "yesetday file watchers");
        dated.speaker = Some("human".into());
        dated.timestamp = Some(Utc.with_ymd_and_hms(2026, 8, 20, 4, 16, 25).unwrap());
        let mut old = MemoryRecord::new("gemini-takeout", "old/1", "unrelated soup");
        old.speaker = Some("human".into());
        old.timestamp = Some(Utc.with_ymd_and_hms(2026, 7, 1, 0, 0, 0).unwrap());
        let mut undated = MemoryRecord::new("jsonl", "none/1", "no clock");
        undated.speaker = Some("human".into());
        let records = HashMap::from([
            (dated.id, dated.clone()),
            (old.id, old),
            (undated.id, undated),
        ]);
        let filters = RecallFilters {
            speakers: vec!["human".into()],
            after: Some(Utc.with_ymd_and_hms(2026, 8, 20, 0, 0, 0).unwrap()),
            ..RecallFilters::default()
        };
        assert!(filters_restrict(&filters));
        let ids = matching_ids(&records, &filters);
        assert_eq!(ids, vec![dated.id], "only the dated human turn survives");
        assert!(
            ids.len() <= FILTER_INJECT_CAP,
            "tiny date slices must be injected, not pool-filtered"
        );
        let limit = candidate_limit_for(5, 8, ids.len(), true);
        assert!(
            limit >= FILTER_CANDIDATE_FLOOR,
            "filtered recall widens the ANN/BM25 pool, got {limit}"
        );
    }

    #[test]
    fn basin_filter_injects_tiny_hot_member_set() {
        use crate::geometry::Basin;
        let member = MemoryRecord::new("cli-session", "well/1", "inside the well");
        let outsider = MemoryRecord::new("cli-session", "well/2", "not in the well");
        let records = HashMap::from([(member.id, member.clone()), (outsider.id, outsider.clone())]);
        let filters = RecallFilters {
            basin_id: Some("well-1".into()),
            ..RecallFilters::default()
        };
        assert!(filters_restrict(&filters));
        let cold = matching_ids(&records, &filters);
        assert_eq!(cold.len(), 2, "cold matches() ignores basin");
        let hot = HotState {
            basins: vec![Basin {
                id: "well-1".into(),
                parent_id: None,
                label: String::new(),
                path: String::new(),
                summary: String::new(),
                label_state: String::new(),
                stability: 1.0,
                centroid: [0.0; 3],
                member_ids: vec![member.id],
                representative_ids: Vec::new(),
                updated_at: chrono::Utc::now(),
            }],
            ..HotState::default()
        };
        assert_eq!(basin_inject_ids(&hot, &filters), vec![member.id]);
        assert!(basin_inject_ids(&HotState::default(), &filters).is_empty());
    }

    #[test]
    fn open_search_does_not_scan_every_id() {
        let record = MemoryRecord::new("cli-session", "x", "hello");
        let records = HashMap::from([(record.id, record)]);
        assert!(matching_ids(&records, &RecallFilters::default()).is_empty());
        assert_eq!(candidate_limit_for(5, 8, 0, false), 40);
    }

    #[test]
    fn rare_typo_substring_finds_the_cold_row() {
        let target = MemoryRecord::new(
            "cli-session",
            "live",
            "<user_query>\nyesetday i had you create file watchers\n</user_query>",
        );
        let decoy = MemoryRecord::new("claude", "yes", "yes");
        let records = HashMap::from([(target.id, target.clone()), (decoy.id, decoy)]);
        assert!(rare_token_query("yesetday"));
        assert!(!rare_token_query(
            "what is the meaning of file watchers today"
        ));
        let ids = substring_ids(&records, "yesetday", 16);
        assert_eq!(ids, vec![target.id]);
    }
}
