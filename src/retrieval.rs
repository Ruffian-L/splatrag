use crate::ann::KeyedAnn;
use crate::config::RetrievalConfig;
use crate::embedding::{EmbeddingClient, matryoshka64};
use crate::geometry::HotState;
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
        let mut sorted_cosine = cosine_scores.values().copied().collect::<Vec<_>>();
        sorted_cosine.sort_by(|a, b| b.total_cmp(a));
        let damping = homeostatic_damping(&sorted_cosine);
        let effective_radiance_weight = self.config.weight_radiance * damping;

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
            // A candidate found by BM25 alone has no entry here — the ANN never returned it, so its
            // cosine was never *measured*. Treating that absence as 0.0 is not "orthogonal to the
            // query", it is a ~7-point penalty at weight_cosine 10.0 against a true value that runs
            // ~0.7 on this corpus, and it systematically demotes exactly the lexical matches the
            // hybrid exists to include. Measured live: the top two hits for "what did I eat for
            // breakfast" both scored cosine 0.0 with bm25 7.4.
            //
            // The ANN is an approximate index for *finding* candidates. Once one is in hand the
            // exact cosine is a 64-d dot product against semantics we already hold, so compute it.
            let cosine = cosine_scores
                .get(&id)
                .copied()
                .or_else(|| splat.map(|splat| crate::geometry::cosine(&local_query, &splat.semantics)))
                // Only reachable for a memory with no splat, i.e. one that has never been through a
                // dream. There genuinely is no vector to compare, so this 0.0 means unknown.
                .unwrap_or(0.0);
            let final_score = bm25_score * self.config.weight_bm25
                + cosine * self.config.weight_cosine
                + radiance * effective_radiance_weight;
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
