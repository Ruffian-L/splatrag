use crate::record::{RecallFilters, RecallHit};
use crate::service::{GeometrySignal, MemoryService};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub const TARGET_RECALL_AT_10: f64 = 0.88;
pub const TARGET_NDCG_AT_10: f64 = 0.75;
pub const MAX_RECALL_DROP: f64 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub queries: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub ndcg_at_k: f64,
}

impl EvalMetrics {
    pub fn meets_targets(&self) -> bool {
        self.recall_at_k >= TARGET_RECALL_AT_10 && self.ndcg_at_k >= TARGET_NDCG_AT_10
    }
}

#[derive(Debug, Deserialize)]
struct QueryRow {
    #[serde(rename = "_id", alias = "id")]
    id: String,
    text: String,
}

pub async fn evaluate_scifact(
    service: &MemoryService,
    dataset_dir: &Path,
    k: usize,
    limit: Option<usize>,
) -> Result<EvalMetrics> {
    let k = k.clamp(1, 100);
    let qrels = load_qrels(&dataset_dir.join("qrels").join("test.tsv"))?;
    let query_path = dataset_dir.join("queries.jsonl");
    let reader = BufReader::new(
        File::open(&query_path)
            .with_context(|| format!("failed to open {}", query_path.display()))?,
    );
    let filters = RecallFilters {
        domains: vec!["scifact".into()],
        ..RecallFilters::default()
    };
    let mut recall_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut evaluated = 0;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let query: QueryRow = serde_json::from_str(&line)
            .with_context(|| format!("invalid query on line {}", line_index + 1))?;
        let Some(relevant) = qrels.get(&query.id) else {
            continue;
        };
        let hits = service.recall(&query.text, k, &filters).await?;
        let retrieved = source_record_ids(&hits);
        let (recall, ndcg) = score_query(&retrieved, relevant, k);
        recall_sum += recall;
        ndcg_sum += ndcg;
        evaluated += 1;
        if evaluated % 25 == 0 {
            eprintln!("evaluated {evaluated} SciFact queries");
        }
        if limit.is_some_and(|limit| evaluated >= limit) {
            break;
        }
    }

    if evaluated == 0 {
        anyhow::bail!(
            "no SciFact queries with relevance judgments found under {}",
            dataset_dir.display()
        );
    }
    Ok(EvalMetrics {
        queries: evaluated,
        k,
        recall_at_k: recall_sum / evaluated as f64,
        ndcg_at_k: ndcg_sum / evaluated as f64,
    })
}

fn source_record_ids(hits: &[RecallHit]) -> Vec<String> {
    hits.iter()
        .filter_map(|hit| hit.memory.source_record_id.clone())
        .collect()
}

fn load_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, u32>>> {
    let reader = BufReader::new(
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?,
    );
    let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() < 3 || columns[0] == "query-id" {
            continue;
        }
        let relevance = columns[2]
            .parse::<u32>()
            .with_context(|| format!("invalid relevance on qrels line {}", line_index + 1))?;
        if relevance > 0 {
            qrels
                .entry(columns[0].to_string())
                .or_default()
                .insert(columns[1].to_string(), relevance);
        }
    }
    Ok(qrels)
}

fn score_query(retrieved: &[String], relevant: &HashMap<String, u32>, k: usize) -> (f64, f64) {
    if relevant.is_empty() {
        return (0.0, 0.0);
    }
    let mut seen = HashSet::new();
    let mut found = 0;
    let mut dcg = 0.0;
    for (rank, document_id) in retrieved.iter().take(k).enumerate() {
        if !seen.insert(document_id) {
            continue;
        }
        if let Some(relevance) = relevant.get(document_id) {
            found += 1;
            dcg += gain(*relevance, rank);
        }
    }
    let mut ideal = relevant.values().copied().collect::<Vec<_>>();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg = ideal
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(rank, relevance)| gain(relevance, rank))
        .sum::<f64>();
    (
        found as f64 / relevant.len() as f64,
        if idcg > 0.0 { dcg / idcg } else { 0.0 },
    )
}

fn gain(relevance: u32, zero_based_rank: usize) -> f64 {
    (2_f64.powi(relevance as i32) - 1.0) / ((zero_based_rank + 2) as f64).log2()
}

/// Why a gold doc missed hybrid top-k (or how hybrid flipped arms).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissClass {
    /// Gold never entered BM25 candidate pool (dense may or may not have it).
    Bm25Miss,
    /// Gold never entered dense (ANN∪Qdrant) pool (BM25 may or may not have it).
    DenseMiss,
    /// Gold missing from *both* arm pools at candidate depth.
    BothMiss,
    /// Gold in at least one arm pool, but hybrid rank > k.
    RankMiss,
    /// Gold is in hybrid top-k (hit).
    Hit,
}

#[derive(Debug, Clone, Serialize)]
pub struct GoldMissRow {
    pub query_id: String,
    pub doc_id: String,
    pub relevance: u32,
    pub class: MissClass,
    pub in_bm25: bool,
    pub in_dense: bool,
    pub hybrid_rank: Option<usize>,
    pub geometry: Option<GeometrySignal>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryMissRow {
    pub query_id: String,
    pub query_text: String,
    pub recall_at_k: f64,
    pub ndcg_at_k: f64,
    /// Primary class for the query: worst missing gold (both > bm25/dense > rank), or hit.
    pub primary_class: MissClass,
    pub golds: Vec<GoldMissRow>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct MissClassCounts {
    pub hit_golds: usize,
    pub bm25_miss: usize,
    pub dense_miss: usize,
    pub both_miss: usize,
    pub rank_miss: usize,
    /// Gold in BM25 only (dense miss but not both)
    pub bm25_only_pool: usize,
    /// Gold in dense only (bm25 miss but not both)
    pub dense_only_pool: usize,
    /// Gold in both pools but rank-miss
    pub both_pool_rank_miss: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MissClassifyReport {
    pub queries: usize,
    pub k: usize,
    pub candidate_limit: usize,
    pub gold_counts: MissClassCounts,
    pub query_primary_counts: HashMap<String, usize>,
    pub failing_queries: usize,
    pub rows: Vec<QueryMissRow>,
}

/// Single free-text flip report (no qrels required): arm membership for hybrid hits + pool sizes.
#[derive(Debug, Clone, Serialize)]
pub struct FlipReport {
    pub query: String,
    pub k: usize,
    pub candidate_limit: usize,
    pub bm25_pool: usize,
    pub dense_pool: usize,
    pub hybrid: Vec<FlipHit>,
    /// Optional gold doc ids (SciFact) if caller supplies them.
    pub golds: Vec<GoldMissRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlipHit {
    pub rank: usize,
    pub memory_id: String,
    pub source_record_id: Option<String>,
    pub in_bm25: bool,
    pub in_dense: bool,
    pub arm: String,
    pub final_score: f32,
    pub bm25: f32,
    pub cosine: f32,
    pub text_preview: String,
    pub geometry: Option<GeometrySignal>,
}

pub fn classify_gold(
    _doc_id: &str,
    in_bm25: bool,
    in_dense: bool,
    hybrid_rank: Option<usize>,
    k: usize,
) -> MissClass {
    if hybrid_rank.is_some_and(|r| r <= k) {
        return MissClass::Hit;
    }
    match (in_bm25, in_dense) {
        (false, false) => MissClass::BothMiss,
        (false, true) => MissClass::Bm25Miss,
        (true, false) => MissClass::DenseMiss,
        (true, true) => MissClass::RankMiss,
    }
}

fn primary_of(golds: &[GoldMissRow]) -> MissClass {
    let mut worst = MissClass::Hit;
    for g in golds {
        worst = match (worst, g.class) {
            (_, MissClass::BothMiss) | (MissClass::BothMiss, _) => MissClass::BothMiss,
            (MissClass::Hit, c) => c,
            (a, MissClass::Hit) => a,
            (MissClass::RankMiss, MissClass::Bm25Miss) => MissClass::Bm25Miss,
            (MissClass::RankMiss, MissClass::DenseMiss) => MissClass::DenseMiss,
            (MissClass::Bm25Miss, MissClass::DenseMiss) => MissClass::Bm25Miss,
            (MissClass::DenseMiss, MissClass::Bm25Miss) => MissClass::DenseMiss,
            (a, _) => a,
        };
    }
    worst
}

/// Full SciFact miss taxonomy over qrels.
pub async fn classify_scifact_misses(
    service: &MemoryService,
    dataset_dir: &Path,
    k: usize,
    candidate_limit: usize,
    limit: Option<usize>,
) -> Result<MissClassifyReport> {
    let k = k.clamp(1, 100);
    let candidate_limit = candidate_limit.clamp(k, 500);
    let qrels = load_qrels(&dataset_dir.join("qrels").join("test.tsv"))?;
    let query_path = dataset_dir.join("queries.jsonl");
    let reader = BufReader::new(
        File::open(&query_path)
            .with_context(|| format!("failed to open {}", query_path.display()))?,
    );
    let filters = RecallFilters {
        domains: vec!["scifact".into()],
        ..RecallFilters::default()
    };

    let mut gold_counts = MissClassCounts::default();
    let mut query_primary_counts: HashMap<String, usize> = HashMap::new();
    let mut rows = Vec::new();
    let mut failing_queries = 0;
    let mut evaluated = 0;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let query: QueryRow = serde_json::from_str(&line)
            .with_context(|| format!("invalid query on line {}", line_index + 1))?;
        let Some(relevant) = qrels.get(&query.id) else {
            continue;
        };

        let arms = service
            .retrieval_arms(&query.text, candidate_limit, k.max(50), &filters)
            .await?;
        let hybrid_rank_by_source: HashMap<String, usize> = arms
            .hybrid
            .iter()
            .enumerate()
            .filter_map(|(i, hit)| {
                hit.memory
                    .source_record_id
                    .clone()
                    .map(|sid| (sid, i + 1))
            })
            .collect();

        let retrieved = source_record_ids(&arms.hybrid);
        let (recall, ndcg) = score_query(&retrieved, relevant, k);

        let mut golds = Vec::new();
        for (doc_id, rel) in relevant {
            let in_bm25 = arms.bm25_source_ids.contains(doc_id);
            let in_dense = arms.dense_source_ids.contains(doc_id);
            let hybrid_rank = hybrid_rank_by_source.get(doc_id).copied();
            let class = classify_gold(doc_id, in_bm25, in_dense, hybrid_rank, k);
            match class {
                MissClass::Hit => gold_counts.hit_golds += 1,
                MissClass::Bm25Miss => {
                    gold_counts.bm25_miss += 1;
                    gold_counts.dense_only_pool += 1;
                }
                MissClass::DenseMiss => {
                    gold_counts.dense_miss += 1;
                    gold_counts.bm25_only_pool += 1;
                }
                MissClass::BothMiss => gold_counts.both_miss += 1,
                MissClass::RankMiss => {
                    gold_counts.rank_miss += 1;
                    if in_bm25 && in_dense {
                        gold_counts.both_pool_rank_miss += 1;
                    }
                }
            }
            golds.push(GoldMissRow {
                query_id: query.id.clone(),
                doc_id: doc_id.clone(),
                relevance: *rel,
                class,
                in_bm25,
                in_dense,
                hybrid_rank,
                geometry: arms.geometry.get(doc_id).cloned(),
            });
        }

        let primary = primary_of(&golds);
        let primary_key = match primary {
            MissClass::Hit => "hit",
            MissClass::Bm25Miss => "bm25_miss",
            MissClass::DenseMiss => "dense_miss",
            MissClass::BothMiss => "both_miss",
            MissClass::RankMiss => "rank_miss",
        };
        *query_primary_counts
            .entry(primary_key.into())
            .or_default() += 1;

        if recall < 1.0 - f64::EPSILON {
            failing_queries += 1;
        }

        rows.push(QueryMissRow {
            query_id: query.id,
            query_text: query.text,
            recall_at_k: recall,
            ndcg_at_k: ndcg,
            primary_class: primary,
            golds,
        });

        evaluated += 1;
        if evaluated % 25 == 0 {
            eprintln!("miss-classify evaluated {evaluated} SciFact queries");
        }
        if limit.is_some_and(|limit| evaluated >= limit) {
            break;
        }
    }

    Ok(MissClassifyReport {
        queries: evaluated,
        k,
        candidate_limit,
        gold_counts,
        query_primary_counts,
        failing_queries,
        rows,
    })
}

/// Flip/flop arm report for one query (optional gold doc ids for SciFact-style class).
pub async fn flip_query(
    service: &MemoryService,
    query: &str,
    k: usize,
    candidate_limit: usize,
    gold_doc_ids: &[(String, u32)],
    domain: Option<&str>,
) -> Result<FlipReport> {
    let k = k.clamp(1, 100);
    let candidate_limit = candidate_limit.clamp(k, 500);
    let filters = RecallFilters {
        domains: domain
            .map(|d| vec![d.into()])
            .unwrap_or_else(|| vec!["scifact".into()]),
        ..RecallFilters::default()
    };
    let arms = service
        .retrieval_arms(query, candidate_limit, k.max(50), &filters)
        .await?;

    let bm25_mem: HashSet<_> = arms.bm25_ids.iter().copied().collect();
    let dense_mem: HashSet<_> = arms.dense_ids.iter().copied().collect();

    let hybrid: Vec<FlipHit> = arms
        .hybrid
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, hit)| {
            let in_bm25 = bm25_mem.contains(&hit.memory.id);
            let in_dense = dense_mem.contains(&hit.memory.id);
            let arm = match (in_bm25, in_dense) {
                (true, true) => "both",
                (true, false) => "bm25_only",
                (false, true) => "dense_only",
                (false, false) => "fusion_only",
            };
            let geometry = hit
                .memory
                .source_record_id
                .as_ref()
                .and_then(|sid| arms.geometry.get(sid).cloned());
            FlipHit {
                rank: i + 1,
                memory_id: hit.memory.id.to_string(),
                source_record_id: hit.memory.source_record_id.clone(),
                in_bm25,
                in_dense,
                arm: arm.into(),
                final_score: hit.scores.final_score,
                bm25: hit.scores.bm25,
                cosine: hit.scores.cosine,
                text_preview: hit.memory.text.chars().take(240).collect(),
                geometry,
            }
        })
        .collect();

    let hybrid_rank_by_source: HashMap<String, usize> = arms
        .hybrid
        .iter()
        .enumerate()
        .filter_map(|(i, hit)| {
            hit.memory
                .source_record_id
                .clone()
                .map(|sid| (sid, i + 1))
        })
        .collect();

    let mut golds = Vec::new();
    for (doc_id, rel) in gold_doc_ids {
        let in_bm25 = arms.bm25_source_ids.contains(doc_id);
        let in_dense = arms.dense_source_ids.contains(doc_id);
        let hybrid_rank = hybrid_rank_by_source.get(doc_id).copied();
        let class = classify_gold(doc_id, in_bm25, in_dense, hybrid_rank, k);
        golds.push(GoldMissRow {
            query_id: String::new(),
            doc_id: doc_id.clone(),
            relevance: *rel,
            class,
            in_bm25,
            in_dense,
            hybrid_rank,
            geometry: arms.geometry.get(doc_id).cloned(),
        });
    }

    Ok(FlipReport {
        query: query.into(),
        k,
        candidate_limit,
        bm25_pool: arms.bm25_source_ids.len().max(arms.bm25_ids.len()),
        dense_pool: arms.dense_source_ids.len().max(arms.dense_ids.len()),
        hybrid,
        golds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_ranking_scores_one() {
        let relevant = HashMap::from([("a".into(), 2), ("b".into(), 1)]);
        let (recall, ndcg) = score_query(&["a".into(), "b".into()], &relevant, 10);
        assert_eq!(recall, 1.0);
        assert!((ndcg - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn duplicate_retrieval_key_is_not_double_counted() {
        let relevant = HashMap::from([("a".into(), 1), ("b".into(), 1)]);
        let (recall, _) = score_query(&["a".into(), "a".into()], &relevant, 10);
        assert_eq!(recall, 0.5);
    }
}
