use crate::record::{RecallFilters, RecallHit};
use crate::service::MemoryService;
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
