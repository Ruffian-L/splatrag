use crate::geometry::cosine as geom_cosine;
use crate::record::{MemoryRecord, RecallFilters, RecallHit};
use crate::retrieval::{rank_map, rrf_score};
use crate::service::{GeometrySignal, MemoryService, ScoredId};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use uuid::Uuid;

pub const TARGET_RECALL_AT_10: f64 = 0.88;
pub const TARGET_NDCG_AT_10: f64 = 0.75;
pub const MAX_RECALL_DROP: f64 = 0.05;
/// Filtered-domain freeze from 2026-08-21 (`eval_rrf_full3_w13`, k=10, queries=300).
/// That run set `RecallFilters.domains = ["scifact"]`, so Urban could not occupy top-10.
pub const PUBLISHED_FILTERED_RECALL: f64 = 0.8804444444444445;
pub const PUBLISHED_FILTERED_NDCG: f64 = 0.7623915560726197;

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
            // best-effort progress flush; a flush error is not actionable here
            drop(io::stderr().flush());
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

/// Open-domain SciFact: poison/chat rows may occupy top-k. Qrels still apply only to SciFact golds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltimaDistract {
    pub rank: usize,
    pub domain: String,
    pub source: String,
    pub arm: String,
    pub bm25: f32,
    pub cosine: f32,
    pub near_dup_sig: String,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltimaQueryRow {
    pub query_id: String,
    pub recall_at_k: f64,
    pub ndcg_at_k: f64,
    pub slots_non_scifact: usize,
    pub slots_chat: usize,
    pub slots_urban: usize,
    pub slots_bm25_only: usize,
    pub slots_dense_only: usize,
    pub slots_both_arms: usize,
    pub near_dup_slots: usize,
    pub near_dup_max_cluster: usize,
    pub distractors: Vec<UltimaDistract>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltimaReport {
    pub queries: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub ndcg_at_k: f64,
    pub published_filtered_recall: f64,
    pub published_filtered_ndcg: f64,
    pub delta_recall: f64,
    pub delta_ndcg: f64,
    pub mean_slots_non_scifact: f64,
    pub mean_slots_chat: f64,
    pub mean_slots_urban: f64,
    pub mean_slots_bm25_only: f64,
    pub mean_slots_dense_only: f64,
    pub mean_slots_both_arms: f64,
    pub mean_near_dup_slots: f64,
    pub queries_with_displacement: usize,
    pub queries_with_near_dup: usize,
    pub open_domain: bool,
    pub rows: Vec<UltimaQueryRow>,
}

/// Alphanumeric lowercase prefix. Clustering always returns clusters; this is the cheap
/// external check that near-identical chat rows ate more than one top-k slot.
pub fn near_dup_signature(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .take(80)
        .collect()
}

fn distractor_arm(bm25: f32, cosine: f32) -> &'static str {
    match (bm25 > 0.0, cosine > 0.0) {
        (true, true) => "both",
        (true, false) => "bm25_only",
        (false, true) => "dense_only",
        (false, false) => "neither",
    }
}

fn cluster_near_dups(sigs: &[String]) -> (usize, usize) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for sig in sigs {
        if sig.len() < 16 {
            continue;
        }
        *counts.entry(sig.clone()).or_default() += 1;
    }
    let mut slots = 0usize;
    let mut max_cluster = 1usize;
    for count in counts.values() {
        if *count >= 2 {
            slots += *count;
            max_cluster = max_cluster.max(*count);
        }
    }
    (slots, if slots == 0 { 0 } else { max_cluster })
}

pub async fn evaluate_scifact_ultima(
    service: &MemoryService,
    dataset_dir: &Path,
    k: usize,
    limit: Option<usize>,
) -> Result<UltimaReport> {
    let k = k.clamp(1, 100);
    let qrels = load_qrels(&dataset_dir.join("qrels").join("test.tsv"))?;
    let query_path = dataset_dir.join("queries.jsonl");
    let reader = BufReader::new(
        File::open(&query_path)
            .with_context(|| format!("failed to open {}", query_path.display()))?,
    );
    // Empty filters: Urban and chat compete for top-k. That is the ultima test.
    let filters = RecallFilters::default();
    let mut recall_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut evaluated = 0;
    let mut sum_non = 0.0;
    let mut sum_chat = 0.0;
    let mut sum_urban = 0.0;
    let mut sum_bm25 = 0.0;
    let mut sum_dense = 0.0;
    let mut sum_both = 0.0;
    let mut sum_near = 0.0;
    let mut displaced = 0usize;
    let mut near_dup_queries = 0usize;
    let mut rows = Vec::new();

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

        let mut slots_non = 0usize;
        let mut slots_chat = 0usize;
        let mut slots_urban = 0usize;
        let mut slots_bm25 = 0usize;
        let mut slots_dense = 0usize;
        let mut slots_both = 0usize;
        let mut distractors = Vec::new();
        let mut sigs = Vec::new();
        for (rank, hit) in hits.iter().take(k).enumerate() {
            if hit.memory.domain == "scifact" {
                continue;
            }
            slots_non += 1;
            if hit.memory.domain == "chat" {
                slots_chat += 1;
            } else if hit.memory.domain == "urban" {
                slots_urban += 1;
            }
            let arm = distractor_arm(hit.scores.bm25, hit.scores.cosine);
            match arm {
                "bm25_only" => slots_bm25 += 1,
                "dense_only" => slots_dense += 1,
                "both" => slots_both += 1,
                _ => {}
            }
            let sig = near_dup_signature(&hit.memory.text);
            sigs.push(sig.clone());
            distractors.push(UltimaDistract {
                rank: rank + 1,
                domain: hit.memory.domain.clone(),
                source: hit.memory.source.clone(),
                arm: arm.into(),
                bm25: hit.scores.bm25,
                cosine: hit.scores.cosine,
                near_dup_sig: sig,
                preview: hit.memory.text.chars().take(160).collect(),
            });
        }
        let (near_dup_slots, near_dup_max_cluster) = cluster_near_dups(&sigs);
        if slots_non > 0 {
            displaced += 1;
        }
        if near_dup_slots > 0 {
            near_dup_queries += 1;
        }

        recall_sum += recall;
        ndcg_sum += ndcg;
        sum_non += slots_non as f64;
        sum_chat += slots_chat as f64;
        sum_urban += slots_urban as f64;
        sum_bm25 += slots_bm25 as f64;
        sum_dense += slots_dense as f64;
        sum_both += slots_both as f64;
        sum_near += near_dup_slots as f64;
        evaluated += 1;
        rows.push(UltimaQueryRow {
            query_id: query.id,
            recall_at_k: recall,
            ndcg_at_k: ndcg,
            slots_non_scifact: slots_non,
            slots_chat,
            slots_urban,
            slots_bm25_only: slots_bm25,
            slots_dense_only: slots_dense,
            slots_both_arms: slots_both,
            near_dup_slots,
            near_dup_max_cluster,
            distractors,
        });
        if evaluated % 25 == 0 {
            eprintln!("ultima evaluated {evaluated} SciFact queries (open-domain)");
            // best-effort progress flush; a flush error is not actionable here
            drop(io::stderr().flush());
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
    let n = evaluated as f64;
    let recall_at_k = recall_sum / n;
    let ndcg_at_k = ndcg_sum / n;
    Ok(UltimaReport {
        queries: evaluated,
        k,
        recall_at_k,
        ndcg_at_k,
        published_filtered_recall: PUBLISHED_FILTERED_RECALL,
        published_filtered_ndcg: PUBLISHED_FILTERED_NDCG,
        delta_recall: recall_at_k - PUBLISHED_FILTERED_RECALL,
        delta_ndcg: ndcg_at_k - PUBLISHED_FILTERED_NDCG,
        mean_slots_non_scifact: sum_non / n,
        mean_slots_chat: sum_chat / n,
        mean_slots_urban: sum_urban / n,
        mean_slots_bm25_only: sum_bm25 / n,
        mean_slots_dense_only: sum_dense / n,
        mean_slots_both_arms: sum_both / n,
        mean_near_dup_slots: sum_near / n,
        queries_with_displacement: displaced,
        queries_with_near_dup: near_dup_queries,
        open_domain: true,
        rows,
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolRecall {
    pub gold_recall: f64,
    pub queries_all_golds_in_pool: usize,
    pub mean_pool_size: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HnswGap {
    pub golds: usize,
    pub in_hnsw: usize,
    pub in_exact: usize,
    pub exact_not_hnsw: usize,
    pub hnsw_not_exact: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolesReport {
    pub queries: usize,
    pub k: usize,
    pub candidate_limit: usize,
    pub rerank_pool: usize,
    pub rrf_k: f32,
    pub weight_bm25: f32,
    pub weight_cosine: f32,
    pub weight_full: f32,
    pub exact64: bool,
    pub golds: usize,
    pub shortlist_64d: PoolRecall,
    pub shortlist_bm25: PoolRecall,
    pub shortlist_union: PoolRecall,
    pub shortlist_exact64: Option<PoolRecall>,
    pub rank_64d_only: EvalMetrics,
    pub rank_bm25_only: EvalMetrics,
    pub rank_4096_on_64d_shortlist: EvalMetrics,
    pub rrf_bm25_64d: EvalMetrics,
    pub rrf_bm25_4096: EvalMetrics,
    pub three_arm: EvalMetrics,
    pub hnsw_vs_exact: Option<HnswGap>,
}

fn exact_shortlist(
    query_64: &[f32],
    exact_ids: &[(Uuid, Vec<f32>)],
    candidate_limit: usize,
) -> HashMap<Uuid, f32> {
    let mut scored: Vec<(Uuid, f32)> = exact_ids
        .iter()
        .map(|(id, sem)| (*id, geom_cosine(query_64, sem).max(0.0)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().take(candidate_limit).collect()
}

fn add_hnsw_gap(
    gap: &mut HnswGap,
    relevant: &HashMap<String, u32>,
    pool: &crate::service::RetrievalPool,
    exact_map: &HashMap<Uuid, f32>,
    records: &HashMap<Uuid, MemoryRecord>,
) {
    for sid in relevant.keys() {
        gap.golds += 1;
        let in_h = pool.hnsw64.iter().any(|h| {
            records
                .get(&h.id)
                .and_then(|m| m.source_record_id.as_deref())
                == Some(sid.as_str())
        });
        let in_e = exact_map.keys().any(|id| {
            records.get(id).and_then(|m| m.source_record_id.as_deref()) == Some(sid.as_str())
        });
        if in_h {
            gap.in_hnsw += 1;
        }
        if in_e {
            gap.in_exact += 1;
        }
        if in_e && !in_h {
            gap.exact_not_hnsw += 1;
        }
        if in_h && !in_e {
            gap.hnsw_not_exact += 1;
        }
    }
}

async fn embed_rerank_docs(
    service: &MemoryService,
    pool: &crate::service::RetrievalPool,
    records: &HashMap<Uuid, MemoryRecord>,
    pool_n: usize,
) -> Result<(Vec<Uuid>, HashMap<Uuid, f32>)> {
    let mut embed_ids: HashSet<Uuid> = HashSet::new();
    for hit in pool.hnsw64.iter().take(pool_n) {
        embed_ids.insert(hit.id);
    }
    for hit in pool.bm25.iter().take(pool_n) {
        embed_ids.insert(hit.id);
    }
    let mut embed_ids: Vec<Uuid> = embed_ids.into_iter().collect();
    embed_ids.sort();
    let texts: Vec<String> = embed_ids
        .iter()
        .map(|id| {
            records
                .get(id)
                .map(|m| m.text.chars().take(1000).collect())
                .unwrap_or_default()
        })
        .collect();
    let mut embeddings = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(8) {
        embeddings.extend(service.embed_texts(chunk).await?);
    }
    let mut full_map: HashMap<Uuid, f32> = HashMap::new();
    for (id, emb) in embed_ids.iter().zip(embeddings.iter()) {
        full_map.insert(*id, geom_cosine(&pool.query_full, emb).max(0.0));
    }
    Ok((embed_ids, full_map))
}

fn rrf_pair(
    union: &HashSet<Uuid>,
    bm25_map: &HashMap<Uuid, f32>,
    hnsw_map: &HashMap<Uuid, f32>,
    full_map: &HashMap<Uuid, f32>,
    weight_bm25: f32,
    weight_cosine: f32,
    rrf_k: f32,
) -> (HashMap<Uuid, f32>, HashMap<Uuid, f32>) {
    let bm25_rank = rank_map(bm25_map);
    let hnsw_rank = rank_map(hnsw_map);
    let full_rank = rank_map(full_map);
    let mut rrf64 = HashMap::new();
    let mut rrf4096 = HashMap::new();
    for id in union {
        rrf64.insert(
            *id,
            rrf_score(
                bm25_rank.get(id).copied(),
                hnsw_rank.get(id).copied(),
                weight_bm25,
                weight_cosine,
                rrf_k,
                false,
            ),
        );
        rrf4096.insert(
            *id,
            rrf_score(
                bm25_rank.get(id).copied(),
                full_rank.get(id).copied(),
                weight_bm25,
                weight_cosine,
                rrf_k,
                false,
            ),
        );
    }
    (rrf64, rrf4096)
}

fn three_arm_map(
    rrf64: &HashMap<Uuid, f32>,
    full_map: &HashMap<Uuid, f32>,
    pool_n: usize,
    weight_full: f32,
    rrf_k: f32,
) -> HashMap<Uuid, f32> {
    let mut fused: Vec<(Uuid, f32)> = rrf64.iter().map(|(id, s)| (*id, *s)).collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    fused.truncate(pool_n);
    let mut order: Vec<usize> = (0..fused.len()).collect();
    order.sort_by(|&left, &right| {
        let cl = full_map.get(&fused[left].0).copied().unwrap_or(0.0);
        let cr = full_map.get(&fused[right].0).copied().unwrap_or(0.0);
        cr.total_cmp(&cl)
            .then_with(|| fused[left].0.cmp(&fused[right].0))
    });
    let mut rank_of = vec![0usize; order.len()];
    for (rank, idx) in order.iter().enumerate() {
        rank_of[*idx] = rank + 1;
    }
    let mut three = HashMap::new();
    for (idx, (id, score)) in fused.iter().enumerate() {
        three.insert(*id, *score + weight_full / (rrf_k + rank_of[idx] as f32));
    }
    three
}

/// Separate 64-d candidate generation from 4096-d discrimination and from fusion.
/// Does not change the live ranker; it scores the same pools three ways.
pub async fn evaluate_scifact_roles(
    service: &MemoryService,
    dataset_dir: &Path,
    k: usize,
    candidate_limit: usize,
    limit: Option<usize>,
    exact64: bool,
) -> Result<RolesReport> {
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
    let cfg = &service.config.retrieval;
    let rrf_k = cfg.rrf_k.max(1.0);
    let weight_bm25 = cfg.weight_bm25;
    let weight_cosine = cfg.weight_cosine;
    let weight_full = if cfg.weight_full != 0.0 {
        cfg.weight_full
    } else {
        cfg.weight_cosine
    };
    let rerank_pool = cfg.rerank_pool.max(1);
    let records = service.records_snapshot();
    let hot = service.hot_snapshot();
    let exact_ids: Vec<(Uuid, Vec<f32>)> = if exact64 {
        hot.splats
            .iter()
            .filter(|splat| !splat.muted)
            .filter_map(|splat| {
                let mem = records.get(&splat.memory_id)?;
                if !mem.matches(&filters) {
                    return None;
                }
                Some((splat.memory_id, splat.semantics.clone()))
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut evaluated = 0usize;
    let mut golds_n = 0usize;
    let mut sl_64 = PoolAcc::default();
    let mut sl_bm25 = PoolAcc::default();
    let mut sl_union = PoolAcc::default();
    let mut sl_exact = PoolAcc::default();
    let mut m_64 = MetricAcc::default();
    let mut m_bm25 = MetricAcc::default();
    let mut m_4096_short = MetricAcc::default();
    let mut m_rrf_64 = MetricAcc::default();
    let mut m_rrf_4096 = MetricAcc::default();
    let mut m_three = MetricAcc::default();
    let mut gap = HnswGap::default();

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

        let pool = service
            .retrieval_pool(&query.text, candidate_limit, &filters)
            .await?;

        let bm25_map: HashMap<Uuid, f32> = pool.bm25.iter().map(|h| (h.id, h.score)).collect();
        let hnsw_map: HashMap<Uuid, f32> = pool.hnsw64.iter().map(|h| (h.id, h.score)).collect();
        let mut union: HashSet<Uuid> = bm25_map.keys().copied().collect();
        union.extend(hnsw_map.keys().copied());
        let exact_map = if exact64 {
            exact_shortlist(&pool.query_64, &exact_ids, candidate_limit)
        } else {
            HashMap::new()
        };

        golds_n += relevant.len();
        sl_64.add(
            relevant,
            &source_set(&pool.hnsw64, &records),
            pool.hnsw64.len(),
        );
        sl_bm25.add(relevant, &source_set(&pool.bm25, &records), pool.bm25.len());
        sl_union.add(
            relevant,
            &source_set_ids(union.iter().copied(), &records),
            union.len(),
        );
        if exact64 {
            sl_exact.add(
                relevant,
                &source_set_ids(exact_map.keys().copied(), &records),
                exact_map.len(),
            );
            add_hnsw_gap(&mut gap, relevant, &pool, &exact_map, &records);
        }

        let pool_n = (k.saturating_mul(rerank_pool)).max(k);
        let (embed_ids, full_map) = embed_rerank_docs(service, &pool, &records, pool_n).await?;
        let (rrf64, rrf4096) = rrf_pair(
            &union,
            &bm25_map,
            &hnsw_map,
            &full_map,
            weight_bm25,
            weight_cosine,
            rrf_k,
        );
        let short_full: HashMap<Uuid, f32> = pool
            .hnsw64
            .iter()
            .filter_map(|h| full_map.get(&h.id).map(|s| (h.id, *s)))
            .collect();
        let three = three_arm_map(&rrf64, &full_map, pool_n, weight_full, rrf_k);

        m_64.push(score_ranked(&hnsw_map, &records, relevant, k));
        m_bm25.push(score_ranked(&bm25_map, &records, relevant, k));
        m_4096_short.push(score_ranked(&short_full, &records, relevant, k));
        m_rrf_64.push(score_ranked(&rrf64, &records, relevant, k));
        m_rrf_4096.push(score_ranked(&rrf4096, &records, relevant, k));
        m_three.push(score_ranked(&three, &records, relevant, k));

        evaluated += 1;
        eprintln!(
            "roles evaluated {evaluated} SciFact queries (embedded {})",
            embed_ids.len()
        );
        drop(io::stderr().flush());
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
    let n = evaluated;
    Ok(RolesReport {
        queries: n,
        k,
        candidate_limit,
        rerank_pool,
        rrf_k,
        weight_bm25,
        weight_cosine,
        weight_full,
        exact64,
        golds: golds_n,
        shortlist_64d: sl_64.finish(n),
        shortlist_bm25: sl_bm25.finish(n),
        shortlist_union: sl_union.finish(n),
        shortlist_exact64: if exact64 {
            Some(sl_exact.finish(n))
        } else {
            None
        },
        rank_64d_only: m_64.finish(n, k),
        rank_bm25_only: m_bm25.finish(n, k),
        rank_4096_on_64d_shortlist: m_4096_short.finish(n, k),
        rrf_bm25_64d: m_rrf_64.finish(n, k),
        rrf_bm25_4096: m_rrf_4096.finish(n, k),
        three_arm: m_three.finish(n, k),
        hnsw_vs_exact: if exact64 { Some(gap) } else { None },
    })
}

#[derive(Default)]
struct PoolAcc {
    gold_hits: f64,
    golds: f64,
    all_in: usize,
    pool_size: usize,
}

impl PoolAcc {
    fn add(&mut self, relevant: &HashMap<String, u32>, pool: &HashSet<String>, pool_len: usize) {
        let mut found = 0usize;
        for sid in relevant.keys() {
            self.golds += 1.0;
            if pool.contains(sid) {
                self.gold_hits += 1.0;
                found += 1;
            }
        }
        if found == relevant.len() {
            self.all_in += 1;
        }
        self.pool_size += pool_len;
    }

    fn finish(&self, queries: usize) -> PoolRecall {
        PoolRecall {
            gold_recall: if self.golds > 0.0 {
                self.gold_hits / self.golds
            } else {
                0.0
            },
            queries_all_golds_in_pool: self.all_in,
            mean_pool_size: if queries == 0 {
                0.0
            } else {
                self.pool_size as f64 / queries as f64
            },
        }
    }
}

#[derive(Default)]
struct MetricAcc {
    recall: f64,
    ndcg: f64,
}

impl MetricAcc {
    fn push(&mut self, pair: (f64, f64)) {
        self.recall += pair.0;
        self.ndcg += pair.1;
    }

    fn finish(&self, queries: usize, k: usize) -> EvalMetrics {
        let n = queries.max(1) as f64;
        EvalMetrics {
            queries,
            k,
            recall_at_k: self.recall / n,
            ndcg_at_k: self.ndcg / n,
        }
    }
}

fn source_set(hits: &[ScoredId], records: &HashMap<Uuid, MemoryRecord>) -> HashSet<String> {
    source_set_ids(hits.iter().map(|h| h.id), records)
}

fn source_set_ids(
    ids: impl IntoIterator<Item = Uuid>,
    records: &HashMap<Uuid, MemoryRecord>,
) -> HashSet<String> {
    ids.into_iter()
        .filter_map(|id| records.get(&id).and_then(|m| m.source_record_id.clone()))
        .collect()
}

fn score_ranked(
    scores: &HashMap<Uuid, f32>,
    records: &HashMap<Uuid, MemoryRecord>,
    relevant: &HashMap<String, u32>,
    k: usize,
) -> (f64, f64) {
    let mut items: Vec<(Uuid, f32)> = scores.iter().map(|(id, s)| (*id, *s)).collect();
    items.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let retrieved: Vec<String> = items
        .into_iter()
        .filter_map(|(id, _)| records.get(&id).and_then(|m| m.source_record_id.clone()))
        .collect();
    score_query(&retrieved, relevant, k)
}

pub(crate) fn source_record_ids(hits: &[RecallHit]) -> Vec<String> {
    hits.iter()
        .filter_map(|hit| hit.memory.source_record_id.clone())
        .collect()
}

pub(crate) fn load_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, u32>>> {
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

pub(crate) fn score_query(
    retrieved: &[String],
    relevant: &HashMap<String, u32>,
    k: usize,
) -> (f64, f64) {
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
            .filter_map(|(i, hit)| hit.memory.source_record_id.clone().map(|sid| (sid, i + 1)))
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
        *query_primary_counts.entry(primary_key.into()).or_default() += 1;

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
        .filter_map(|(i, hit)| hit.memory.source_record_id.clone().map(|sid| (sid, i + 1)))
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

    #[test]
    fn near_dup_signature_collapses_case_and_punct() {
        let a = near_dup_signature("Hello, embeddings!!!");
        let b = near_dup_signature("hello embeddings");
        assert_eq!(a, b);
        assert!(a.starts_with("helloembed"));
    }

    #[test]
    fn near_dup_clusters_count_slots_not_groups() {
        let sigs = vec![
            "aaaaaaaaaaaaaaaa".into(),
            "aaaaaaaaaaaaaaaa".into(),
            "aaaaaaaaaaaaaaaa".into(),
            "bbbbbbbbbbbbbbbb".into(),
        ];
        let (slots, max_cluster) = cluster_near_dups(&sigs);
        assert_eq!(slots, 3);
        assert_eq!(max_cluster, 3);
    }
}
