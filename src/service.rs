use crate::ann::KeyedAnn;
use crate::cold_store::ColdStore;
use crate::config::AppConfig;
use crate::embedding::{EmbeddingClient, LabelingClient, matryoshka64};
use crate::geometry::{Basin, HotState};
use crate::inversion::{self, InversionOp, collapse_risk};
use crate::lexical::LexicalIndex;
use crate::packet::{
    CombineReceipt, DEFAULT_COMBINE_T, FieldCentroid, MemoryPacket, VqCodebook, combine,
};
use crate::physics::{DreamReport, domain_state_color, dream, dream_with_progress};
use crate::pick::{MemoryPickSet, PickConfig};
use crate::qdrant::QdrantIndex;
use crate::record::{MemoryRecord, RecallFilters, RecallHit};
use crate::retrieval::RetrievalEngine;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::sync::{Mutex, broadcast};
use uuid::Uuid;

/// Per-document field geometry attached to a SciFact source id (when present in hot).
#[derive(Debug, Clone, Serialize)]
pub struct GeometrySignal {
    pub memory_id: Uuid,
    pub position: [f32; 3],
    pub basin_id: Option<String>,
    pub domain: String,
    pub radiance: f32,
    pub mass: f32,
}

#[derive(Debug, Clone)]
pub struct RetrievalArms {
    pub bm25_ids: Vec<Uuid>,
    pub dense_ids: Vec<Uuid>,
    pub bm25_source_ids: HashSet<String>,
    pub dense_source_ids: HashSet<String>,
    pub hybrid: Vec<RecallHit>,
    pub geometry: HashMap<String, GeometrySignal>,
}

#[derive(Debug, Clone)]
pub struct ScoredId {
    pub id: Uuid,
    pub score: f32,
}

/// Scored BM25 and 64-d HNSW pools plus the query embeddings. Used by role diagnostics;
/// ranking is unchanged.
#[derive(Debug, Clone)]
pub struct RetrievalPool {
    pub query_full: Vec<f32>,
    pub query_64: Vec<f32>,
    pub bm25: Vec<ScoredId>,
    pub hnsw64: Vec<ScoredId>,
}

/// One live-field frame for the SplatLens SSE feed / frame log.
#[derive(Debug, Clone, Serialize)]
pub struct LiveFrame {
    pub step: usize,
    pub kinetic_energy: f32,
    pub basins: usize,
    pub dream_cycle: u64,
    pub dreaming: bool,
    pub done: bool,
    /// Compact positions [x,y,z, …] length = 3 * n_splats (memories view).
    pub positions: Vec<f32>,
    /// RGBA bytes length = 4 * n_splats.
    pub colors: Vec<u8>,
    pub n: usize,
}

pub struct MemoryService {
    pub config: AppConfig,
    cold: ColdStore,
    embedding: EmbeddingClient,
    labeler: LabelingClient,
    qdrant: QdrantIndex,
    lexical: LexicalIndex,
    ann: KeyedAnn,
    records: RwLock<HashMap<Uuid, MemoryRecord>>,
    hot: RwLock<HotState>,
    mutation: Mutex<()>,
    live_tx: broadcast::Sender<LiveFrame>,
    dreaming: AtomicBool,
    /// Optional append-only frame log for disk watching (set via env or serve).
    live_log: RwLock<Option<PathBuf>>,
    /// Byte offset in `cold/memories.jsonl` already loaded into `records`.
    cold_len: AtomicU64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryStatus {
    pub scope_key: String,
    pub cold_records: usize,
    pub hnsw_keys: usize,
    pub qdrant_points: Option<u64>,
    pub splats: usize,
    pub basins: usize,
    pub dream_cycle: u64,
    pub kinetic_energy: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RememberReport {
    pub requested: usize,
    pub appended: usize,
    pub total_records: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackfillReport {
    pub missing: usize,
    pub embedded: usize,
    pub cold_records: usize,
    pub hnsw_keys: usize,
    pub splats: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HydrateReport {
    pub added: usize,
    pub skipped_present: usize,
    pub skipped_bad_dim: usize,
    pub cold_records: usize,
    pub hnsw_keys: usize,
    pub splats: usize,
}

/// Compact basin listing for agents. Full `member_ids` belong in `browse_basin`.
#[derive(Debug, Clone, Serialize)]
pub struct BasinSummary {
    pub id: String,
    pub label: String,
    pub path: String,
    pub summary: String,
    pub label_state: String,
    pub stability: f32,
    pub centroid: [f32; 3],
    pub member_count: usize,
    pub representative_ids: Vec<Uuid>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl From<Basin> for BasinSummary {
    fn from(basin: Basin) -> Self {
        Self {
            id: basin.id,
            label: basin.label,
            path: basin.path,
            summary: basin.summary,
            label_state: basin.label_state,
            stability: basin.stability,
            centroid: basin.centroid,
            member_count: basin.member_ids.len(),
            representative_ids: basin.representative_ids,
            updated_at: basin.updated_at,
        }
    }
}

/// Independent knobs for the steering lane.
///
/// - `gain` → semantic invert/amplify (ontological inversion when negative)
/// - `mass` → if set, physics mass; negative mass **repels** in dream
/// - they are never coupled: neg gain does not set neg mass
#[derive(Debug, Clone)]
pub struct SteerOpts {
    pub gain: f32,
    pub op: InversionOp,
    /// When `Some`, overwrite splat mass (use negative to repel). `None` leaves mass alone.
    pub mass: Option<f32>,
    pub basin_id: Option<String>,
    pub basin_locked: bool,
    /// Concept axis source. Default [`SteerAxisMode::Auto`] (non-self μ).
    pub axis: inversion::SteerAxisMode,
}

impl Default for SteerOpts {
    fn default() -> Self {
        Self {
            gain: 0.0,
            op: InversionOp::Polarity,
            mass: None,
            basin_id: None,
            basin_locked: false,
            axis: inversion::SteerAxisMode::Auto,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SteerReport {
    pub memory_id: Uuid,
    pub gain: f32,
    pub op: InversionOp,
    pub mass: f32,
    pub cosine_before_after: f32,
    pub collapse_risk: bool,
    pub basin_id: Option<String>,
    pub basin_locked: bool,
    /// Where the concept axis came from (`basin_peers`, `field_mean`, `pca0`, `self`, …).
    pub axis_source: String,
    /// Cosine between memory semantics and the axis *before* steering. Near ±1 ⇒ degenerate.
    pub axis_cosine: f32,
}

/// One place to steer. Pin if `memory_id` is set. Room if only `basin_id`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SteerAddress {
    #[serde(default)]
    pub memory_id: Option<Uuid>,
    #[serde(default)]
    pub basin_id: Option<String>,
    /// Fraction of the batch budget. Missing → even split.
    #[serde(default)]
    pub share: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AddressSteerReport {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_id: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub basin_id: Option<String>,
    pub share: f32,
    pub applied_gain: f32,
    pub members_touched: usize,
    pub cosine_before_after: f32,
    pub axis_source: String,
    pub axis_cosine: f32,
    /// Basin-only: the room 64-d after the shove (member mean, not member 0).
    /// Look-only. Pins omit this — use `pack_64` for a pin.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantics_64: Option<Vec<f32>>,
}

/// One call, N addresses, one budget.
#[derive(Debug, Clone, Serialize)]
pub struct SteerBatchReport {
    pub budget: f32,
    pub total_applied: f32,
    pub collapse_risk: bool,
    pub targets: Vec<AddressSteerReport>,
}

impl MemoryService {
    pub fn open(config: AppConfig) -> Result<Self> {
        let cold = ColdStore::open(config.cold_path())?;
        let cold_len = cold.byte_len().unwrap_or(0);
        let records = cold.record_map()?;
        let embedding = EmbeddingClient::new(config.embedding.clone())?;
        let labeler = LabelingClient::new(config.labeling.clone())?;
        let qdrant = QdrantIndex::new(
            config.qdrant.clone(),
            config.scope_key.clone(),
            config.embedding.dimensions,
        )?;
        let lexical = LexicalIndex::open(config.lexical_path())?;
        let ann = KeyedAnn::open(config.ann_graph_path(), 64, &config.retrieval)?;
        let hot = HotState::load(&config.hot_state_path())?;
        let (live_tx, _) = broadcast::channel(16);
        Ok(Self {
            config,
            cold,
            embedding,
            labeler,
            qdrant,
            lexical,
            ann,
            records: RwLock::new(records),
            hot: RwLock::new(hot),
            mutation: Mutex::new(()),
            live_tx,
            dreaming: AtomicBool::new(false),
            live_log: RwLock::new(None),
            cold_len: AtomicU64::new(cold_len),
        })
    }

    pub fn set_live_log_path(&self, path: Option<PathBuf>) {
        *self.live_log.write().expect("live_log lock") = path;
    }

    pub fn subscribe_live(&self) -> broadcast::Receiver<LiveFrame> {
        self.live_tx.subscribe()
    }

    pub fn is_dreaming(&self) -> bool {
        self.dreaming.load(Ordering::Relaxed)
    }

    pub fn current_live_frame(&self) -> LiveFrame {
        let ke = self.hot_read().kinetic_energy;
        self.snapshot_live_frame(0, ke, self.is_dreaming(), false)
    }

    /// Recolor the hot field by basin for static viewing (no physics).
    /// Personal stores are almost all one domain; basin hue is what makes clusters readable.
    pub fn recolor_field(&self) -> Result<()> {
        let mut hot = self.hot_write();
        for splat in &mut hot.splats {
            splat.color_rgba = domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0);
        }
        drop(hot);
        self.persist_hot_and_ann()?;
        let frame = self.snapshot_live_frame(0, self.hot_read().kinetic_energy, false, false);
        let _ = self.live_tx.send(frame);
        Ok(())
    }

    fn snapshot_live_frame(
        &self,
        step: usize,
        kinetic_energy: f32,
        dreaming: bool,
        done: bool,
    ) -> LiveFrame {
        live_frame_from(&self.hot_read(), step, kinetic_energy, dreaming, done)
    }

    fn emit_live_frame(&self, frame: LiveFrame) {
        if let Ok(guard) = self.live_log.read()
            && let Some(path) = guard.as_ref()
            && let Ok(line) = serde_json::to_string(&serde_json::json!({
                "step": frame.step,
                "kinetic_energy": frame.kinetic_energy,
                "basins": frame.basins,
                "dream_cycle": frame.dream_cycle,
                "dreaming": frame.dreaming,
                "done": frame.done,
                "n": frame.n,
            }))
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = writeln!(f, "{line}");
            }
        }
        let _ = self.live_tx.send(frame);
    }

    pub async fn initialize(&self) -> Result<()> {
        self.qdrant.validate().await?;
        self.qdrant.ensure_scope_index().await?;
        self.persist_hot_and_ann()
    }

    pub async fn reindex_existing(&self) -> Result<RememberReport> {
        let _guard = self.mutation.lock().await;
        self.qdrant.validate().await?;
        self.qdrant.ensure_scope_index().await?;
        let mut records = self.records_read().values().cloned().collect::<Vec<_>>();
        records.sort_by_key(|record| record.id);
        let requested = records.len();
        let batch_size = self.embedding.batch_size();
        for batch in records.chunks(batch_size) {
            let texts = batch
                .iter()
                .map(|record| record.text.clone())
                .collect::<Vec<_>>();
            let embeddings = self.embedding.embed_batch(&texts).await?;
            self.qdrant.upsert(batch, &embeddings).await?;
            for (record, embedding) in batch.iter().zip(&embeddings) {
                self.ann.set(record.id, &matryoshka64(embedding)?)?;
            }
            self.lexical.add_records(batch)?;
            self.hot_write().add_embeddings(batch, &embeddings)?;
            eprintln!("reindexed {}/{} memories", self.ann.len(), requested);
        }
        self.persist_hot_and_ann()?;
        Ok(RememberReport {
            requested,
            appended: 0,
            total_records: requested,
        })
    }

    pub async fn remember(&self, record: MemoryRecord) -> Result<RememberReport> {
        self.remember_batch(vec![record]).await
    }

    pub async fn remember_batch(&self, records: Vec<MemoryRecord>) -> Result<RememberReport> {
        let _guard = self.mutation.lock().await;
        let requested = records.len();
        let new_records = {
            let known = self.records_read();
            let mut seen = HashSet::with_capacity(known.len() + records.len());
            seen.extend(known.keys().copied());
            records
                .into_iter()
                .filter(|record| seen.insert(record.id))
                .collect::<Vec<_>>()
        };
        if new_records.is_empty() {
            return Ok(RememberReport {
                requested,
                appended: 0,
                total_records: self.records_read().len(),
            });
        }
        let texts = new_records
            .iter()
            .map(|record| record.text.clone())
            .collect::<Vec<_>>();
        let mut embeddings = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.embedding.batch_size()) {
            embeddings.extend(self.embedding.embed_batch(batch).await?);
        }

        let appended = self.cold.append_new(&new_records)?;
        if appended != new_records.len() {
            anyhow::bail!(
                "cold-store concurrency mismatch: prepared {}, appended {}",
                new_records.len(),
                appended
            );
        }
        if let Ok(len) = self.cold.byte_len() {
            self.cold_len.store(len, Ordering::Relaxed);
        }
        self.qdrant.upsert(&new_records, &embeddings).await?;
        for (record, embedding) in new_records.iter().zip(&embeddings) {
            self.ann.set(record.id, &matryoshka64(embedding)?)?;
        }
        self.lexical.add_records(&new_records)?;
        {
            let mut hot = self.hot_write();
            hot.add_embeddings(&new_records, &embeddings)?;
        }
        {
            let mut map = self.records_write();
            for record in new_records {
                map.insert(record.id, record);
            }
        }
        self.persist_hot_and_ann()?;
        Ok(RememberReport {
            requested,
            appended,
            total_records: self.records_read().len(),
        })
    }

    /// Pull rows another process appended to cold, and refresh the BM25 reader.
    ///
    /// Sidecar `splatrag ingest` and a long-lived MCP used to diverge: Tantivy
    /// could rank a new id, then `records.get` dropped it. Call this before recall.
    pub fn refresh_from_cold(&self) -> Result<usize> {
        let offset = self.cold_len.load(Ordering::Relaxed);
        let (new_offset, new_records, rewritten) = self.cold.read_new_since(offset)?;
        let mut added = 0;
        {
            let mut map = self.records_write();
            if rewritten {
                added = new_records.len();
                *map = new_records
                    .into_iter()
                    .map(|record| (record.id, record))
                    .collect();
            } else {
                for record in new_records {
                    if map.insert(record.id, record).is_none() {
                        added += 1;
                    }
                }
            }
        }
        self.cold_len.store(new_offset, Ordering::Relaxed);
        self.lexical.reload()?;
        Ok(added)
    }

    pub async fn ingest_path(
        &self,
        source: crate::ingest::SourceKind,
        path: PathBuf,
        domain: String,
    ) -> Result<(crate::ingest::IngestReport, usize)> {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(256);
        let quarantine = self.config.quarantine_path();
        let parser = tokio::task::spawn_blocking(move || {
            crate::ingest::Ingestor::new(quarantine).ingest_path(source, &path, &domain, |record| {
                sender
                    .blocking_send(record)
                    .context("ingest receiver closed")
            })
        });
        let mut batch = Vec::new();
        let mut appended = 0;
        while let Some(record) = receiver.recv().await {
            batch.push(record);
            if batch.len() >= self.embedding.batch_size() {
                appended += self
                    .remember_batch(std::mem::take(&mut batch))
                    .await?
                    .appended;
            }
        }
        if !batch.is_empty() {
            appended += self.remember_batch(batch).await?.appended;
        }
        Ok((parser.await??, appended))
    }

    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        filters: &RecallFilters,
    ) -> Result<Vec<RecallHit>> {
        self.refresh_from_cold()?;
        let records = self.records_read().clone();
        let hot = self.hot_read().clone();
        RetrievalEngine {
            embedding: &self.embedding,
            qdrant: &self.qdrant,
            ann: &self.ann,
            lexical: &self.lexical,
            config: &self.config.retrieval,
        }
        .recall(query, limit, filters, &records, &hot)
        .await
    }

    /// Separate retrieval arms for miss classification (BM25 vs dense vs hybrid rank).
    ///
    /// - `bm25_ids` / `dense_ids`: memory UUIDs in each arm's top-`candidate_limit`
    /// - `hybrid`: full hybrid recall ranked list (length `hybrid_limit`)
    /// - dense = local HNSW ∪ Qdrant (when enabled)
    pub async fn retrieval_arms(
        &self,
        query: &str,
        candidate_limit: usize,
        hybrid_limit: usize,
        filters: &RecallFilters,
    ) -> Result<RetrievalArms> {
        let candidate_limit = candidate_limit.clamp(1, 500);
        let hybrid_limit = hybrid_limit.clamp(1, 100);
        self.refresh_from_cold()?;
        let records = self.records_read().clone();
        let hot = self.hot_read().clone();

        let lexical = self.lexical.search(query, candidate_limit)?;
        let full_query = self.embedding.embed(query).await?;
        let local_query = matryoshka64(&full_query)?;
        let local_dense = self.ann.search(&local_query, candidate_limit)?;
        let qdrant_dense = self.qdrant.search(&full_query, candidate_limit).await?;

        let mut bm25_ids = Vec::new();
        let mut bm25_source = HashMap::new();
        for hit in &lexical {
            if let Some(mem) = records.get(&hit.id) {
                if !mem.matches(filters) {
                    continue;
                }
                bm25_ids.push(hit.id);
                if let Some(sid) = &mem.source_record_id {
                    bm25_source.insert(sid.clone(), hit.id);
                }
            }
        }

        let mut dense_ids = Vec::new();
        let mut dense_seen = HashSet::new();
        let mut dense_source = HashMap::new();
        let mut push_dense = |id: Uuid| {
            if !dense_seen.insert(id) {
                return;
            }
            if let Some(mem) = records.get(&id) {
                if !mem.matches(filters) {
                    return;
                }
                dense_ids.push(id);
                if let Some(sid) = &mem.source_record_id {
                    dense_source.insert(sid.clone(), id);
                }
            }
        };
        for hit in &local_dense {
            push_dense(hit.id);
        }
        for hit in &qdrant_dense {
            push_dense(hit.id);
        }

        let hybrid = RetrievalEngine {
            embedding: &self.embedding,
            qdrant: &self.qdrant,
            ann: &self.ann,
            lexical: &self.lexical,
            config: &self.config.retrieval,
        }
        .recall(query, hybrid_limit, filters, &records, &hot)
        .await?;

        // Geometry for residual signal: 3d position + basin from hot field.
        let splat_map = hot.splat_map();
        let mut geometry = HashMap::new();
        for mem in records.values() {
            if let Some(sid) = &mem.source_record_id
                && let Some(splat) = splat_map.get(&mem.id)
            {
                geometry.insert(
                    sid.clone(),
                    GeometrySignal {
                        memory_id: mem.id,
                        position: splat.position,
                        basin_id: splat.basin_id.clone(),
                        domain: splat.domain.clone(),
                        radiance: splat.radiance,
                        mass: splat.mass,
                    },
                );
            }
        }

        Ok(RetrievalArms {
            bm25_ids,
            dense_ids,
            bm25_source_ids: bm25_source.keys().cloned().collect(),
            dense_source_ids: dense_source.keys().cloned().collect(),
            hybrid,
            geometry,
        })
    }

    /// BM25 and 64-d HNSW candidate lists with scores. Does not run fusion or 4096-d rerank.
    pub async fn retrieval_pool(
        &self,
        query: &str,
        candidate_limit: usize,
        filters: &RecallFilters,
    ) -> Result<RetrievalPool> {
        let candidate_limit = candidate_limit.clamp(1, 500);
        self.refresh_from_cold()?;
        let records = self.records_read();
        let lexical = self.lexical.search(query, candidate_limit)?;
        let full_query = self.embedding.embed(query).await?;
        let local_query = matryoshka64(&full_query)?;
        let local_dense = self.ann.search(&local_query, candidate_limit)?;

        let mut bm25 = Vec::new();
        for hit in lexical {
            if let Some(mem) = records.get(&hit.id)
                && mem.matches(filters)
            {
                bm25.push(ScoredId {
                    id: hit.id,
                    score: hit.score,
                });
            }
        }
        let mut hnsw64 = Vec::new();
        for hit in local_dense {
            if let Some(mem) = records.get(&hit.id)
                && mem.matches(filters)
            {
                hnsw64.push(ScoredId {
                    id: hit.id,
                    score: (1.0 - hit.distance).max(0.0),
                });
            }
        }
        Ok(RetrievalPool {
            query_full: full_query,
            query_64: local_query,
            bm25,
            hnsw64,
        })
    }

    pub async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embedding.embed_batch(texts).await
    }

    pub fn records_snapshot(&self) -> HashMap<Uuid, MemoryRecord> {
        self.records_read().clone()
    }

    /// Pick memories for a live steering host, with the knobs to apply them.
    ///
    /// Deliberately over-fetches to a depth that does **not** track `limit`: `separation` needs
    /// candidates below the cut to judge whether the top hit is distinctly right, and a pool that
    /// grew with `limit` would make confidence a function of how many picks were requested rather
    /// than of the query. Asking for more memories must not change how confident we are.
    pub async fn pick(
        &self,
        prompt: &str,
        filters: &RecallFilters,
        config: &PickConfig,
    ) -> Result<MemoryPickSet> {
        let depth = config.limit.max(1).max(crate::pick::CONFIDENCE_POOL);
        let mut hits = self.recall(prompt, depth, filters).await?;
        let hybrid_cfg = crate::hybrid::HybridConfig::for_pick(config.hybrid_beta, config.easyway);
        // One embed: null model + hybrid fill of ANN-miss cosine. Recall already
        // embedded internally; this is the copy pick keeps.
        let query_64 = crate::embedding::matryoshka64(&self.embedding.embed(prompt).await?)?;
        if config.hybrid {
            let hot = self.hot_read();
            crate::hybrid::fill_missing_cosine(&mut hits, &hot.splats, &query_64);
            crate::hybrid::rerank_hits(&mut hits, &hot.splats, &hybrid_cfg);
        }
        let hot = self.hot_read().clone();
        let null =
            crate::pick::NullModel::estimate(&query_64, &hot.splats, crate::pick::NULL_SAMPLES);
        let set = crate::pick::build(
            prompt,
            &hits,
            &hot,
            &self.config.embedding.model,
            null,
            config,
        );
        // Persist only when asked. `--hybrid` alone is a read-only cage check.
        if config.track_usage {
            let _guard = self.mutation.lock().await;
            {
                let mut hot = self.hot_write();
                if config.hybrid {
                    crate::hybrid::apply_step(&mut hot.splats, &hits, &hybrid_cfg);
                } else if let Some(top) = set.picks.first() {
                    if let Some(splat) = hot
                        .splats
                        .iter_mut()
                        .find(|splat| splat.memory_id == top.memory_id)
                    {
                        crate::hybrid::bump_usage(splat, hybrid_cfg.delta_tau);
                    }
                }
            }
            self.persist_hot_and_ann()?;
        }
        Ok(set)
    }

    pub async fn dream(&self) -> Result<DreamReport> {
        let _guard = self.mutation.lock().await;
        self.collapse_duplicates_if_enabled()?;
        // Clone the field, then drop the hot write lock so status/splats keep
        // serving the last committed picture while the force loop runs.
        let mut work = self.hot_read().clone();
        let report = dream(&mut work, &self.ann, &self.config.physics)?;
        {
            let mut hot = self.hot_write();
            *hot = work;
        }
        self.persist_hot_and_ann()?;
        Ok(report)
    }

    /// Garbage pass: mute byte-identical copies so the physics runs on one of each.
    ///
    /// Runs before every dream, reads cold read-only, and writes nothing but hot flags.
    /// Idempotent — a second pass finds the groups already collapsed and mutes nothing.
    fn collapse_duplicates_if_enabled(&self) -> Result<()> {
        if !self.config.physics.dedupe_duplicates {
            return Ok(());
        }
        let records = self.cold.record_map()?;
        let report = {
            let mut hot = self.hot_write();
            crate::dedupe::collapse_duplicates(&mut hot, &records)
        };
        if report.muted > 0 || report.already_muted > 0 {
            eprintln!(
                "dream: duplicate collapse — {} groups, {} muted this pass ({} already), \
                 largest group {}, {} active splats",
                report.groups,
                report.muted,
                report.already_muted,
                report.largest_group,
                report.active
            );
        }
        Ok(())
    }

    /// Dream with live frames every `emit_every` steps. Publishes to SSE subscribers.
    /// Blocks until the dream finishes. Only one dream at a time.
    pub async fn dream_live(&self, emit_every: usize) -> Result<DreamReport> {
        if self
            .dreaming
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            anyhow::bail!("dream already running");
        }
        let emit_every = emit_every.max(1);
        let _guard = self.mutation.lock().await;
        if let Err(error) = self.collapse_duplicates_if_enabled() {
            self.dreaming.store(false, Ordering::SeqCst);
            return Err(error);
        }
        // Physics is O(N²) CPU; keep the async runtime free for SSE while we run.
        let result = tokio::task::block_in_place(|| self.dream_live_inner(emit_every));
        self.dreaming.store(false, Ordering::SeqCst);
        result
    }

    fn dream_live_inner(&self, emit_every: usize) -> Result<DreamReport> {
        let mut work = {
            let mut hot = self.hot_write();
            for splat in &mut hot.splats {
                splat.color_rgba =
                    domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0);
            }
            hot.clone()
        };
        self.emit_live_frame(live_frame_from(&work, 0, work.kinetic_energy, true, false));

        let tx = self.live_tx.clone();
        let log_path = self.live_log.read().ok().and_then(|g| g.clone());
        let report = dream_with_progress(
            &mut work,
            &self.ann,
            &self.config.physics,
            emit_every,
            |step, energy, hot| {
                let frame = live_frame_from(hot, step, energy, true, false);
                if let Some(path) = &log_path
                    && let Ok(line) = serde_json::to_string(&serde_json::json!({
                        "step": frame.step,
                        "kinetic_energy": frame.kinetic_energy,
                        "basins": frame.basins,
                        "n": frame.n,
                    }))
                {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                    {
                        let _ = writeln!(f, "{line}");
                    }
                }
                let _ = tx.send(frame);
            },
        )?;
        {
            let mut hot = self.hot_write();
            *hot = work;
        }
        self.persist_hot_and_ann()?;
        let mut done = self.snapshot_live_frame(report.steps, report.kinetic_energy, false, true);
        done.basins = report.basins;
        self.emit_live_frame(done);
        Ok(report)
    }

    /// Name basins with the local labeler (original SplatRAG contract).
    ///
    /// Labels are **derived cosmetic metadata** — not the meaning engine. Evidence is the
    /// radiance-ranked `representative_ids` from basin discovery. `force` re-labels stable basins.
    /// See `research_logs/2026-08-07_basins_labels_and_5k_bags.md`.
    pub async fn label_basins(&self) -> Result<usize> {
        self.label_basins_with_options(false).await
    }

    pub async fn label_basins_with_options(&self, force: bool) -> Result<usize> {
        if !self.labeler.enabled() {
            return Ok(0);
        }
        let _guard = self.mutation.lock().await;
        let records = self.records_read().clone();
        let pending = self
            .hot_read()
            .basins
            .iter()
            .filter(|basin| force || basin.label_state == "pending")
            .filter(|basin| basin.label_state != "locked")
            .map(|basin| {
                (
                    basin.id.clone(),
                    basin
                        .representative_ids
                        .iter()
                        .filter_map(|id| records.get(id).map(|record| record.text.clone()))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let mut labels = Vec::new();
        for (id, representatives) in pending {
            if representatives.is_empty() {
                continue;
            }
            // Labels are derived, cosmetic metadata. A basin the local model fails to name stays
            // `pending` and is retried next pass — one bad response must not discard the labels
            // already earned in this run, nor fail a dream whose physics has already landed.
            match self.labeler.label_basin(&representatives).await {
                Ok(draft) => labels.push((id, draft)),
                Err(error) => eprintln!("basin {id} left unlabeled: {error:#}"),
            }
        }
        let count = labels.len();
        {
            let mut hot = self.hot_write();
            for (id, draft) in labels {
                if let Some(basin) = hot.basins.iter_mut().find(|basin| basin.id == id) {
                    basin.label = draft.label;
                    basin.path = draft.path;
                    basin.summary = draft.summary;
                    basin.label_state = "stable".into();
                }
            }
        }
        self.persist_hot_and_ann()?;
        Ok(count)
    }

    pub fn list_basins(&self) -> Vec<Basin> {
        let mut basins = self.hot_read().basins.clone();
        basins.sort_by(|a, b| {
            b.member_ids
                .len()
                .cmp(&a.member_ids.len())
                .then_with(|| a.id.cmp(&b.id))
        });
        basins
    }

    /// Agent-safe basin list: counts instead of every member id.
    /// The giant unlabeled basin on personal is ~60k ids and blows MCP context.
    pub fn list_basin_summaries(&self) -> Vec<BasinSummary> {
        self.list_basins()
            .into_iter()
            .map(BasinSummary::from)
            .collect()
    }

    /// Embed any cold records missing from HNSW / hot geometry. Does not wipe dream state.
    pub async fn backfill_missing_derived(&self) -> Result<BackfillReport> {
        let _guard = self.mutation.lock().await;
        let ann_keys: HashSet<Uuid> = self.ann.keys().into_iter().collect();
        let hot_keys: HashSet<Uuid> = self
            .hot_read()
            .splats
            .iter()
            .map(|splat| splat.memory_id)
            .collect();
        let mut missing = self
            .records_read()
            .values()
            .filter(|record| !ann_keys.contains(&record.id) || !hot_keys.contains(&record.id))
            .cloned()
            .collect::<Vec<_>>();
        missing.sort_by_key(|record| record.id);
        let missing_count = missing.len();
        if missing.is_empty() {
            return Ok(BackfillReport {
                missing: 0,
                embedded: 0,
                cold_records: self.records_read().len(),
                hnsw_keys: self.ann.len(),
                splats: self.hot_read().splats.len(),
            });
        }
        self.qdrant.validate().await?;
        self.qdrant.ensure_scope_index().await?;
        let mut embedded = 0;
        let batch_size = self.embedding.batch_size();
        for batch in missing.chunks(batch_size) {
            let texts = batch
                .iter()
                .map(|record| {
                    if record.text.trim().is_empty() {
                        format!("[empty] {}", record.source_key)
                    } else {
                        record.text.clone()
                    }
                })
                .collect::<Vec<_>>();
            let embeddings = self.embedding.embed_batch(&texts).await?;
            self.qdrant.upsert(batch, &embeddings).await?;
            for (record, embedding) in batch.iter().zip(&embeddings) {
                self.ann.set(record.id, &matryoshka64(embedding)?)?;
            }
            self.lexical.add_records(batch)?;
            self.hot_write().add_embeddings(batch, &embeddings)?;
            embedded += batch.len();
            eprintln!("backfill-derived {}/{} missing", embedded, missing_count);
        }
        self.persist_hot_and_ann()?;
        Ok(BackfillReport {
            missing: missing_count,
            embedded,
            cold_records: self.records_read().len(),
            hnsw_keys: self.ann.len(),
            splats: self.hot_read().splats.len(),
        })
    }

    /// Insert 64-d vectors already on hot splats into HNSW. No embedder.
    /// Used to mix a second store's cartography arm without re-embedding 60k chats.
    pub async fn hydrate_ann_from_hot(&self) -> Result<HydrateReport> {
        let _guard = self.mutation.lock().await;
        let mut known: HashSet<Uuid> = self.ann.keys().into_iter().collect();
        let hot = self.hot_read();
        let mut added = 0usize;
        let mut skipped_present = 0usize;
        let mut skipped_bad_dim = 0usize;
        for splat in &hot.splats {
            if !known.insert(splat.memory_id) {
                skipped_present += 1;
                continue;
            }
            if splat.semantics.len() != 64 {
                skipped_bad_dim += 1;
                continue;
            }
            self.ann.set(splat.memory_id, &splat.semantics)?;
            added += 1;
            if added % 5000 == 0 {
                eprintln!("hydrate-ann {added} new keys");
            }
        }
        drop(hot);
        self.ann.save()?;
        Ok(HydrateReport {
            added,
            skipped_present,
            skipped_bad_dim,
            cold_records: self.records_read().len(),
            hnsw_keys: self.ann.len(),
            splats: self.hot_read().splats.len(),
        })
    }

    pub fn browse_basin(&self, basin_id: &str, offset: usize, limit: usize) -> Vec<MemoryRecord> {
        let hot = self.hot_read();
        let Some(basin) = hot.basins.iter().find(|basin| basin.id == basin_id) else {
            return Vec::new();
        };
        let records = self.records_read();
        basin
            .member_ids
            .iter()
            .skip(offset)
            .take(limit.clamp(1, 500))
            .filter_map(|id| records.get(id).cloned())
            .collect()
    }

    pub async fn status(&self) -> Result<MemoryStatus> {
        let qdrant_points = if self.qdrant.enabled() {
            Some(self.qdrant.count_scope().await?)
        } else {
            None
        };
        let hot = self.hot_read();
        Ok(MemoryStatus {
            scope_key: self.config.scope_key.clone(),
            cold_records: self.records_read().len(),
            hnsw_keys: self.ann.len(),
            qdrant_points,
            splats: hot.splats.len(),
            basins: hot.basins.len(),
            dream_cycle: hot.dream_cycle,
            kinetic_energy: hot.kinetic_energy,
        })
    }

    pub async fn doctor(&self) -> Result<MemoryStatus> {
        self.embedding.doctor().await?;
        self.qdrant.validate().await?;
        let status = self.status().await?;
        if status.cold_records != status.hnsw_keys || status.cold_records != status.splats {
            anyhow::bail!(
                "derived index drift: cold={}, hnsw={}, splats={}; run backfill-derived (rebuild-index wipes dream/hot state)",
                status.cold_records,
                status.hnsw_keys,
                status.splats
            );
        }
        if let Some(qdrant) = status.qdrant_points
            && qdrant != status.cold_records as u64
        {
            anyhow::bail!(
                "Qdrant scope drift: cold={}, qdrant={}; run rebuild-index",
                status.cold_records,
                qdrant
            );
        }
        Ok(status)
    }

    pub fn hot_snapshot(&self) -> HotState {
        self.hot_read().clone()
    }

    pub fn record(&self, id: Uuid) -> Option<MemoryRecord> {
        self.records_read().get(&id).cloned()
    }

    /// Pack one splat into a 64D memory packet (raw floats + optional Unicode VQ).
    pub fn pack_packet(&self, id: Uuid, codebook: Option<&VqCodebook>) -> Result<MemoryPacket> {
        let hot = self.hot_read();
        let splat = hot
            .splats
            .iter()
            .find(|splat| splat.memory_id == id)
            .with_context(|| format!("no splat for memory {id}"))?;
        let mut packet = MemoryPacket::from_semantics(
            Some(id),
            &splat.semantics,
            splat.gain,
            splat.mass,
            splat.basin_id.clone(),
            splat.basin_locked,
        )?;
        if let Some(cb) = codebook {
            packet = packet.with_unicode(cb)?;
        }
        Ok(packet)
    }

    /// Combine several splats into one 64D packet, off the generic centre of the store.
    ///
    /// Read-only. This reads hot state and hands back a packet plus its receipt; nothing is
    /// written. Applying the result is [`Self::unpack_packet`]'s job and a separate decision, on
    /// purpose — a combined state is a hypothesis until someone chooses to store it.
    ///
    /// `t` defaults to [`DEFAULT_COMBINE_T`]. See [`combine`] for what the knob was measured to do.
    pub fn combine_packet(&self, ids: &[Uuid], t: Option<f32>) -> Result<CombineReceipt> {
        if ids.is_empty() {
            bail!("combine needs at least one memory_id");
        }
        let t = t.unwrap_or(DEFAULT_COMBINE_T);
        let hot = self.hot_read();

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
        for id in ids {
            let splat = hot
                .splats
                .iter()
                .find(|splat| splat.memory_id == *id)
                .with_context(|| format!("no splat for memory {id}"))?;
            vectors.push(splat.semantics.clone());
        }

        // Strided the same way the null model is, and for the same reason: a centroid that moved
        // with scan depth would make the knob mean something different on every call.
        let samples = crate::pick::NULL_SAMPLES;
        let stride = (hot.splats.len() / samples.max(1)).max(1);
        let sampled: Vec<&[f32]> = hot
            .splats
            .iter()
            .step_by(stride)
            .take(samples)
            .map(|splat| splat.semantics.as_slice())
            .collect();
        let centroid = FieldCentroid::estimate(&sampled)
            .context("field has no measurable centroid — is the store settled?")?;

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let naive = combine(&refs, &centroid, 0.0)?;
        let combined = combine(&refs, &centroid, t)?;

        // A combined state has never been through a dream: no gain, neutral mass, no basin. Same
        // reasoning as the pick path — zero mass would silently delete it from the force law.
        let packet = MemoryPacket::from_semantics(None, &combined, 0.0, 1.0, None, false)?;
        Ok(CombineReceipt {
            packet,
            t,
            inputs: ids.to_vec(),
            centroid_samples: centroid.samples,
            cosine_to_centroid_naive: crate::packet::cosine(&naive, &centroid.axis),
            cosine_to_centroid: crate::packet::cosine(&combined, &centroid.axis),
        })
    }

    /// Apply a packet onto an existing splat (by packet.memory_id or override id).
    ///
    /// Writes 64D semantics into hot + HNSW. Gain/mass/basin from the packet.
    /// Cold text is never rewritten. Unicode is only used if raw semantics are absent.
    /// Same `mutation.lock()` as steer/ingest. `KeyedAnn::set` stays insert-only.
    pub async fn unpack_packet(
        &self,
        packet: &MemoryPacket,
        codebook: Option<&VqCodebook>,
        id_override: Option<Uuid>,
    ) -> Result<SteerReport> {
        let _guard = self.mutation.lock().await;
        let id = id_override
            .or(packet.memory_id)
            .context("packet has no memory_id and no override")?;
        let semantics = packet.resolve_semantics(codebook)?;
        let before;
        let report = {
            let mut hot = self.hot_write();
            let projection = hot.projection.clone();
            let splat = hot
                .splats
                .iter_mut()
                .find(|splat| splat.memory_id == id)
                .with_context(|| format!("no splat for memory {id}"))?;
            before = splat.semantics.clone();
            splat.semantics = semantics.clone();
            splat.gain = packet.gain;
            splat.mass = packet.mass;
            if let Some(basin_id) = packet.basin_id.clone() {
                splat.basin_id = Some(basin_id);
            }
            splat.basin_locked = packet.basin_locked || splat.basin_locked;
            if let Some(projection) = &projection {
                splat.position = projection.apply(&splat.semantics);
            }
            SteerReport {
                memory_id: id,
                gain: splat.gain,
                op: InversionOp::Polarity,
                mass: splat.mass,
                cosine_before_after: crate::geometry::cosine(&before, &splat.semantics),
                collapse_risk: collapse_risk(packet.gain),
                basin_id: splat.basin_id.clone(),
                basin_locked: splat.basin_locked,
                // unpack64 writes semantics directly — no OI axis used.
                axis_source: "packet_direct".into(),
                axis_cosine: 1.0,
            }
        };
        self.ann.set(id, &semantics)?;
        self.persist_hot_and_ann()?;
        Ok(report)
    }

    /// Steer one splat in the hot field.
    ///
    /// - Negative **gain** inverts semantics (OI). Cold text is untouched.
    /// - Negative **mass** (if provided) makes the splat a repulsive bollard in dream.
    /// - These knobs do not imply each other.
    /// - Concept **axis** defaults to a non-self μ (basin peers / field mean / PCA).
    ///   Self-axis is opt-in via [`SteerAxisMode::SelfAxis`] (α inert — ablation only).
    pub async fn steer(&self, id: Uuid, opts: SteerOpts) -> Result<SteerReport> {
        let _guard = self.mutation.lock().await;
        let mut hot = self.hot_write();
        let projection = hot.projection.clone();

        // Resolve axis *before* mutably borrowing the target splat (need peer means).
        let before_for_axis = hot
            .splats
            .iter()
            .find(|s| s.memory_id == id)
            .map(|s| s.semantics.clone())
            .with_context(|| format!("no splat for memory {id}"))?;
        let (axis, axis_source) = resolve_steer_axis(&hot, id, &before_for_axis, opts.axis)?;
        let axis_cosine = inversion::cosine(&before_for_axis, &axis)?;

        let splat = hot
            .splats
            .iter_mut()
            .find(|splat| splat.memory_id == id)
            .with_context(|| format!("no splat for memory {id}"))?;

        let before = splat.semantics.clone();
        if opts.gain != 0.0 {
            let steered = inversion::apply_steering(&before, &axis, opts.gain, opts.op)?;
            splat.semantics = steered;
            // Record the α that was *applied*, not the one requested: `apply_steering`
            // clamps into the measured sweet band, so `--gain -0.9` moves the vector by
            // 0.35. Storing the request would claim a rotation the semantics never took,
            // and would let values outside the band back into the field that the
            // legacy-1.0 migration relies on being impossible.
            splat.gain = inversion::clamp_alpha(opts.gain);
            if let Some(projection) = &projection {
                splat.position = projection.apply(&splat.semantics);
            }
        }
        if let Some(mass) = opts.mass {
            splat.mass = mass;
        }
        if let Some(basin_id) = opts.basin_id.clone() {
            splat.basin_id = Some(basin_id);
        }
        if opts.basin_locked {
            splat.basin_locked = true;
        }

        let cosine_before_after = crate::geometry::cosine(&before, &splat.semantics);
        let semantics_for_ann = splat.semantics.clone();
        let report = SteerReport {
            memory_id: id,
            gain: splat.gain,
            op: opts.op,
            mass: splat.mass,
            cosine_before_after,
            // Deliberately the *requested* gain, not the stored one. Clamping caps the
            // applied α at 0.35, below the 0.40 collapse onset, so scoring the clamped
            // value would make this flag permanently false and useless. The caller asked
            // for something past the band; say so.
            collapse_risk: collapse_risk(opts.gain),
            basin_id: splat.basin_id.clone(),
            basin_locked: splat.basin_locked,
            axis_source,
            axis_cosine,
        };
        drop(hot);
        if opts.gain != 0.0 {
            self.ann.set(id, &semantics_for_ann)?;
        }
        self.persist_hot_and_ann()?;
        Ok(report)
    }

    /// Steer N addresses under one budget. Split once. One write lock.
    ///
    /// `memory_id` keys the pin. Basin-only keys the room (member-mean 64-d, not member 0).
    /// `|budget|` is capped at `ALPHA_MAX` once. Per-share α is not lifted to `ALPHA_MIN`.
    pub async fn steer_addresses(
        &self,
        addresses: Vec<SteerAddress>,
        opts: SteerOpts,
        budget: f32,
    ) -> Result<SteerBatchReport> {
        if addresses.is_empty() {
            bail!("steer_addresses needs at least one address");
        }
        let shares = batch_shares(&addresses);
        let sign = if opts.gain < 0.0 {
            -1.0
        } else if opts.gain > 0.0 {
            1.0
        } else {
            0.0
        };
        let mag = budget.abs().min(inversion::ALPHA_MAX);
        let _guard = self.mutation.lock().await;
        let mut hot = self.hot_write();
        let projection = hot.projection.clone();
        let mut targets = Vec::with_capacity(addresses.len());
        let mut ann_updates: Vec<(Uuid, Vec<f32>)> = Vec::new();

        for (addr, share) in addresses.iter().zip(shares.iter()) {
            let applied = if sign == 0.0 { 0.0 } else { sign * mag * share };
            let report = apply_one_address(
                &mut hot,
                addr,
                *share,
                applied,
                &opts,
                projection.as_ref(),
                &mut ann_updates,
            )?;
            targets.push(report);
        }
        drop(hot);
        for (id, sem) in ann_updates {
            self.ann.set(id, &sem)?;
        }
        self.persist_hot_and_ann()?;
        let total_applied: f32 = targets.iter().map(|t| t.applied_gain.abs()).sum();
        Ok(SteerBatchReport {
            budget: sign * mag,
            total_applied,
            collapse_risk: collapse_risk(budget),
            targets,
        })
    }

    fn persist_hot_and_ann(&self) -> Result<()> {
        let hot_path = self.config.hot_state_path();
        let hot_dir = hot_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.config.data_dir.clone());

        // Serialise writers for the duration of this write only. Holding it for the
        // process lifetime starved the ingest CLIs the store depends on.
        let mut guard =
            crate::hotlock::WriteGuard::acquire(&hot_dir, std::time::Duration::from_secs(30))?;

        // The clobber that actually happened was not concurrent, it was stale: a
        // snapshot loaded before someone else's dream, written afterwards. Compare
        // what is on disk against what we are about to write.
        let ours = self.hot_read().dream_cycle;
        if let Some(on_disk) = crate::hotlock::peek_dream_cycle(&hot_path) {
            if on_disk > ours {
                anyhow::bail!(
                    "stale hot state: disk is at dream_cycle {on_disk}, this process holds \
                     {ours}; refusing to overwrite a newer dream"
                );
            }
        }

        // Owning the path is not owning the file — a lock removed underneath us means
        // another writer may already be running.
        if !guard.still_owned() {
            anyhow::bail!(
                "lost the hot-state write lock before persisting (lock file replaced or \
                 removed); refusing to write unserialised"
            );
        }

        self.ann.save()?;
        let result = self
            .hot_read()
            .save(&hot_path, &self.config.packed_geometry_path());
        guard.release();
        result
    }

    fn records_read(&self) -> RwLockReadGuard<'_, HashMap<Uuid, MemoryRecord>> {
        self.records
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn records_write(&self) -> RwLockWriteGuard<'_, HashMap<Uuid, MemoryRecord>> {
        self.records
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn hot_read(&self) -> RwLockReadGuard<'_, HotState> {
        self.hot.read().unwrap_or_else(|poison| poison.into_inner())
    }

    fn hot_write(&self) -> RwLockWriteGuard<'_, HotState> {
        self.hot
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

fn live_frame_from(
    hot: &HotState,
    step: usize,
    kinetic_energy: f32,
    dreaming: bool,
    done: bool,
) -> LiveFrame {
    // Same filter as GET /api/splats?level=memories so Basin Field can zip by boot id list.
    let visible: Vec<_> = hot.splats.iter().filter(|splat| !splat.muted).collect();
    let n = visible.len();
    let mut positions = Vec::with_capacity(n * 3);
    let mut colors = Vec::with_capacity(n * 4);
    for splat in &visible {
        positions.extend_from_slice(&splat.position);
        colors.extend_from_slice(&splat.color_rgba);
    }
    LiveFrame {
        step,
        kinetic_energy,
        basins: hot.basins.len(),
        dream_cycle: hot.dream_cycle,
        dreaming,
        done,
        positions,
        colors,
        n,
    }
}

fn batch_shares(addresses: &[SteerAddress]) -> Vec<f32> {
    let n = addresses.len() as f32;
    if n < 1.0 {
        return Vec::new();
    }
    let raw: Vec<f32> = addresses
        .iter()
        .map(|a| a.share.unwrap_or(0.0).max(0.0))
        .collect();
    let sum: f32 = raw.iter().sum();
    if sum > 1e-6 {
        raw.iter().map(|s| s / sum).collect()
    } else {
        vec![1.0 / n; addresses.len()]
    }
}

fn room_semantics(hot: &HotState, basin_id: &str) -> Result<Vec<f32>> {
    let basin = hot
        .basins
        .iter()
        .find(|b| b.id == basin_id)
        .with_context(|| format!("no basin {basin_id}"))?;
    if basin.member_ids.is_empty() {
        bail!("basin {basin_id} has no members");
    }
    let map = hot.splat_map();
    let vecs: Vec<&[f32]> = basin
        .member_ids
        .iter()
        .filter_map(|id| map.get(id).map(|s| s.semantics.as_slice()))
        .collect();
    let mean = mean_semantics(vecs.into_iter())
        .with_context(|| format!("basin {basin_id} has no member semantics"))?;
    inversion::normalize(&mean)
}

fn ride_room(member: &[f32], room_before: &[f32], room_after: &[f32]) -> Result<Vec<f32>> {
    if member.len() != room_before.len() || member.len() != room_after.len() {
        bail!("room ride dim mismatch");
    }
    let mut out: Vec<f32> = member
        .iter()
        .zip(room_before)
        .zip(room_after)
        .map(|((m, b), a)| *m + (*a - *b))
        .collect();
    let n = inversion::norm(&out);
    if n > f32::EPSILON {
        for v in &mut out {
            *v /= n;
        }
    }
    Ok(out)
}

fn apply_opts_to_splat(splat: &mut crate::geometry::Splat, opts: &SteerOpts) {
    if let Some(mass) = opts.mass {
        splat.mass = mass;
    }
    if let Some(basin_id) = opts.basin_id.clone() {
        splat.basin_id = Some(basin_id);
    }
    if opts.basin_locked {
        splat.basin_locked = true;
    }
}

fn apply_one_address(
    hot: &mut HotState,
    addr: &SteerAddress,
    share: f32,
    applied: f32,
    opts: &SteerOpts,
    projection: Option<&crate::geometry::Projection>,
    ann_updates: &mut Vec<(Uuid, Vec<f32>)>,
) -> Result<AddressSteerReport> {
    if let Some(id) = addr.memory_id {
        let before_for_axis = hot
            .splats
            .iter()
            .find(|s| s.memory_id == id)
            .map(|s| s.semantics.clone())
            .with_context(|| format!("no splat for memory {id}"))?;
        let (axis, axis_source) = resolve_steer_axis(hot, id, &before_for_axis, opts.axis)?;
        let axis_cosine = inversion::cosine(&before_for_axis, &axis)?;
        let splat = hot
            .splats
            .iter_mut()
            .find(|s| s.memory_id == id)
            .with_context(|| format!("no splat for memory {id}"))?;
        let before = splat.semantics.clone();
        if applied != 0.0 {
            splat.semantics = inversion::apply_steering_share(&before, &axis, applied, opts.op)?;
            splat.gain = inversion::clamp_alpha_share(applied);
            if let Some(p) = projection {
                splat.position = p.apply(&splat.semantics);
            }
        }
        apply_opts_to_splat(splat, opts);
        let cosine_before_after = crate::geometry::cosine(&before, &splat.semantics);
        let basin_id = splat.basin_id.clone();
        if applied != 0.0 {
            ann_updates.push((id, splat.semantics.clone()));
        }
        return Ok(AddressSteerReport {
            kind: "memory".into(),
            memory_id: Some(id),
            basin_id,
            share,
            applied_gain: inversion::clamp_alpha_share(applied),
            members_touched: 1,
            cosine_before_after,
            axis_source,
            axis_cosine,
            semantics_64: None,
        });
    }

    let basin_id = addr
        .basin_id
        .clone()
        .context("address needs memory_id or basin_id")?;
    let room = room_semantics(hot, &basin_id)?;
    let (axis, axis_source) = resolve_steer_axis_for_room(hot, &basin_id, &room, opts.axis)?;
    let axis_cosine = inversion::cosine(&room, &axis)?;
    let steered = if applied != 0.0 {
        inversion::apply_steering_share(&room, &axis, applied, opts.op)?
    } else {
        room.clone()
    };
    let cosine_before_after = crate::geometry::cosine(&room, &steered);
    let member_ids: Vec<Uuid> = hot
        .basins
        .iter()
        .find(|b| b.id == basin_id)
        .map(|b| b.member_ids.clone())
        .with_context(|| format!("no basin {basin_id}"))?;
    let mut touched = 0usize;
    for id in &member_ids {
        let Some(splat) = hot.splats.iter_mut().find(|s| s.memory_id == *id) else {
            continue;
        };
        if applied != 0.0 {
            splat.semantics = ride_room(&splat.semantics, &room, &steered)?;
            splat.gain = inversion::clamp_alpha_share(applied);
            if let Some(p) = projection {
                splat.position = p.apply(&splat.semantics);
            }
            ann_updates.push((*id, splat.semantics.clone()));
        }
        apply_opts_to_splat(splat, opts);
        touched += 1;
    }
    Ok(AddressSteerReport {
        kind: "basin".into(),
        memory_id: None,
        basin_id: Some(basin_id),
        share,
        applied_gain: inversion::clamp_alpha_share(applied),
        members_touched: touched,
        cosine_before_after,
        axis_source,
        axis_cosine,
        semantics_64: Some(steered),
    })
}

fn resolve_steer_axis_for_room(
    hot: &HotState,
    basin_id: &str,
    room: &[f32],
    mode: inversion::SteerAxisMode,
) -> Result<(Vec<f32>, String)> {
    if matches!(mode, inversion::SteerAxisMode::SelfAxis) {
        return Ok((inversion::normalize(room)?, "self_room".into()));
    }
    let in_room: HashSet<Uuid> = hot
        .basins
        .iter()
        .find(|b| b.id == basin_id)
        .map(|b| b.member_ids.iter().copied().collect())
        .unwrap_or_default();
    let try_accept = |axis: Vec<f32>, source: &str| -> Result<Option<(Vec<f32>, String)>> {
        if axis.len() != room.len() {
            return Ok(None);
        }
        if inversion::norm(&axis) <= f32::EPSILON {
            return Ok(None);
        }
        let unit = inversion::normalize(&axis)?;
        if inversion::axis_is_degenerate(room, &unit)? {
            return Ok(None);
        }
        Ok(Some((unit, source.into())))
    };
    if !matches!(mode, inversion::SteerAxisMode::Basin) {
        if let Some(proj) = &hot.projection
            && proj.mean.len() == room.len()
            && inversion::norm(&proj.mean) > f32::EPSILON
        {
            if let Some(ok) = try_accept(proj.mean.clone(), "field_mean")? {
                return Ok(ok);
            }
        }
    }
    let outsiders: Vec<&[f32]> = hot
        .splats
        .iter()
        .filter(|s| !in_room.contains(&s.memory_id))
        .map(|s| s.semantics.as_slice())
        .collect();
    if let Some(mean) = mean_semantics(outsiders.into_iter())
        && let Some(ok) = try_accept(mean, "outside_room")?
    {
        return Ok(ok);
    }
    if let Some(proj) = &hot.projection
        && let Some(b) = proj.basis.first()
        && let Some(ok) = try_accept(b.clone(), "pca0")?
    {
        return Ok(ok);
    }
    Ok((
        inversion::orthogonal_fallback_axis(room)?,
        "orthogonal_fallback".into(),
    ))
}

/// Choose a non-degenerate concept axis for OI steering.
///
/// Order for [`SteerAxisMode::Auto`]:
/// 1. mean of other basin peers (same `basin_id`)
/// 2. field mean (`projection.mean`, else mean of all other splats)
/// 3. first PCA basis vector
/// 4. orthogonal geometric fallback
///
/// Rejects candidates with |cos(sem, axis)| ≥ [`inversion::AXIS_PARALLEL_COS`].
fn resolve_steer_axis(
    hot: &HotState,
    memory_id: Uuid,
    semantics: &[f32],
    mode: inversion::SteerAxisMode,
) -> Result<(Vec<f32>, String)> {
    use inversion::SteerAxisMode;

    if matches!(mode, SteerAxisMode::SelfAxis) {
        let axis = inversion::normalize(semantics)?;
        return Ok((axis, "self".into()));
    }

    let basin_id = hot
        .splats
        .iter()
        .find(|s| s.memory_id == memory_id)
        .and_then(|s| s.basin_id.clone());

    let try_accept = |axis: Vec<f32>, source: &str| -> Result<Option<(Vec<f32>, String)>> {
        if axis.len() != semantics.len() {
            return Ok(None);
        }
        let n = inversion::norm(&axis);
        if n <= f32::EPSILON {
            return Ok(None);
        }
        let unit = inversion::normalize(&axis)?;
        if inversion::axis_is_degenerate(semantics, &unit)? {
            return Ok(None);
        }
        Ok(Some((unit, source.into())))
    };

    let basin_peer_mean = || -> Option<Vec<f32>> {
        let bid = basin_id.as_ref()?;
        let peers: Vec<&_> = hot
            .splats
            .iter()
            .filter(|s| s.memory_id != memory_id && s.basin_id.as_ref() == Some(bid))
            .collect();
        if peers.is_empty() {
            return None;
        }
        mean_semantics(peers.iter().map(|s| s.semantics.as_slice()))
    };

    let field_mean = || -> Option<Vec<f32>> {
        if let Some(proj) = &hot.projection
            && proj.mean.len() == semantics.len()
            && inversion::norm(&proj.mean) > f32::EPSILON
        {
            return Some(proj.mean.clone());
        }
        let others: Vec<&[f32]> = hot
            .splats
            .iter()
            .filter(|s| s.memory_id != memory_id)
            .map(|s| s.semantics.as_slice())
            .collect();
        if others.is_empty() {
            return None;
        }
        mean_semantics(others.into_iter())
    };

    let pca0 = || -> Option<Vec<f32>> {
        hot.projection
            .as_ref()
            .and_then(|p| p.basis.first().cloned())
            .filter(|b| b.len() == semantics.len())
    };

    match mode {
        SteerAxisMode::SelfAxis => unreachable!(),
        SteerAxisMode::Basin => {
            if let Some(m) = basin_peer_mean()
                && let Some(ok) = try_accept(m, "basin_peers")?
            {
                return Ok(ok);
            }
            if let Some(m) = field_mean()
                && let Some(ok) = try_accept(m, "field_mean_fallback")?
            {
                return Ok(ok);
            }
        }
        SteerAxisMode::Field => {
            if let Some(m) = field_mean()
                && let Some(ok) = try_accept(m, "field_mean")?
            {
                return Ok(ok);
            }
        }
        SteerAxisMode::Auto => {
            if let Some(m) = basin_peer_mean()
                && let Some(ok) = try_accept(m, "basin_peers")?
            {
                return Ok(ok);
            }
            if let Some(m) = field_mean()
                && let Some(ok) = try_accept(m, "field_mean")?
            {
                return Ok(ok);
            }
            if let Some(m) = pca0()
                && let Some(ok) = try_accept(m, "pca0")?
            {
                return Ok(ok);
            }
        }
    }

    let fallback = inversion::orthogonal_fallback_axis(semantics)?;
    Ok((fallback, "orthogonal_fallback".into()))
}

fn mean_semantics<'a, I>(iter: I) -> Option<Vec<f32>>
where
    I: Iterator<Item = &'a [f32]>,
{
    let mut acc: Option<Vec<f32>> = None;
    let mut n = 0usize;
    for v in iter {
        match &mut acc {
            None => {
                acc = Some(v.to_vec());
                n = 1;
            }
            Some(sum) => {
                if sum.len() != v.len() {
                    continue;
                }
                for (s, x) in sum.iter_mut().zip(v) {
                    *s += *x;
                }
                n += 1;
            }
        }
    }
    let mut sum = acc?;
    if n == 0 {
        return None;
    }
    let inv = 1.0 / n as f32;
    for s in &mut sum {
        *s *= inv;
    }
    Some(sum)
}

#[cfg(test)]
mod axis_tests {
    use super::*;
    use crate::geometry::Splat;
    use crate::inversion::{self, InversionOp, SteerAxisMode};

    fn unit_at(dim: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[i] = 1.0;
        v
    }

    fn splat(id: Uuid, sem: Vec<f32>, basin: Option<&str>) -> Splat {
        Splat {
            memory_id: id,
            semantics: sem,
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [255, 255, 255, 255],
            mass: 1.0,
            radiance: 1.0,
            usage: 0.0,
            conductivity: 1.0,
            gain: 0.0,
            basin_locked: false,
            muted: false,
            duplicate_count: 1,
            domain: "chat".into(),
            basin_id: basin.map(str::to_string),
            lineage: Vec::new(),
        }
    }

    #[test]
    fn auto_prefers_basin_peers_over_self() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let mut hot = HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: vec![
                splat(a, unit_at(64, 0), Some("basin-1")),
                // peer mostly on dim 1
                splat(b, unit_at(64, 1), Some("basin-1")),
            ],
            basins: Vec::new(),
            projection: None,
        };
        // minimal HotState may need more fields - check geometry
        let _ = &mut hot;
        let (axis, source) =
            resolve_steer_axis(&hot, a, &unit_at(64, 0), SteerAxisMode::Auto).unwrap();
        assert_eq!(source, "basin_peers");
        assert!(!inversion::axis_is_degenerate(&unit_at(64, 0), &axis).unwrap());
        // peer mean ≈ e1 → cos with e0 ≈ 0
        assert!(axis[1].abs() > 0.9);
    }

    #[test]
    fn self_mode_reports_self() {
        let a = Uuid::from_u128(3);
        let sem = unit_at(64, 2);
        let hot = HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: vec![splat(a, sem.clone(), None)],
            basins: Vec::new(),
            projection: None,
        };
        let (axis, source) = resolve_steer_axis(&hot, a, &sem, SteerAxisMode::SelfAxis).unwrap();
        assert_eq!(source, "self");
        assert!((inversion::cosine(&sem, &axis).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn auto_alpha_is_responsive_with_peer_axis() {
        let a = Uuid::from_u128(4);
        let b = Uuid::from_u128(5);
        // Memory aligned with e0; peer pulls axis toward e0+e1 mix
        let mut sem = vec![0.0f32; 64];
        sem[0] = 1.0;
        let mut peer = vec![0.0f32; 64];
        peer[0] = 0.8;
        peer[1] = 0.6;
        peer = inversion::normalize(&peer).unwrap();
        let hot = HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: vec![splat(a, sem.clone(), Some("b")), splat(b, peer, Some("b"))],
            basins: Vec::new(),
            projection: None,
        };
        let (axis, _) = resolve_steer_axis(&hot, a, &sem, SteerAxisMode::Auto).unwrap();
        let mut prev = 1.0f32;
        for alpha in [-0.05f32, -0.15, -0.25, -0.35] {
            let out =
                inversion::apply_steering(&sem, &axis, alpha, InversionOp::NegativeGain).unwrap();
            let cos = inversion::cosine(&sem, &out).unwrap();
            assert!(cos < prev, "α={alpha} cos={cos} prev={prev}");
            prev = cos;
        }
    }

    #[test]
    fn two_addresses_split_the_budget_evenly() {
        let addrs = vec![
            SteerAddress {
                memory_id: Some(Uuid::from_u128(1)),
                basin_id: None,
                share: None,
            },
            SteerAddress {
                memory_id: Some(Uuid::from_u128(2)),
                basin_id: None,
                share: None,
            },
        ];
        let s = batch_shares(&addrs);
        assert_eq!(s.len(), 2);
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!((s[1] - 0.5).abs() < 1e-6);
        assert!((s[0] + s[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn room_64d_is_member_mean_not_member_zero() {
        let a = Uuid::from_u128(10);
        let b = Uuid::from_u128(11);
        let basin = Basin {
            id: "room-1".into(),
            parent_id: None,
            label: String::new(),
            path: String::new(),
            summary: String::new(),
            label_state: String::new(),
            stability: 1.0,
            centroid: [0.0, 0.0, 0.0],
            member_ids: vec![a, b],
            representative_ids: Vec::new(),
            updated_at: chrono::Utc::now(),
        };
        let hot = HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: vec![
                splat(a, unit_at(64, 0), Some("room-1")),
                splat(b, unit_at(64, 1), Some("room-1")),
            ],
            basins: vec![basin],
            projection: None,
        };
        let room = room_semantics(&hot, "room-1").unwrap();
        let m0 = &hot.splats[0].semantics;
        let cos0 = inversion::cosine(&room, m0).unwrap();
        assert!(cos0 < 0.9, "room 64-d must not be member 0, cos={cos0}");
        assert!((room[0] - room[1]).abs() < 1e-4, "mean of e0 and e1");
    }

    #[test]
    fn basin_steer_is_one_object_not_two_pins() {
        let a = Uuid::from_u128(20);
        let b = Uuid::from_u128(21);
        let basin = Basin {
            id: "room-2".into(),
            parent_id: None,
            label: String::new(),
            path: String::new(),
            summary: String::new(),
            label_state: String::new(),
            stability: 1.0,
            centroid: [0.0, 0.0, 0.0],
            member_ids: vec![a, b],
            representative_ids: Vec::new(),
            updated_at: chrono::Utc::now(),
        };
        let mut hot = HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: vec![
                splat(a, unit_at(64, 0), Some("room-2")),
                splat(b, unit_at(64, 1), Some("room-2")),
            ],
            basins: vec![basin],
            projection: None,
        };
        let room_before = room_semantics(&hot, "room-2").unwrap();
        let member0_before = hot.splats[0].semantics.clone();
        let addr = SteerAddress {
            memory_id: None,
            basin_id: Some("room-2".into()),
            share: Some(1.0),
        };
        let opts = SteerOpts {
            gain: -0.2,
            op: InversionOp::Polarity,
            mass: None,
            basin_id: None,
            basin_locked: false,
            axis: inversion::SteerAxisMode::Auto,
        };
        let mut ann = Vec::new();
        let report = apply_one_address(&mut hot, &addr, 1.0, -0.2, &opts, None, &mut ann).unwrap();
        assert_eq!(report.kind, "basin");
        assert!(report.memory_id.is_none(), "room is not a pin");
        assert_eq!(report.members_touched, 2);
        assert_eq!(ann.len(), 2, "members ride; still one steer object");
        let room = report.semantics_64.expect("one 64-d for the room");
        assert_eq!(room.len(), 64);
        let cos_moved = inversion::cosine(&room_before, &room).unwrap();
        assert!(
            cos_moved < 0.999,
            "gain≠0 must move the room, cos={cos_moved}"
        );
        let cos_m0 = inversion::cosine(&room, &member0_before).unwrap();
        assert!(cos_m0 < 0.9, "room 64-d must not be member 0, cos={cos_m0}");
        assert!(
            (report.cosine_before_after - cos_moved).abs() < 1e-5,
            "receipt cosine is the room, not a member"
        );
    }

    fn empty_hot(splats: Vec<Splat>, basins: Vec<Basin>) -> HotState {
        HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats,
            basins,
            projection: None,
        }
    }

    fn default_opts() -> SteerOpts {
        SteerOpts {
            gain: 0.0,
            op: InversionOp::Polarity,
            mass: None,
            basin_id: None,
            basin_locked: false,
            axis: inversion::SteerAxisMode::Auto,
        }
    }

    #[test]
    fn pin_honors_lock_and_assign_basin() {
        let a = Uuid::from_u128(30);
        let mut hot = empty_hot(vec![splat(a, unit_at(64, 0), None)], Vec::new());
        let addr = SteerAddress {
            memory_id: Some(a),
            basin_id: None,
            share: Some(1.0),
        };
        let mut opts = default_opts();
        opts.basin_id = Some("assigned".into());
        opts.basin_locked = true;
        let mut ann = Vec::new();
        let report = apply_one_address(&mut hot, &addr, 1.0, 0.0, &opts, None, &mut ann).unwrap();
        assert_eq!(hot.splats[0].basin_id.as_deref(), Some("assigned"));
        assert!(hot.splats[0].basin_locked);
        assert_eq!(report.basin_id.as_deref(), Some("assigned"));
        assert!(ann.is_empty(), "lock/assign must not rewrite HNSW");
    }

    #[test]
    fn zero_gain_pin_reports_this_call_alpha_not_leftover() {
        let a = Uuid::from_u128(31);
        let mut leftover = splat(a, unit_at(64, 0), None);
        leftover.gain = 0.35;
        let mut hot = empty_hot(vec![leftover], Vec::new());
        let addr = SteerAddress {
            memory_id: Some(a),
            basin_id: None,
            share: Some(1.0),
        };
        let opts = default_opts();
        let mut ann = Vec::new();
        let report = apply_one_address(&mut hot, &addr, 1.0, 0.0, &opts, None, &mut ann).unwrap();
        assert_eq!(report.applied_gain, 0.0);
        assert!(
            (hot.splats[0].gain - 0.35).abs() < 1e-6,
            "leftover splat.gain stays; receipt is this call"
        );
        assert!(ann.is_empty());
    }

    #[test]
    fn basin_steer_honors_lock_and_assign_basin_on_members() {
        let a = Uuid::from_u128(32);
        let b = Uuid::from_u128(33);
        let basin = Basin {
            id: "room-3".into(),
            parent_id: None,
            label: String::new(),
            path: String::new(),
            summary: String::new(),
            label_state: String::new(),
            stability: 1.0,
            centroid: [0.0, 0.0, 0.0],
            member_ids: vec![a, b],
            representative_ids: Vec::new(),
            updated_at: chrono::Utc::now(),
        };
        let mut hot = empty_hot(
            vec![
                splat(a, unit_at(64, 0), Some("room-3")),
                splat(b, unit_at(64, 1), Some("room-3")),
            ],
            vec![basin],
        );
        let addr = SteerAddress {
            memory_id: None,
            basin_id: Some("room-3".into()),
            share: Some(1.0),
        };
        let mut opts = default_opts();
        opts.basin_id = Some("rehomed".into());
        opts.basin_locked = true;
        let mut ann = Vec::new();
        let report = apply_one_address(&mut hot, &addr, 1.0, 0.0, &opts, None, &mut ann).unwrap();
        assert_eq!(report.kind, "basin");
        assert_eq!(report.applied_gain, 0.0);
        assert!(ann.is_empty());
        for splat in &hot.splats {
            assert!(splat.basin_locked);
            assert_eq!(splat.basin_id.as_deref(), Some("rehomed"));
        }
    }
}

#[cfg(test)]
mod cold_refresh_tests {
    use super::*;
    use crate::cold_store::ColdStore;
    use crate::config::AppConfig;

    #[test]
    fn refresh_from_cold_loads_sidecar_append() {
        let temp = tempfile::tempdir().unwrap();
        let mut config = AppConfig::default();
        config.data_dir = temp.path().to_path_buf();
        config.qdrant.enabled = false;
        let service = MemoryService::open(config).unwrap();
        let record = MemoryRecord::new("cli-session", "live", "yesetday file watchers");
        let cold = ColdStore::open(service.config.cold_path()).unwrap();
        assert_eq!(cold.append_new(std::slice::from_ref(&record)).unwrap(), 1);
        assert!(
            service.record(record.id).is_none(),
            "sidecar append is not in the MCP RAM map yet"
        );
        let added = service.refresh_from_cold().unwrap();
        assert_eq!(added, 1);
        let loaded = service
            .record(record.id)
            .expect("refresh must insert the row");
        assert!(
            loaded.text.contains("yesetday"),
            "live jsonl text must be queryable after refresh"
        );
    }
}
