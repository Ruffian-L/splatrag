use crate::ann::KeyedAnn;
use crate::cold_store::ColdStore;
use crate::config::AppConfig;
use crate::embedding::{EmbeddingClient, LabelingClient, matryoshka64};
use crate::geometry::{Basin, HotState};
use crate::inversion::{self, InversionOp, collapse_risk};
use crate::lexical::LexicalIndex;
use crate::packet::{MemoryPacket, VqCodebook};
use crate::physics::{DreamReport, dream, dream_with_progress, domain_state_color};
use crate::pick::{MemoryPickSet, PickConfig};
use crate::qdrant::QdrantIndex;
use crate::record::{MemoryRecord, RecallFilters, RecallHit};
use crate::retrieval::RetrievalEngine;
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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

impl MemoryService {
    pub fn open(config: AppConfig) -> Result<Self> {
        let cold = ColdStore::open(config.cold_path())?;
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

    /// Recolor the hot field by domain/basin for static viewing (no physics).
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
        let hot = self.hot_read();
        let n = hot.splats.len();
        let mut positions = Vec::with_capacity(n * 3);
        let mut colors = Vec::with_capacity(n * 4);
        for splat in &hot.splats {
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

    fn emit_live_frame(&self, frame: LiveFrame) {
        if let Ok(guard) = self.live_log.read() {
            if let Some(path) = guard.as_ref() {
                if let Ok(line) = serde_json::to_string(&serde_json::json!({
                    "step": frame.step,
                    "kinetic_energy": frame.kinetic_energy,
                    "basins": frame.basins,
                    "dream_cycle": frame.dream_cycle,
                    "dreaming": frame.dreaming,
                    "done": frame.done,
                    "n": frame.n,
                })) {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                    {
                        let _ = writeln!(f, "{line}");
                    }
                }
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

    pub async fn recall(
        &self,
        query: &str,
        limit: usize,
        filters: &RecallFilters,
    ) -> Result<Vec<RecallHit>> {
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
            if let Some(sid) = &mem.source_record_id {
                if let Some(splat) = splat_map.get(&mem.id) {
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
        let hits = self.recall(prompt, depth, filters).await?;
        let hot = self.hot_read().clone();
        // Second embed of the same prompt. Retrieval computes this vector internally but does not
        // hand it back, and the null needs it to score the field. Worth ~50ms rather than widening
        // the retrieval signature — revisit if `pick` ever runs per token.
        let query_64 = crate::embedding::matryoshka64(&self.embedding.embed(prompt).await?)?;
        let null = crate::pick::NullModel::estimate(
            &query_64,
            &hot.splats,
            crate::pick::NULL_SAMPLES,
        );
        Ok(crate::pick::build(
            prompt,
            &hits,
            &hot,
            &self.config.embedding.model,
            null,
            config,
        ))
    }

    pub async fn dream(&self) -> Result<DreamReport> {
        let _guard = self.mutation.lock().await;
        let report = {
            let mut hot = self.hot_write();
            dream(&mut hot, &self.ann, &self.config.physics)?
        };
        self.persist_hot_and_ann()?;
        Ok(report)
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
        // Physics is O(N²) CPU; keep the async runtime free for SSE while we run.
        let result = tokio::task::block_in_place(|| self.dream_live_inner(emit_every));
        self.dreaming.store(false, Ordering::SeqCst);
        result
    }

    fn dream_live_inner(&self, emit_every: usize) -> Result<DreamReport> {
        {
            let mut hot = self.hot_write();
            for splat in &mut hot.splats {
                splat.color_rgba =
                    domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0);
            }
            let n = hot.splats.len();
            let mut positions = Vec::with_capacity(n * 3);
            let mut colors = Vec::with_capacity(n * 4);
            for splat in &hot.splats {
                positions.extend_from_slice(&splat.position);
                colors.extend_from_slice(&splat.color_rgba);
            }
            let start = LiveFrame {
                step: 0,
                kinetic_energy: hot.kinetic_energy,
                basins: hot.basins.len(),
                dream_cycle: hot.dream_cycle,
                dreaming: true,
                done: false,
                positions,
                colors,
                n,
            };
            drop(hot);
            self.emit_live_frame(start);
        }

        let report = {
            let mut hot = self.hot_write();
            let tx = self.live_tx.clone();
            let log_path = self.live_log.read().ok().and_then(|g| g.clone());
            dream_with_progress(
                &mut hot,
                &self.ann,
                &self.config.physics,
                emit_every,
                |step, energy, hot| {
                    let n = hot.splats.len();
                    let mut positions = Vec::with_capacity(n * 3);
                    let mut colors = Vec::with_capacity(n * 4);
                    for splat in &hot.splats {
                        positions.extend_from_slice(&splat.position);
                        colors.extend_from_slice(&splat.color_rgba);
                    }
                    let frame = LiveFrame {
                        step,
                        kinetic_energy: energy,
                        basins: hot.basins.len(),
                        dream_cycle: hot.dream_cycle,
                        dreaming: true,
                        done: false,
                        positions,
                        colors,
                        n,
                    };
                    if let Some(path) = &log_path {
                        if let Ok(line) = serde_json::to_string(&serde_json::json!({
                            "step": frame.step,
                            "kinetic_energy": frame.kinetic_energy,
                            "basins": frame.basins,
                            "n": frame.n,
                        })) {
                            use std::io::Write;
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(path)
                            {
                                let _ = writeln!(f, "{line}");
                            }
                        }
                    }
                    let _ = tx.send(frame);
                },
            )?
        };
        self.persist_hot_and_ann()?;
        let mut done = self.snapshot_live_frame(report.steps, report.kinetic_energy, false, true);
        done.basins = report.basins;
        self.emit_live_frame(done);
        Ok(report)
    }

    pub async fn label_basins(&self) -> Result<usize> {
        if !self.labeler.enabled() {
            return Ok(0);
        }
        let _guard = self.mutation.lock().await;
        let records = self.records_read().clone();
        let pending = self
            .hot_read()
            .basins
            .iter()
            .filter(|basin| basin.label_state == "pending")
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
                "derived index drift: cold={}, hnsw={}, splats={}; run rebuild-index",
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

    /// Apply a packet onto an existing splat (by packet.memory_id or override id).
    ///
    /// Writes 64D semantics into hot + HNSW. Gain/mass/basin from the packet.
    /// Cold text is never rewritten. Unicode is only used if raw semantics are absent.
    pub fn unpack_packet(
        &self,
        packet: &MemoryPacket,
        codebook: Option<&VqCodebook>,
        id_override: Option<Uuid>,
    ) -> Result<SteerReport> {
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
        let (axis, axis_source) =
            resolve_steer_axis(&hot, id, &before_for_axis, opts.axis)?;
        let axis_cosine = inversion::cosine(&before_for_axis, &axis)?;

        let splat = hot
            .splats
            .iter_mut()
            .find(|splat| splat.memory_id == id)
            .with_context(|| format!("no splat for memory {id}"))?;

        let before = splat.semantics.clone();
        if opts.gain != 0.0 {
            let steered =
                inversion::apply_steering(&before, &axis, opts.gain, opts.op)?;
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

    fn persist_hot_and_ann(&self) -> Result<()> {
        self.ann.save()?;
        self.hot_read().save(
            &self.config.hot_state_path(),
            &self.config.packed_geometry_path(),
        )
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
        if let Some(proj) = &hot.projection {
            if proj.mean.len() == semantics.len() && inversion::norm(&proj.mean) > f32::EPSILON {
                return Some(proj.mean.clone());
            }
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
            if let Some(m) = basin_peer_mean() {
                if let Some(ok) = try_accept(m, "basin_peers")? {
                    return Ok(ok);
                }
            }
            if let Some(m) = field_mean() {
                if let Some(ok) = try_accept(m, "field_mean_fallback")? {
                    return Ok(ok);
                }
            }
        }
        SteerAxisMode::Field => {
            if let Some(m) = field_mean() {
                if let Some(ok) = try_accept(m, "field_mean")? {
                    return Ok(ok);
                }
            }
        }
        SteerAxisMode::Auto => {
            if let Some(m) = basin_peer_mean() {
                if let Some(ok) = try_accept(m, "basin_peers")? {
                    return Ok(ok);
                }
            }
            if let Some(m) = field_mean() {
                if let Some(ok) = try_accept(m, "field_mean")? {
                    return Ok(ok);
                }
            }
            if let Some(m) = pca0() {
                if let Some(ok) = try_accept(m, "pca0")? {
                    return Ok(ok);
                }
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
    use crate::inversion::{self, SteerAxisMode};

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
            gain: 0.0,
            basin_locked: false,
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
        let (axis, source) =
            resolve_steer_axis(&hot, a, &sem, SteerAxisMode::SelfAxis).unwrap();
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
            splats: vec![
                splat(a, sem.clone(), Some("b")),
                splat(b, peer, Some("b")),
            ],
            basins: Vec::new(),
            projection: None,
        };
        let (axis, _) = resolve_steer_axis(&hot, a, &sem, SteerAxisMode::Auto).unwrap();
        let mut prev = 1.0f32;
        for alpha in [-0.05f32, -0.15, -0.25, -0.35] {
            let out = inversion::apply_steering(&sem, &axis, alpha, InversionOp::NegativeGain)
                .unwrap();
            let cos = inversion::cosine(&sem, &out).unwrap();
            assert!(cos < prev, "α={alpha} cos={cos} prev={prev}");
            prev = cos;
        }
    }
}
