use crate::ann::KeyedAnn;
use crate::cold_store::ColdStore;
use crate::config::AppConfig;
use crate::embedding::{EmbeddingClient, LabelingClient, matryoshka64};
use crate::geometry::{Basin, HotState};
use crate::lexical::LexicalIndex;
use crate::physics::{DreamReport, dream};
use crate::qdrant::QdrantIndex;
use crate::record::{MemoryRecord, RecallFilters, RecallHit};
use crate::retrieval::RetrievalEngine;
use anyhow::Result;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tokio::sync::Mutex;
use uuid::Uuid;

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
        })
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

    pub async fn dream(&self) -> Result<DreamReport> {
        let _guard = self.mutation.lock().await;
        let report = {
            let mut hot = self.hot_write();
            dream(&mut hot, &self.ann, &self.config.physics)?
        };
        self.persist_hot_and_ann()?;
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
