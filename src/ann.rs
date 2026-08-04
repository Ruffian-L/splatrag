use crate::config::RetrievalConfig;
use anyhow::{Context, Result};
use fast_hnsw::Builder;
use fast_hnsw::distance::Cosine;
use fast_hnsw::labeled::LabeledIndex;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use uuid::Uuid;

pub struct KeyedAnn {
    index: RwLock<LabeledIndex<Cosine, String>>,
    key_to_index: RwLock<HashMap<Uuid, usize>>,
    path: PathBuf,
    dim: usize,
    ef_search: usize,
}

#[derive(Debug, Clone)]
pub struct AnnHit {
    pub id: Uuid,
    pub distance: f32,
}

impl KeyedAnn {
    pub fn open(path: impl Into<PathBuf>, dim: usize, config: &RetrievalConfig) -> Result<Self> {
        let path = path.into();
        let index: LabeledIndex<Cosine, String> = if path.exists() {
            LabeledIndex::load(&path, Cosine)
                .with_context(|| format!("failed to load HNSW {}", path.display()))?
        } else {
            Builder::new()
                .m(config.hnsw_m)
                .ef_construction(config.hnsw_ef_construction)
                .capacity(config.hnsw_max_nodes)
                .seed(42)
                .build_labeled(Cosine)
        };
        if !index.is_empty() && index.inner.dim() != Some(dim) {
            anyhow::bail!(
                "HNSW dimension mismatch: expected {}, found {:?}",
                dim,
                index.inner.dim()
            );
        }
        let mut key_to_index = HashMap::with_capacity(index.len());
        for internal_id in 0..index.len() {
            if let Ok(id) = Uuid::parse_str(index.get_payload(internal_id)) {
                key_to_index.insert(id, internal_id);
            }
        }
        Ok(Self {
            index: RwLock::new(index),
            key_to_index: RwLock::new(key_to_index),
            path,
            dim,
            ef_search: config.hnsw_ef_search,
        })
    }

    pub fn set(&self, id: Uuid, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            anyhow::bail!(
                "HNSW vector dimension mismatch: expected {}, got {}",
                self.dim,
                vector.len()
            );
        }
        if self.keys_read().contains_key(&id) {
            return Ok(());
        }
        let mut index = self.index_write();
        let internal_id = index.insert(vector.to_vec(), id.to_string());
        self.keys_write().insert(id, internal_id);
        Ok(())
    }

    pub fn search(&self, vector: &[f32], limit: usize) -> Result<Vec<AnnHit>> {
        let index = self.index_read();
        if index.is_empty() {
            return Ok(Vec::new());
        }
        let mut hits = index
            .search(vector, limit, self.ef_search.max(limit))
            .into_iter()
            .filter_map(|hit| {
                Uuid::parse_str(hit.payload).ok().map(|id| AnnHit {
                    id,
                    distance: hit.distance,
                })
            })
            .collect::<Vec<_>>();
        hits.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.id.cmp(&b.id))
        });
        Ok(hits)
    }

    pub fn neighbors_for_key(&self, id: Uuid) -> Result<Vec<Uuid>> {
        let index = self.index_read();
        let Some(internal_id) = self.keys_read().get(&id).copied() else {
            return Ok(Vec::new());
        };
        let embedding = index.get_embedding(internal_id);
        let mut keys = index
            .search(embedding, 33, self.ef_search.max(33))
            .into_iter()
            .filter_map(|hit| Uuid::parse_str(hit.payload).ok())
            .filter(|other| *other != id)
            .collect::<Vec<_>>();
        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    pub fn save(&self) -> Result<()> {
        create_parent(&self.path)?;
        if self.index_read().is_empty() {
            return Ok(());
        }
        let temp = self.path.with_extension("index.tmp");
        self.index_read().save(&temp)?;
        fs::rename(temp, &self.path)?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.index_read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.index_read().is_empty()
    }

    pub fn keys(&self) -> Vec<Uuid> {
        let mut keys = self.keys_read().keys().copied().collect::<Vec<_>>();
        keys.sort();
        keys
    }

    fn index_read(&self) -> RwLockReadGuard<'_, LabeledIndex<Cosine, String>> {
        self.index
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn index_write(&self) -> RwLockWriteGuard<'_, LabeledIndex<Cosine, String>> {
        self.index
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn keys_read(&self) -> RwLockReadGuard<'_, HashMap<Uuid, usize>> {
        self.key_to_index
            .read()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn keys_write(&self) -> RwLockWriteGuard<'_, HashMap<Uuid, usize>> {
        self.key_to_index
            .write()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

fn create_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(())
}
