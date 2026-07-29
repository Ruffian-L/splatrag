use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub data_dir: PathBuf,
    pub scope_key: String,
    pub qdrant: QdrantConfig,
    pub embedding: EmbeddingConfig,
    pub labeling: LabelingConfig,
    pub retrieval: RetrievalConfig,
    pub physics: PhysicsConfig,
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QdrantConfig {
    pub url: String,
    pub collection: String,
    pub api_key: Option<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub url: String,
    pub model: String,
    pub dimensions: usize,
    pub batch_size: usize,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LabelingConfig {
    pub url: String,
    pub model: String,
    pub enabled: bool,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalConfig {
    pub weight_cosine: f32,
    pub weight_bm25: f32,
    pub weight_radiance: f32,
    pub candidate_multiplier: usize,
    pub hnsw_max_nodes: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsConfig {
    pub steps: usize,
    pub dt: f32,
    pub damping: f32,
    pub origin_pull: f32,
    pub neighbor_radius: f32,
    pub attraction: f32,
    pub semantic_threshold: f32,
    pub repulsion_radius: f32,
    pub repulsion: f32,
    pub cross_domain_repulsion: f32,
    pub basin_radius: f32,
    pub min_basin_size: usize,
    pub spatial_cell_size: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub bind: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            scope_key: "splatrag-main".into(),
            qdrant: QdrantConfig::default(),
            embedding: EmbeddingConfig::default(),
            labeling: LabelingConfig::default(),
            retrieval: RetrievalConfig::default(),
            physics: PhysicsConfig::default(),
            server: ServerConfig::default(),
        }
    }
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:6360".into(),
            collection: "export-conversations".into(),
            api_key: None,
            enabled: true,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8081".into(),
            model: "Qwen3-Embedding-8B-Q8_0.gguf".into(),
            dimensions: 4096,
            batch_size: 32,
            timeout_seconds: 180,
        }
    }
}

impl Default for LabelingConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:8082".into(),
            model: "gemma-3-4b-it-q4_0.gguf".into(),
            enabled: true,
            timeout_seconds: 180,
        }
    }
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            weight_cosine: 10.0,
            weight_bm25: 1.0,
            weight_radiance: 5.0,
            candidate_multiplier: 8,
            hnsw_max_nodes: 1_000_000,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 80,
        }
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            steps: 120,
            dt: 0.016,
            damping: 0.92,
            origin_pull: 0.002,
            neighbor_radius: 2.0,
            attraction: 0.02,
            semantic_threshold: 0.55,
            repulsion_radius: 0.45,
            repulsion: 0.12,
            cross_domain_repulsion: 0.35,
            basin_radius: 0.8,
            min_basin_size: 3,
            spatial_cell_size: 1.0,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1:8765".into(),
        }
    }
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read config {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("invalid config {}", path.display()))
    }

    pub fn write_default(path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if path.exists() {
            anyhow::bail!("refusing to overwrite existing config {}", path.display());
        }
        fs::write(path, toml::to_string_pretty(&Self::default())?)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn cold_path(&self) -> PathBuf {
        self.data_dir.join("cold").join("memories.jsonl")
    }

    pub fn quarantine_path(&self) -> PathBuf {
        self.data_dir.join("quarantine").join("ingest-errors.jsonl")
    }

    pub fn lexical_path(&self) -> PathBuf {
        self.data_dir.join("indexes").join("tantivy")
    }

    pub fn ann_graph_path(&self) -> PathBuf {
        self.data_dir.join("indexes").join("hnsw.index")
    }

    pub fn hot_state_path(&self) -> PathBuf {
        self.data_dir.join("hot").join("state.json")
    }

    pub fn packed_geometry_path(&self) -> PathBuf {
        self.data_dir.join("hot").join("geometry.bin")
    }
}
