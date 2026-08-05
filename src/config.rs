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
    /// `linear` (default) or `rrf` (reciprocal rank fusion). RRF ignores raw score scale.
    #[serde(default = "default_fusion")]
    pub fusion: String,
    /// RRF constant k (classic = 60). Only used when fusion = "rrf".
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,
    /// Min-max normalize BM25 and cosine over the candidate set before linear fusion.
    #[serde(default)]
    pub normalize_scores: bool,
    /// Boost: max dense cosine among peers in the same basin (geometry co-membership).
    #[serde(default)]
    pub weight_basin_peer: f32,
    /// Boost: cosine(query 64d, basin member-mean semantics).
    #[serde(default)]
    pub weight_basin_centroid: f32,
    /// Pull additional members from basins of top dense hits into the candidate pool
    /// when basin size ≤ this (avoids mega-basin explosion). 0 = disabled.
    #[serde(default)]
    pub basin_expand_max_size: usize,
    /// How many top dense hits seed basin expansion.
    #[serde(default = "default_basin_expand_seeds")]
    pub basin_expand_seeds: usize,
    /// Boost: local semantic neighborhood of dense hits (soft cluster).
    #[serde(default)]
    pub weight_basin_local: f32,
    /// Cosine threshold for local neighborhood membership.
    #[serde(default = "default_basin_local_threshold")]
    pub basin_local_threshold: f32,
    /// Expand candidate pool with HNSW neighbors of top dense hits (geometry graph).
    /// 0 = off. Typical 8–16.
    #[serde(default)]
    pub basin_graph_expand: usize,
    /// If true, local neighborhood uses sum of nearby dense scores (clump center);
    /// if false (default), uses max only.
    #[serde(default)]
    pub basin_local_sum: bool,
}

fn default_basin_local_threshold() -> f32 {
    0.55
}

fn default_basin_expand_seeds() -> usize {
    8
}

fn default_fusion() -> String {
    "linear".into()
}

fn default_rrf_k() -> f32 {
    60.0
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
    #[serde(default = "default_semantic_repulsion")]
    pub semantic_repulsion: f32,
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
            fusion: default_fusion(),
            rrf_k: default_rrf_k(),
            normalize_scores: false,
            weight_basin_peer: 0.0,
            weight_basin_centroid: 0.0,
            basin_expand_max_size: 0,
            basin_expand_seeds: default_basin_expand_seeds(),
            weight_basin_local: 0.0,
            basin_local_threshold: default_basin_local_threshold(),
            basin_graph_expand: 0,
            basin_local_sum: false,
        }
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            // Tuned by sweep against two measured objectives on the real memory corpus:
            // semantic fidelity (correlation of 3-d distance with semantic cosine) and quarantine
            // (median cross-domain distance over median same-domain distance). Settles by ~600
            // steps and holds through 2400.
            steps: 1200,
            dt: 0.02,
            damping: 0.9,
            origin_pull: 0.05,
            neighbor_radius: 2.0,
            // Spring stiffness for semantically similar pairs.
            attraction: 2.0,
            // Now the neutral point of the pair force, not a gate: above it pairs attract, below
            // it they repel.
            semantic_threshold: 0.55,
            repulsion_radius: 0.5,
            // Hard-core only — stops a settled cluster collapsing onto a single point.
            repulsion: 5.0,
            // Strength of the decaying semantic repulsion between dissimilar pairs.
            semantic_repulsion: 2.0,
            // Similarity penalty for a foreign domain. This is the quarantine dial: 0.0 gives
            // separation 1.3 (none), 0.15 gives 3.6, 1.0 gives 12.8, and semantic fidelity falls
            // as it rises. 0.15 keeps clusters meaningful while still isolating a junk domain.
            cross_domain_repulsion: 0.15,
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

/// Refuse all-interfaces / unspecified binds. Loopback only (`127.0.0.1`, `localhost`, `::1`).
pub fn assert_loopback_bind(bind: &str, field: &str) -> Result<()> {
    let bind = bind.trim();
    let host = if let Some(rest) = bind.strip_prefix('[') {
        // [IPv6]:port
        rest.split(']')
            .next()
            .unwrap_or(rest)
            .trim()
            .to_ascii_lowercase()
    } else if bind.matches(':').count() == 1 {
        // host:port
        bind.rsplit_once(':')
            .map(|(h, _)| h.trim().to_ascii_lowercase())
            .unwrap_or_else(|| bind.to_ascii_lowercase())
    } else {
        // bare host or weird form — treat whole string
        bind.to_ascii_lowercase()
    };

    let forbidden = matches!(
        host.as_str(),
        "0.0.0.0" | "::" | "[::]" | "*" | "" | "0" | "[::0]" | "::0"
    ) || host.starts_with("0.0.0.0")
        || host == "[::]"
        || host == "::0";

    if forbidden {
        anyhow::bail!(
            "{field}={bind:?} is not allowed: bind must be loopback only \
             (use 127.0.0.1:PORT or [::1]:PORT). All-interfaces binds (0.0.0.0 / ::) are rejected."
        );
    }

    let ok = host == "127.0.0.1"
        || host == "localhost"
        || host == "::1"
        || host == "[::1]";
    if !ok {
        anyhow::bail!(
            "{field}={bind:?} is not loopback. Allowed hosts: 127.0.0.1, localhost, ::1. \
             Refusing to open a non-local socket."
        );
    }
    Ok(())
}

impl AppConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let config = if !path.exists() {
            Self::default()
        } else {
            let text = fs::read_to_string(path)
                .with_context(|| format!("failed to read config {}", path.display()))?;
            toml::from_str(&text).with_context(|| format!("invalid config {}", path.display()))?
        };
        config.validate_localhost_only()?;
        Ok(config)
    }

    /// HTTP / service bind must be loopback only. Never `0.0.0.0`, `::`, or `*`.
    pub fn validate_localhost_only(&self) -> Result<()> {
        assert_loopback_bind(&self.server.bind, "server.bind")?;
        Ok(())
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

    /// OCR results, keyed by asset id. Derived state: expensive to rebuild (hours of vision model
    /// time), but never authoritative — deleting it costs time, not memories.
    pub fn ocr_cache_path(&self) -> PathBuf {
        self.data_dir.join("derived").join("extracted.jsonl")
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

/// Older config files predate the split between hard-core and semantic repulsion.
fn default_semantic_repulsion() -> f32 {
    2.0
}
