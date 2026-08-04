use crate::config::QdrantConfig;
use crate::record::MemoryRecord;
use anyhow::{Context, Result};
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use uuid::Uuid;

#[derive(Clone)]
pub struct QdrantIndex {
    client: Client,
    config: QdrantConfig,
    scope_key: String,
    dimensions: usize,
}

#[derive(Debug, Clone)]
pub struct DenseHit {
    pub id: Uuid,
    pub score: f32,
}

#[derive(Debug, Serialize)]
struct Point<'a> {
    id: String,
    vector: &'a [f32],
    payload: Value,
}

#[derive(Debug, Deserialize)]
struct SearchPoint {
    id: Value,
    score: f32,
}

impl QdrantIndex {
    pub fn new(config: QdrantConfig, scope_key: String, dimensions: usize) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(180))
            .build()?;
        Ok(Self {
            client,
            config,
            scope_key,
            dimensions,
        })
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub async fn validate(&self) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let response: Value = self
            .authorized(self.client.get(format!(
                "{}/collections/{}",
                self.config.url.trim_end_matches('/'),
                self.config.collection
            )))
            .send()
            .await
            .context("failed to reach Qdrant")?
            .error_for_status()?
            .json()
            .await?;
        let size = response
            .pointer("/result/config/params/vectors/size")
            .and_then(Value::as_u64)
            .context("Qdrant collection response omitted vector size")? as usize;
        let distance = response
            .pointer("/result/config/params/vectors/distance")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if size != self.dimensions || !distance.eq_ignore_ascii_case("cosine") {
            anyhow::bail!(
                "Qdrant collection must be {}-dimensional Cosine; found {} {}",
                self.dimensions,
                size,
                distance
            );
        }
        Ok(())
    }

    pub async fn ensure_scope_index(&self) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let url = format!(
            "{}/collections/{}/index?wait=true",
            self.config.url.trim_end_matches('/'),
            self.config.collection
        );
        self.authorized(self.client.put(url))
            .json(&serde_json::json!({
                "field_name": "scope_key",
                "field_schema": "keyword"
            }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn upsert(&self, records: &[MemoryRecord], vectors: &[Vec<f32>]) -> Result<()> {
        if !self.enabled() || records.is_empty() {
            return Ok(());
        }
        if records.len() != vectors.len() {
            anyhow::bail!("Qdrant upsert record/vector count mismatch");
        }
        let mut points = Vec::with_capacity(records.len());
        for (record, vector) in records.iter().zip(vectors) {
            if vector.len() != self.dimensions {
                anyhow::bail!("refusing to upsert wrong-dimensional vector");
            }
            points.push(Point {
                id: record.id.to_string(),
                vector,
                payload: serde_json::json!({
                    "scope_key": self.scope_key,
                    "memory_id": record.id,
                    "domain": record.domain,
                    "source": record.source,
                    "model": record.model,
                    "conversation_id": record.conversation_id,
                    "timestamp": record.timestamp,
                }),
            });
        }
        let url = format!(
            "{}/collections/{}/points?wait=true",
            self.config.url.trim_end_matches('/'),
            self.config.collection
        );
        self.authorized(self.client.put(url))
            .json(&serde_json::json!({ "points": points }))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn search(&self, vector: &[f32], limit: usize) -> Result<Vec<DenseHit>> {
        if !self.enabled() {
            return Ok(Vec::new());
        }
        let url = format!(
            "{}/collections/{}/points/query",
            self.config.url.trim_end_matches('/'),
            self.config.collection
        );
        let response: Value = self
            .authorized(self.client.post(url))
            .json(&serde_json::json!({
                "query": vector,
                "limit": limit,
                "with_payload": false,
                "filter": {
                    "must": [{
                        "key": "scope_key",
                        "match": { "value": self.scope_key }
                    }]
                }
            }))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let points = response
            .pointer("/result/points")
            .or_else(|| response.get("result"))
            .and_then(Value::as_array)
            .context("Qdrant query response omitted points")?;
        let mut hits = Vec::with_capacity(points.len());
        for value in points {
            let point: SearchPoint = serde_json::from_value(value.clone())?;
            let id_text = point
                .id
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| point.id.to_string());
            if let Ok(id) = Uuid::parse_str(id_text.trim_matches('"')) {
                hits.push(DenseHit {
                    id,
                    score: point.score,
                });
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        Ok(hits)
    }

    pub async fn count_scope(&self) -> Result<u64> {
        if !self.enabled() {
            return Ok(0);
        }
        let url = format!(
            "{}/collections/{}/points/count",
            self.config.url.trim_end_matches('/'),
            self.config.collection
        );
        let response: Value = self
            .authorized(self.client.post(url))
            .json(&serde_json::json!({
                "exact": true,
                "filter": {
                    "must": [{
                        "key": "scope_key",
                        "match": { "value": self.scope_key }
                    }]
                }
            }))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        response
            .pointer("/result/count")
            .and_then(Value::as_u64)
            .context("Qdrant count response omitted count")
    }

    fn authorized(&self, builder: RequestBuilder) -> RequestBuilder {
        if let Some(api_key) = &self.config.api_key {
            builder.header("api-key", api_key)
        } else {
            builder
        }
    }
}
