//! Frontend-safe JSON DTOs for the SplatLens HTTP surface.
//!
//! Domain types (`MemoryRecord`, `RecallHit`, `Basin`) carry internal fields the palace UI does not
//! use. These wire types serialize only the subset in the frontend Zod schemas so proxy re-validation
//! stays stable across Zod 3/4.

use crate::geometry::Basin;
use crate::record::{MemoryRecord, RecallHit, ScoreBreakdown};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;

fn rfc3339(dt: DateTime<Utc>) -> String {
    dt.to_rfc3339_opts(SecondsFormat::Millis, true)
}

#[derive(Debug, Clone, Serialize)]
pub struct WireMemoryRecord {
    pub id: String,
    pub source_key: String,
    pub text: String,
    pub domain: String,
    pub timestamp: Option<String>,
    pub ingested_at: String,
    pub source: String,
    pub speaker: Option<String>,
    pub model: Option<String>,
    pub conversation_id: Option<String>,
    pub turn_index: Option<u64>,
}

impl From<&MemoryRecord> for WireMemoryRecord {
    fn from(m: &MemoryRecord) -> Self {
        Self {
            id: m.id.to_string(),
            source_key: m.source_key.clone(),
            text: m.text.clone(),
            domain: m.domain.clone(),
            timestamp: m.timestamp.map(rfc3339),
            ingested_at: rfc3339(m.ingested_at),
            source: m.source.clone(),
            speaker: m.speaker.clone(),
            model: m.model.clone(),
            conversation_id: m.conversation_id.clone(),
            turn_index: m.turn_index,
        }
    }
}

impl From<MemoryRecord> for WireMemoryRecord {
    fn from(m: MemoryRecord) -> Self {
        WireMemoryRecord::from(&m)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WireRecallHit {
    pub memory: WireMemoryRecord,
    pub basin_id: Option<String>,
    pub basin_label: Option<String>,
    pub scores: ScoreBreakdown,
    // deliberately omit `context` — frontend schema has no such field
}

impl From<&RecallHit> for WireRecallHit {
    fn from(hit: &RecallHit) -> Self {
        Self {
            memory: WireMemoryRecord::from(&hit.memory),
            basin_id: hit.basin_id.clone(),
            basin_label: hit.basin_label.clone(),
            scores: hit.scores.clone(),
        }
    }
}

impl From<RecallHit> for WireRecallHit {
    fn from(hit: RecallHit) -> Self {
        WireRecallHit::from(&hit)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WireBasin {
    pub id: String,
    pub label: String,
    pub path: String,
    pub summary: String,
    /// Always `"pending"` or `"stable"` on the wire (Zod enum).
    pub label_state: String,
    pub stability: f32,
    pub centroid: [f32; 3],
    pub member_count: usize,
    /// Sample only — full membership is `member_count` + `browse_basin`.
    /// The personal unlabeled basin is ~62k ids and blows the palace boot.
    pub member_ids: Vec<String>,
    pub representative_ids: Vec<String>,
    pub updated_at: String,
}

/// Frontend Zod allows only `"pending" | "stable"`. Topology also writes `"locked"` for
/// basin-locked splats; treat that as stable on the wire so re-validation does not fail.
fn wire_label_state(raw: &str) -> String {
    match raw {
        "pending" => "pending".into(),
        _ => "stable".into(), // "stable", "locked", or any future value
    }
}

impl From<&Basin> for WireBasin {
    fn from(b: &Basin) -> Self {
        Self {
            id: b.id.clone(),
            label: b.label.clone(),
            path: b.path.clone(),
            summary: b.summary.clone(),
            label_state: wire_label_state(&b.label_state),
            stability: b.stability.clamp(0.0, 1.0),
            centroid: b.centroid,
            member_count: b.member_ids.len(),
            member_ids: b
                .representative_ids
                .iter()
                .map(ToString::to_string)
                .collect(),
            representative_ids: b
                .representative_ids
                .iter()
                .map(ToString::to_string)
                .collect(),
            updated_at: rfc3339(b.updated_at),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{RecallContext, ScoreBreakdown};
    use serde_json::Value;
    use uuid::Uuid;

    #[test]
    fn wire_memory_omits_internal_fields() {
        let mut m = MemoryRecord::new("claude", "c/1", "hello");
        m.speaker = Some("human".into());
        m.turn_index = Some(0);
        let v: Value = serde_json::to_value(WireMemoryRecord::from(&m)).unwrap();
        let obj = v.as_object().unwrap();
        assert!(obj.contains_key("id"));
        assert!(obj.contains_key("text"));
        assert!(obj.contains_key("ingested_at"));
        assert!(!obj.contains_key("content_sha256"));
        assert!(!obj.contains_key("attachments"));
        assert!(!obj.contains_key("metadata"));
        assert!(!obj.contains_key("source_file"));
        assert!(!obj.contains_key("parent_id"));
        assert!(Uuid::parse_str(obj["id"].as_str().unwrap()).is_ok());
        let ingested = obj["ingested_at"].as_str().unwrap();
        assert!(ingested.ends_with('Z') || ingested.contains('+'));
    }

    #[test]
    fn wire_recall_omits_context() {
        let m = MemoryRecord::new("claude", "c/1", "hello");
        let hit = RecallHit {
            memory: m,
            context: RecallContext {
                before: vec![],
                after: vec![],
            },
            basin_id: Some("basin-1".into()),
            basin_label: Some("Dream physics".into()),
            scores: ScoreBreakdown {
                bm25: 0.5,
                cosine: 0.4,
                radiance: 0.3,
                radiance_weight: 0.1,
                final_score: 0.45,
            },
        };
        let v: Value = serde_json::to_value(WireRecallHit::from(&hit)).unwrap();
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("context"));
        assert!(obj.contains_key("memory"));
        assert!(obj.contains_key("scores"));
        assert!(
            !obj["memory"]
                .as_object()
                .unwrap()
                .contains_key("content_sha256")
        );
    }

    #[test]
    fn wire_basin_omits_parent_id() {
        let b = Basin {
            id: "a1b2c3d4-e5f6-4f97-a628-c1e477206b3d".into(),
            parent_id: Some("root".into()),
            label: "Dream physics".into(),
            path: "memory/research/physics".into(),
            summary: "force law".into(),
            label_state: "stable".into(),
            stability: 0.88,
            centroid: [3.8, -0.6, 1.2],
            member_ids: vec![Uuid::nil()],
            representative_ids: vec![Uuid::nil()],
            updated_at: Utc::now(),
        };
        let v: Value = serde_json::to_value(WireBasin::from(&b)).unwrap();
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("parent_id"));
        assert_eq!(obj["label_state"], "stable");
        assert!(!obj["member_ids"][0].as_str().unwrap().is_empty());
        let updated = obj["updated_at"].as_str().unwrap();
        assert!(updated.ends_with('Z') || updated.contains('+'));
    }

    #[test]
    fn wire_basin_maps_locked_label_state() {
        let b = Basin {
            id: "basin-1".into(),
            parent_id: None,
            label: "Locked".into(),
            path: "memory/locked".into(),
            summary: "".into(),
            label_state: "locked".into(),
            stability: 1.2, // clamp on wire
            centroid: [0.0, 0.0, 0.0],
            member_ids: vec![],
            representative_ids: vec![],
            updated_at: Utc::now(),
        };
        let wire = WireBasin::from(&b);
        assert_eq!(wire.label_state, "stable");
        assert_eq!(wire.stability, 1.0);
    }
}
