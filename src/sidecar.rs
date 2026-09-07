//! Epistemic sidecar: belief about a splat, never the splat.
//!
//! `MemoryRecord.text` is the evidence. Its embedding is frozen. This store answers
//! whether that evidence is currently admissible. Updating it must not re-embed.

use crate::record::{RecallHit, ScoreBreakdown};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

const KEY: &str = "splatrag.sidecar";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SidecarStatus {
    #[default]
    Active,
    Superseded,
    Retracted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicPolicy {
    /// Leave ranking alone. Default so SciFact / freeze numbers do not move.
    #[default]
    PassThrough,
    /// Now-production: drop retracted/superseded, penalize low authority.
    CurrentProduction,
    /// Historical: superseded is ok if `t` is inside the validity interval.
    HistoricalResearch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SidecarState {
    #[serde(default = "one")]
    pub authority: f32,
    #[serde(default)]
    pub status: SidecarStatus,
    #[serde(default)]
    pub superseded_by: Vec<Uuid>,
    #[serde(default)]
    pub valid_from: Option<DateTime<Utc>>,
    #[serde(default)]
    pub valid_until: Option<DateTime<Utc>>,
}

fn one() -> f32 {
    1.0
}

impl Default for SidecarState {
    fn default() -> Self {
        Self {
            authority: 1.0,
            status: SidecarStatus::Active,
            superseded_by: Vec::new(),
            valid_from: None,
            valid_until: None,
        }
    }
}

impl SidecarState {
    pub fn valid_at(&self, t: DateTime<Utc>) -> bool {
        if let Some(from) = self.valid_from
            && t < from
        {
            return false;
        }
        if let Some(until) = self.valid_until
            && t >= until
        {
            return false;
        }
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decision {
    Retain,
    Penalize(f32),
    Drop,
}

/// Semantic retrieval proposes evidence. This decides admissibility.
pub fn decide(state: &SidecarState, policy: EpistemicPolicy, t: DateTime<Utc>) -> Decision {
    match policy {
        EpistemicPolicy::PassThrough => Decision::Retain,
        EpistemicPolicy::CurrentProduction => {
            if state.status == SidecarStatus::Retracted {
                return Decision::Drop;
            }
            if state.status == SidecarStatus::Superseded || !state.superseded_by.is_empty() {
                return Decision::Drop;
            }
            if !state.valid_at(t) {
                return Decision::Drop;
            }
            if state.authority < 0.35 {
                return Decision::Drop;
            }
            if state.authority < 0.7 {
                return Decision::Penalize(state.authority);
            }
            Decision::Retain
        }
        EpistemicPolicy::HistoricalResearch => {
            if state.status == SidecarStatus::Retracted {
                return Decision::Drop;
            }
            if !state.valid_at(t) {
                return Decision::Drop;
            }
            if state.authority < 0.2 {
                return Decision::Penalize(state.authority.max(0.05));
            }
            Decision::Retain
        }
    }
}

pub fn apply_decision(scores: &ScoreBreakdown, decision: Decision) -> Option<ScoreBreakdown> {
    match decision {
        Decision::Drop => None,
        Decision::Retain => Some(scores.clone()),
        Decision::Penalize(v) => {
            let mut s = scores.clone();
            s.final_score *= v.clamp(0.0, 1.0);
            Some(s)
        }
    }
}

/// Never concatenate sidecar into embed text. This is the gate.
pub fn embed_payload(text: &str) -> &str {
    text
}

pub fn metadata_is_sidecar_key(key: &str) -> bool {
    key == KEY
}

#[derive(Debug, Clone)]
pub struct SidecarStore {
    path: PathBuf,
    states: HashMap<Uuid, SidecarState>,
    pub policy: EpistemicPolicy,
}

impl SidecarStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let states = if path.exists() {
            let raw = fs::read_to_string(&path)
                .with_context(|| format!("reading sidecar {}", path.display()))?;
            if raw.trim().is_empty() {
                HashMap::new()
            } else {
                serde_json::from_str(&raw)
                    .with_context(|| format!("invalid sidecar {}", path.display()))?
            }
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            states,
            policy: EpistemicPolicy::PassThrough,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn get(&self, id: Uuid) -> SidecarState {
        self.states.get(&id).cloned().unwrap_or_default()
    }

    pub fn put(&mut self, id: Uuid, state: SidecarState) -> Result<()> {
        if state == SidecarState::default() {
            self.states.remove(&id);
        } else {
            self.states.insert(id, state);
        }
        self.flush()
    }

    pub fn filter_hits(&self, hits: Vec<RecallHit>, t: DateTime<Utc>) -> Vec<RecallHit> {
        if self.policy == EpistemicPolicy::PassThrough {
            return hits;
        }
        let mut out = Vec::with_capacity(hits.len());
        for hit in hits {
            let state = self.get(hit.memory.id);
            let decision = decide(&state, self.policy, t);
            if let Some(scores) = apply_decision(&hit.scores, decision) {
                out.push(RecallHit { scores, ..hit });
            }
        }
        out.sort_by(|a, b| b.scores.final_score.total_cmp(&a.scores.final_score));
        out
    }

    fn flush(&self) -> Result<()> {
        let tmp = self.path.with_extension("json.tmp");
        {
            let file = File::create(&tmp)
                .with_context(|| format!("writing sidecar tmp {}", tmp.display()))?;
            let mut w = BufWriter::new(file);
            serde_json::to_writer(&mut w, &self.states)?;
            w.flush()?;
        }
        fs::rename(&tmp, &self.path)
            .with_context(|| format!("installing sidecar {}", self.path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::MemoryRecord;

    fn hit(text: &str, score: f32) -> RecallHit {
        RecallHit {
            memory: MemoryRecord::new("t", text, text),
            context: Default::default(),
            basin_id: None,
            basin_label: None,
            scores: ScoreBreakdown {
                bm25: score,
                cosine: score,
                radiance: 0.0,
                radiance_weight: 0.0,
                final_score: score,
            },
        }
    }

    #[test]
    fn default_is_admissible() {
        let s = SidecarState::default();
        assert_eq!(
            decide(&s, EpistemicPolicy::CurrentProduction, Utc::now()),
            Decision::Retain
        );
    }

    #[test]
    fn superseded_drops_in_production_keeps_in_history() {
        let s = SidecarState {
            status: SidecarStatus::Superseded,
            superseded_by: vec![Uuid::nil()],
            ..Default::default()
        };
        let t = Utc::now();
        assert_eq!(
            decide(&s, EpistemicPolicy::CurrentProduction, t),
            Decision::Drop
        );
        assert_eq!(
            decide(&s, EpistemicPolicy::HistoricalResearch, t),
            Decision::Retain
        );
        assert_eq!(decide(&s, EpistemicPolicy::PassThrough, t), Decision::Retain);
    }

    #[test]
    fn low_authority_penalizes_not_same_as_retract() {
        let low = SidecarState {
            authority: 0.5,
            ..Default::default()
        };
        let dead = SidecarState {
            status: SidecarStatus::Retracted,
            ..Default::default()
        };
        let t = Utc::now();
        assert_eq!(
            decide(&low, EpistemicPolicy::CurrentProduction, t),
            Decision::Penalize(0.5)
        );
        assert_eq!(
            decide(&dead, EpistemicPolicy::CurrentProduction, t),
            Decision::Drop
        );
    }

    #[test]
    fn interval_semantics() {
        let t0 = DateTime::parse_from_rfc3339("2023-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let t1 = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let s = SidecarState {
            valid_from: Some(t0),
            valid_until: Some(t1),
            ..Default::default()
        };
        let mid = DateTime::parse_from_rfc3339("2023-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(s.valid_at(mid));
        assert!(!s.valid_at(t1));
        assert_eq!(
            decide(&s, EpistemicPolicy::HistoricalResearch, t1),
            Decision::Drop
        );
    }

    #[test]
    fn pass_through_does_not_touch_hits() {
        let dir = tempfile::tempdir().unwrap();
        let store = SidecarStore::open(dir.path().join("epistemic.json")).unwrap();
        let hits = vec![hit("wal replication", 1.0)];
        let out = store.filter_hits(hits.clone(), Utc::now());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].scores.final_score, 1.0);
    }

    #[test]
    fn production_policy_drops_superseded_hit() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SidecarStore::open(dir.path().join("epistemic.json")).unwrap();
        store.policy = EpistemicPolicy::CurrentProduction;
        let h = hit("old policy", 0.9);
        let id = h.memory.id;
        store
            .put(
                id,
                SidecarState {
                    status: SidecarStatus::Superseded,
                    ..Default::default()
                },
            )
            .unwrap();
        let out = store.filter_hits(vec![h], Utc::now());
        assert!(out.is_empty());
    }

    #[test]
    fn sidecar_write_does_not_change_text_or_hash() {
        let rec = MemoryRecord::new("src", "k", "PostgreSQL uses WAL");
        let text_before = rec.text.clone();
        let hash_before = rec.content_sha256.clone();
        let dir = tempfile::tempdir().unwrap();
        let mut store = SidecarStore::open(dir.path().join("epistemic.json")).unwrap();
        store
            .put(
                rec.id,
                SidecarState {
                    authority: 0.2,
                    status: SidecarStatus::Superseded,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(rec.text, text_before);
        assert_eq!(rec.content_sha256, hash_before);
        assert_eq!(embed_payload(&rec.text), "PostgreSQL uses WAL");
        assert!(!embed_payload(&rec.text).contains("Superseded"));
        assert!(metadata_is_sidecar_key(KEY));
    }

    #[test]
    fn put_roundtrips_sparse_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epistemic.json");
        let id = Uuid::new_v4();
        {
            let mut store = SidecarStore::open(&path).unwrap();
            store
                .put(
                    id,
                    SidecarState {
                        authority: 0.4,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let store = SidecarStore::open(&path).unwrap();
        assert!((store.get(id).authority - 0.4).abs() < 1e-6);
        assert_eq!(store.get(Uuid::new_v4()).authority, 1.0);
    }
}
