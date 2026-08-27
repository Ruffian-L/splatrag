use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use uuid::Uuid;

const MEMORY_NAMESPACE: Uuid = Uuid::from_u128(0x5af2_a1b3_4e7c_5d89_9abc_def0_1357_2468);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttachmentRef {
    pub path: Option<String>,
    pub name: Option<String>,
    pub media_type: Option<String>,
    pub sha256: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: Uuid,
    pub source_key: String,
    pub text: String,
    pub content_sha256: String,
    pub domain: String,
    pub timestamp: Option<DateTime<Utc>>,
    pub ingested_at: DateTime<Utc>,
    pub source: String,
    pub source_file: Option<String>,
    pub source_record_id: Option<String>,
    pub speaker: Option<String>,
    pub model: Option<String>,
    pub conversation_id: Option<String>,
    pub parent_id: Option<String>,
    pub turn_index: Option<u64>,
    #[serde(default)]
    pub attachments: Vec<AttachmentRef>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

/// Every field is optional on the wire. A caller narrowing by one axis sends only that
/// axis; without `serde(default)` a partial `filters` object is rejected outright, which
/// makes the filter unusable from MCP and HTTP alike.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RecallFilters {
    pub domains: Vec<String>,
    /// Which archive a memory came from: `claude`, `grok-export`, `gemini`, `claude-memory`, …
    ///
    /// This is the axis for "show me only the Claude thread". It is a *relationship* filter, not a
    /// speaker filter: a source contains both sides of the conversation. Measured on this store,
    /// `claude` is 778 human / 778 assistant and `grok-export` is 453 / 443 — filtering by source
    /// keeps the human in, because the human is half of every thread.
    ///
    /// `source` is also part of the identity key (UUID v5 over `{source}\0{source_key}`), so it is
    /// stable across re-imports and cannot drift.
    pub sources: Vec<String>,
    /// Who produced the text: `human`, `collaborator`.
    ///
    /// Brand seats (claude / grok / sol / chatgpt / assistant) collapse to `collaborator` at
    /// ingest. Older cold rows may still say `assistant`; the filter treats those as the same
    /// seat so a `collaborator` query does not drop the archive.
    ///
    /// **This is the filter that removes a participant.** Use it deliberately — asking for
    /// `collaborator` only is asking for one voice with the other party stripped out, which is a
    /// different thing from asking for a conversation. Prefer [`RecallFilters::sources`] when what
    /// you want is "that relationship's memories".
    pub speakers: Vec<String>,
    pub models: Vec<String>,
    pub basin_id: Option<String>,
    pub conversation_id: Option<String>,
    pub after: Option<DateTime<Utc>>,
    pub before: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub bm25: f32,
    pub cosine: f32,
    pub radiance: f32,
    pub radiance_weight: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallContext {
    pub before: Vec<MemoryRecord>,
    pub after: Vec<MemoryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    pub memory: MemoryRecord,
    pub context: RecallContext,
    pub basin_id: Option<String>,
    pub basin_label: Option<String>,
    pub scores: ScoreBreakdown,
}

/// Named people in the room keep their names. Brand seats and "assistant" do not.
const TEAM_SPEAKERS: &[&str] = &["shep", "echo", "lumina", "nex"];
const HUMAN_SPEAKERS: &[&str] = &["human", "user", "jason", "ruffian", "ruffianl"];

/// Trim, case-fold, and collapse vendor/role names to `collaborator`.
///
/// `assistant`, `claude`, `grok`, `sol`, `chatgpt` (and close variants) are one seat.
/// Shep / Echo / Lumina / Nex stay — they are people in the room, not product names.
pub fn clean_speaker(value: Option<String>) -> Option<String> {
    let trimmed = value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())?;
    let folded = trimmed.to_ascii_lowercase();
    if HUMAN_SPEAKERS.contains(&folded.as_str()) {
        return Some("human".into());
    }
    if TEAM_SPEAKERS.contains(&folded.as_str()) {
        return Some(folded);
    }
    Some("collaborator".into())
}

/// `collaborator` and legacy `assistant` are the same seat for recall filters.
pub fn speaker_matches(wanted: &str, speaker: &str) -> bool {
    if wanted.eq_ignore_ascii_case(speaker) {
        return true;
    }
    let wanted = wanted.to_ascii_lowercase();
    let speaker = speaker.to_ascii_lowercase();
    matches!(
        (wanted.as_str(), speaker.as_str()),
        ("collaborator", "assistant") | ("assistant", "collaborator")
    )
}

impl MemoryRecord {
    pub fn new(
        source: impl Into<String>,
        source_key: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        let source = source.into();
        let source_key = source_key.into();
        let text = text.into();
        let identity = format!("{source}\0{source_key}");
        let id = Uuid::new_v5(&MEMORY_NAMESPACE, identity.as_bytes());
        Self {
            id,
            source_key,
            content_sha256: sha256_hex(normalize_text(&text).as_bytes()),
            text,
            domain: "chat".into(),
            timestamp: None,
            ingested_at: Utc::now(),
            source,
            source_file: None,
            source_record_id: None,
            speaker: None,
            model: None,
            conversation_id: None,
            parent_id: None,
            turn_index: None,
            attachments: Vec::new(),
            metadata: Map::new(),
        }
    }

    pub fn matches(&self, filters: &RecallFilters) -> bool {
        if !filters.domains.is_empty() && !filters.domains.iter().any(|d| d == &self.domain) {
            return false;
        }
        if !filters.sources.is_empty()
            && !filters.sources.iter().any(|wanted| wanted == &self.source)
        {
            return false;
        }
        // Speaker is normalized at ingest. `collaborator` and legacy `assistant` are one seat.
        if !filters.speakers.is_empty()
            && !self.speaker.as_ref().is_some_and(|speaker| {
                filters
                    .speakers
                    .iter()
                    .any(|wanted| speaker_matches(wanted, speaker))
            })
        {
            return false;
        }
        if !filters.models.is_empty()
            && !self
                .model
                .as_ref()
                .is_some_and(|m| filters.models.iter().any(|wanted| wanted == m))
        {
            return false;
        }
        if let Some(conversation_id) = &filters.conversation_id
            && self.conversation_id.as_ref() != Some(conversation_id)
        {
            return false;
        }
        if filters.after.is_some() || filters.before.is_some() {
            let Some(ts) = self.timestamp else {
                return false;
            };
            if let Some(after) = filters.after
                && ts < after
            {
                return false;
            }
            if let Some(before) = filters.before
                && ts > before
            {
                return false;
            }
        }
        true
    }
}

pub fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_source_stable_not_content_dedup() {
        let a = MemoryRecord::new("claude", "conversation/message-1", "same text");
        let again = MemoryRecord::new("claude", "conversation/message-1", "same text");
        let other = MemoryRecord::new("grok", "conversation/message-1", "same text");
        assert_eq!(a.id, again.id);
        assert_ne!(a.id, other.id);
        assert_eq!(a.content_sha256, other.content_sha256);
    }

    /// A caller narrowing by one axis sends only that axis. Requiring every field made
    /// `filters` unusable over MCP and HTTP: a partial object was rejected outright.
    #[test]
    fn recall_filters_accept_a_partial_object() {
        let one: RecallFilters = serde_json::from_str(r#"{"domains":["bench"]}"#).unwrap();
        assert_eq!(one.domains, vec!["bench"]);
        assert!(one.sources.is_empty());
        assert!(one.speakers.is_empty());
        assert!(one.basin_id.is_none());

        let empty: RecallFilters = serde_json::from_str("{}").unwrap();
        assert!(empty.domains.is_empty());
        assert!(empty.after.is_none());

        let speaker: RecallFilters = serde_json::from_str(r#"{"speakers":["human"]}"#).unwrap();
        assert_eq!(speaker.speakers, vec!["human"]);
    }

    #[test]
    fn clean_speaker_collapses_brand_seats_and_keeps_the_room() {
        assert_eq!(
            clean_speaker(Some("ASSISTANT".into())).as_deref(),
            Some("collaborator")
        );
        assert_eq!(
            clean_speaker(Some("claude".into())).as_deref(),
            Some("collaborator")
        );
        assert_eq!(
            clean_speaker(Some("Grok".into())).as_deref(),
            Some("collaborator")
        );
        assert_eq!(
            clean_speaker(Some("sol".into())).as_deref(),
            Some("collaborator")
        );
        assert_eq!(
            clean_speaker(Some("ChatGPT".into())).as_deref(),
            Some("collaborator")
        );
        assert_eq!(clean_speaker(Some("user".into())).as_deref(), Some("human"));
        assert_eq!(clean_speaker(Some("Shep".into())).as_deref(), Some("shep"));
        assert_eq!(clean_speaker(Some("  ".into())), None);
        assert!(speaker_matches("collaborator", "assistant"));
        assert!(speaker_matches("assistant", "collaborator"));
        assert!(!speaker_matches("human", "collaborator"));
    }

    #[test]
    fn date_filter_drops_rows_without_a_clock() {
        use chrono::{TimeZone, Utc};
        let after = Utc.with_ymd_and_hms(2026, 8, 20, 0, 0, 0).unwrap();
        let filters = RecallFilters {
            after: Some(after),
            ..RecallFilters::default()
        };
        let mut missing = MemoryRecord::new("cli-session", "no-clock", "yesetday");
        missing.speaker = Some("human".into());
        assert!(
            !missing.matches(&filters),
            "undated rows must not sneak through after="
        );
        missing.timestamp = Some(Utc.with_ymd_and_hms(2026, 8, 19, 23, 0, 0).unwrap());
        assert!(!missing.matches(&filters));
        missing.timestamp = Some(Utc.with_ymd_and_hms(2026, 8, 20, 4, 16, 0).unwrap());
        assert!(missing.matches(&filters));
    }
}
