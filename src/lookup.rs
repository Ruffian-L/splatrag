//! Durable lookup index.
//!
//! Cold `MemoryRecord` is the source of truth (byte-for-byte, including harness
//! wrappers). Tantivy and HNSW are *derived* from [`lookup_text`], which strips
//! thread-start noise so 13k judge clones and Codex transcript dumps do not own
//! BM25. Rebuild = replay cold through this function.
//!
//! Speaker is a **query** filter. Human turns stay rows. Sidecar is not lookup.
//! Mute (hot) means the clone is not a retrieval candidate; cold still has it.

use crate::record::MemoryRecord;
use std::borrow::Cow;

/// Whole-record harness. Not a memory of Jason — do not index.
const SKIP_PREFIXES: &[&str] = &[
    "You are a strict dataset judge",
    "HARD OUTPUT RULES:",
    "You are an **adversarial verifier",
    "You are the four-drive folder runner",
    "{\"risk_level\":",
];

pub fn is_indexable(record: &MemoryRecord) -> bool {
    !lookup_text(record).trim().is_empty()
}

/// What the derived indexes see. Cold `record.text` is untouched.
pub fn lookup_text(record: &MemoryRecord) -> Cow<'_, str> {
    strip_thread_noise(&record.text)
}

pub fn strip_thread_noise(text: &str) -> Cow<'_, str> {
    let raw = text.trim();
    if raw.is_empty() {
        return Cow::Borrowed("");
    }
    if SKIP_PREFIXES
        .iter()
        .any(|p| raw.starts_with(p) || raw.trim_start().starts_with(p))
    {
        return Cow::Borrowed("");
    }

    if let Some(inner) = last_user_query(raw) {
        let inner = inner.trim();
        if inner.is_empty() {
            return Cow::Borrowed("");
        }
        return Cow::Owned(inner.to_string());
    }

    if raw.contains(">>> TRANSCRIPT") || raw.contains(">>> TRANSCRIPT DELTA") {
        if let Some(user) = first_transcript_user(raw) {
            return Cow::Owned(user);
        }
        return Cow::Borrowed("");
    }

    if raw.starts_with("<user_info>") || raw.starts_with("This session is being continued") {
        return Cow::Borrowed("");
    }

    Cow::Borrowed(text)
}

fn last_user_query(text: &str) -> Option<&str> {
    let mut last = None;
    let mut rest = text;
    while let Some(start) = rest.find("<user_query>") {
        let after = &rest[start + "<user_query>".len()..];
        match after.find("</user_query>") {
            Some(end) => {
                last = Some(after[..end].trim());
                rest = &after[end + "</user_query>".len()..];
            }
            None => {
                last = Some(after.trim());
                break;
            }
        }
    }
    last.filter(|s| !s.is_empty())
}

fn first_transcript_user(text: &str) -> Option<String> {
    // `[1] user: the actual question`
    let idx = text.find("] user:").or_else(|| text.find("\nuser:"))?;
    let after = if text[idx..].starts_with("] user:") {
        &text[idx + "] user:".len()..]
    } else {
        &text[idx + "\nuser:".len()..]
    };
    let line = after.trim_start();
    let end = line
        .find("\n\n")
        .or_else(|| line.find("\n["))
        .unwrap_or(line.len().min(800));
    let out = line[..end].trim();
    if out.is_empty() {
        None
    } else {
        Some(out.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexical::LexicalIndex;
    use crate::record::clean_speaker;

    #[test]
    fn rebuild_indexes_human_turns_as_their_own_rows() {
        let temp = tempfile::tempdir().unwrap();
        let mut human = MemoryRecord::new("test", "u1", "shift click move any window gnome");
        human.speaker = clean_speaker(Some("human".into()));
        human.conversation_id = Some("c1".into());
        human.turn_index = Some(0);
        let mut asst = MemoryRecord::new(
            "test",
            "a1",
            "Super plus left drag. mouse-button-modifier is Super.",
        );
        asst.speaker = clean_speaker(Some("collaborator".into()));
        asst.conversation_id = Some("c1".into());
        asst.turn_index = Some(1);

        assert!(is_indexable(&human));
        assert!(is_indexable(&asst));
        assert_eq!(lookup_text(&human).as_ref(), human.text.as_str());
        assert!(!lookup_text(&asst).contains("shift click"));

        let index =
            LexicalIndex::rebuild(temp.path().join("tantivy"), [human.clone(), asst.clone()])
                .unwrap();
        let hits = index.search("shift click window", 10).unwrap();
        assert!(
            hits.iter().any(|h| h.id == human.id),
            "human turn must survive lexical rebuild, hits={:?}",
            hits.iter().map(|h| h.id).collect::<Vec<_>>()
        );
        assert_eq!(
            hits[0].id, human.id,
            "user phrasing must hit the user row, not the assistant"
        );
    }

    #[test]
    fn empty_text_is_not_indexable() {
        let mut rec = MemoryRecord::new("test", "empty", "x");
        rec.text.clear();
        assert!(!is_indexable(&rec));
    }

    #[test]
    fn judge_prompt_is_not_indexable() {
        let rec = MemoryRecord::new(
            "cli-session",
            "j1",
            "You are a strict dataset judge for a Qwen3.8-27B QLoRA run.\n\nMANDATORY OPERATOR OVERRIDES\nshard-99",
        );
        assert!(!is_indexable(&rec));
        assert_eq!(lookup_text(&rec).as_ref(), "");
    }

    #[test]
    fn user_query_tag_keeps_the_ask() {
        let rec = MemoryRecord::new(
            "agent-jsonl",
            "q1",
            "<user_info>\nOS Version: linux\n</user_info>\n<user_query>\nset up autoingestion into splatrag\n</user_query>",
        );
        assert_eq!(lookup_text(&rec).as_ref(), "set up autoingestion into splatrag");
        assert!(is_indexable(&rec));
    }

    #[test]
    fn transcript_wrapper_keeps_first_user_line() {
        let rec = MemoryRecord::new(
            "codex-cli",
            "t1",
            "The following is the Codex agent history whose request action you are assessing.\n\n>>> TRANSCRIPT START\n\n[1] user: my computer shut off can you tell me why?\n\n[2] assistant: I’ll check the logs.\n",
        );
        let got = lookup_text(&rec);
        assert!(
            got.contains("my computer shut off"),
            "got {got:?}"
        );
        assert!(!got.contains("TRANSCRIPT"));
        assert!(!got.contains("I’ll check"));
    }

    #[test]
    fn ordinary_chat_is_unchanged() {
        let rec = MemoryRecord::new("cli-session", "n1", "Niodoo treats hidden state as physics");
        assert_eq!(lookup_text(&rec).as_ref(), rec.text.as_str());
    }
}
