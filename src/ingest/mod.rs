use crate::record::{AttachmentRef, MemoryRecord};
use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use clap::ValueEnum;
use serde::de::{DeserializeSeed, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum SourceKind {
    Auto,
    Claude,
    Grok,
    Gemini,
    AgentJsonl,
    SemanticMd,
    Jsonl,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestReport {
    pub files: usize,
    pub source_items: usize,
    pub emitted: usize,
    pub rejected: usize,
}

pub struct Ingestor {
    quarantine_path: PathBuf,
}

impl Ingestor {
    pub fn new(quarantine_path: impl Into<PathBuf>) -> Self {
        Self {
            quarantine_path: quarantine_path.into(),
        }
    }

    pub fn ingest_path<F>(
        &self,
        kind: SourceKind,
        path: &Path,
        domain: &str,
        mut emit: F,
    ) -> Result<IngestReport>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        let mut report = IngestReport::default();
        self.walk(kind, path, domain, &mut report, &mut emit)?;
        Ok(report)
    }

    fn walk<F>(
        &self,
        kind: SourceKind,
        path: &Path,
        domain: &str,
        report: &mut IngestReport,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        if path.is_dir() {
            let mut children: Vec<_> = fs::read_dir(path)?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .collect();
            children.sort();
            for child in children {
                if child.is_dir() || is_supported_file(&child) {
                    self.walk(kind, &child, domain, report, emit)?;
                }
            }
            return Ok(());
        }

        report.files += 1;
        let resolved = resolve_kind(kind, path);
        match resolved {
            SourceKind::Claude => self.ingest_json_array(
                path,
                report,
                |value| claude_records(path, value, domain),
                emit,
            ),
            SourceKind::Grok => self.ingest_json_array(
                path,
                report,
                |value| grok_records(path, value, domain),
                emit,
            ),
            SourceKind::Gemini if extension(path) == "html" => {
                self.ingest_gemini_html(path, domain, report, emit)
            }
            SourceKind::Gemini | SourceKind::AgentJsonl | SourceKind::Jsonl => {
                self.ingest_jsonl(resolved, path, domain, report, emit)
            }
            SourceKind::SemanticMd => self.ingest_semantic_md(path, domain, report, emit),
            SourceKind::Auto => unreachable!("auto source kind must be resolved"),
        }
    }

    fn ingest_json_array<F, G>(
        &self,
        path: &Path,
        report: &mut IngestReport,
        mut convert: G,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
        G: FnMut(Value) -> Result<Vec<MemoryRecord>>,
    {
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(1024 * 1024, file);
        let mut deserializer = serde_json::Deserializer::from_reader(reader);
        let mut callback = |value: Value| -> Result<()> {
            report.source_items += 1;
            match convert(value) {
                Ok(records) => {
                    for record in records {
                        emit(record)?;
                        report.emitted += 1;
                    }
                }
                Err(error) => {
                    report.rejected += 1;
                    self.quarantine(path, report.source_items, &error.to_string())?;
                }
            }
            Ok(())
        };
        ArraySeed {
            callback: &mut callback,
        }
        .deserialize(&mut deserializer)
        .with_context(|| format!("failed to stream JSON array {}", path.display()))?;
        Ok(())
    }

    fn ingest_jsonl<F>(
        &self,
        kind: SourceKind,
        path: &Path,
        domain: &str,
        report: &mut IngestReport,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        let file = File::open(path)?;
        for (line_index, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            report.source_items += 1;
            let result = serde_json::from_str::<Value>(&line)
                .map(|value| jsonl_record(kind, path, line_index, value, domain));
            match result {
                Ok(Some(record)) => {
                    emit(record)?;
                    report.emitted += 1;
                }
                Ok(None) => {}
                Err(error) => {
                    report.rejected += 1;
                    self.quarantine(path, line_index + 1, &error.to_string())?;
                }
            }
        }
        Ok(())
    }

    fn ingest_semantic_md<F>(
        &self,
        path: &Path,
        domain: &str,
        report: &mut IngestReport,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        let file = File::open(path)?;
        let mut current: Option<(usize, String)> = None;
        let mut flush = |line_number: usize, block: String| -> Result<()> {
            report.source_items += 1;
            match semantic_md_record(path, line_number, &block, domain) {
                Ok(record) => {
                    emit(record)?;
                    report.emitted += 1;
                }
                Err(error) => {
                    report.rejected += 1;
                    self.quarantine(path, line_number, &error.to_string())?;
                }
            }
            Ok(())
        };

        for (line_index, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.starts_with("- **") {
                if let Some((start, block)) = current.take() {
                    flush(start, block)?;
                }
                current = Some((line_index + 1, line));
            } else if let Some((_, block)) = &mut current
                && !line.trim().is_empty()
            {
                block.push('\n');
                block.push_str(line.trim());
            }
        }
        if let Some((start, block)) = current {
            flush(start, block)?;
        }
        Ok(())
    }

    fn ingest_gemini_html<F>(
        &self,
        path: &Path,
        domain: &str,
        report: &mut IngestReport,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        let mut html = String::new();
        BufReader::new(File::open(path)?).read_to_string(&mut html)?;
        let marker = "content-cell";
        for (index, section) in html.split(marker).skip(1).enumerate() {
            let content = section
                .split_once('>')
                .map(|(_, content)| content)
                .unwrap_or(section);
            let text = strip_html(content.split("</div>").next().unwrap_or_default());
            if text.trim().is_empty() {
                continue;
            }
            report.source_items += 1;
            let source_key = format!("{}#activity-{}", path.display(), index);
            let mut record = MemoryRecord::new("gemini-takeout", source_key, text);
            record.domain = domain.into();
            record.source_file = Some(path.display().to_string());
            record.turn_index = Some(index as u64);
            emit(record)?;
            report.emitted += 1;
        }
        Ok(())
    }

    fn quarantine(&self, path: &Path, item: usize, error: &str) -> Result<()> {
        if let Some(parent) = self.quarantine_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.quarantine_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "source_file": path,
                "source_item": item,
                "error": error,
                "quarantined_at": Utc::now(),
            }),
        )?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }
}

struct ArraySeed<'a, F> {
    callback: &'a mut F,
}

impl<'de, F> DeserializeSeed<'de> for ArraySeed<'_, F>
where
    F: FnMut(Value) -> Result<()>,
{
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<(), D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(ArrayVisitor {
            callback: self.callback,
        })
    }
}

struct ArrayVisitor<'a, F> {
    callback: &'a mut F,
}

impl<'de, F> Visitor<'de> for ArrayVisitor<'_, F>
where
    F: FnMut(Value) -> Result<()>,
{
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON array")
    }

    fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        while let Some(value) = sequence.next_element::<Value>()? {
            (self.callback)(value).map_err(serde::de::Error::custom)?;
        }
        Ok(())
    }
}

fn claude_records(path: &Path, value: Value, domain: &str) -> Result<Vec<MemoryRecord>> {
    let conversation_id = string_at(&value, &["uuid"])
        .or_else(|| string_at(&value, &["id"]))
        .unwrap_or_else(|| format!("conversation-{}", stable_value_fragment(&value)));
    let conversation_name = string_at(&value, &["name"]);
    let messages = value
        .get("chat_messages")
        .or_else(|| value.get("messages"))
        .and_then(Value::as_array)
        .context("Claude conversation has no chat_messages array")?;
    let mut records = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let Some(text) = text_from_value(
            message
                .get("text")
                .or_else(|| message.get("content"))
                .unwrap_or(&Value::Null),
        ) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let message_id = string_at(message, &["uuid"])
            .or_else(|| string_at(message, &["id"]))
            .unwrap_or_else(|| index.to_string());
        let source_key = format!("{conversation_id}/{message_id}");
        let mut record = MemoryRecord::new("claude", source_key, text);
        record.domain = domain.into();
        record.source_file = Some(path.display().to_string());
        record.source_record_id = Some(message_id);
        record.conversation_id = Some(conversation_id.clone());
        record.turn_index = Some(index as u64);
        record.speaker = string_at(message, &["sender"]).or_else(|| string_at(message, &["role"]));
        record.model = string_at(message, &["model"]);
        record.parent_id = string_at(message, &["parent_message_uuid"])
            .or_else(|| string_at(message, &["parent_id"]));
        record.timestamp = timestamp_at(message);
        record.attachments = attachment_values(message);
        if let Some(name) = &conversation_name {
            record
                .metadata
                .insert("conversation_name".into(), Value::String(name.clone()));
        }
        records.push(record);
    }
    Ok(records)
}

fn grok_records(path: &Path, value: Value, domain: &str) -> Result<Vec<MemoryRecord>> {
    let conversation = value.get("conversation").unwrap_or(&value);
    let conversation_id = string_at(conversation, &["id"])
        .or_else(|| string_at(conversation, &["uuid"]))
        .or_else(|| string_at(conversation, &["conversation_id"]))
        .unwrap_or_else(|| format!("conversation-{}", stable_value_fragment(conversation)));
    let responses = value
        .get("responses")
        .or_else(|| value.get("messages"))
        .and_then(Value::as_array)
        .context("Grok conversation has no responses array")?;
    let mut records = Vec::with_capacity(responses.len());
    for (index, wrapper) in responses.iter().enumerate() {
        let response = wrapper.get("response").unwrap_or(wrapper);
        let Some(text) = ["text", "message", "content", "response"]
            .iter()
            .find_map(|key| response.get(*key).and_then(text_from_value))
        else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let message_id = ["id", "uuid", "response_id"]
            .iter()
            .find_map(|key| string_at(response, &[*key]))
            .unwrap_or_else(|| index.to_string());
        let mut record = MemoryRecord::new("grok", format!("{conversation_id}/{message_id}"), text);
        record.domain = domain.into();
        record.source_file = Some(path.display().to_string());
        record.source_record_id = Some(message_id);
        record.conversation_id = Some(conversation_id.clone());
        record.turn_index = Some(index as u64);
        record.speaker = ["sender", "role", "author"]
            .iter()
            .find_map(|key| string_at(response, &[*key]));
        record.model = string_at(response, &["model"]).or_else(|| string_at(wrapper, &["model"]));
        record.parent_id = string_at(response, &["parent_id"]);
        record.timestamp = timestamp_at(response);
        record.attachments = attachment_values(response);
        records.push(record);
    }
    Ok(records)
}

fn jsonl_record(
    kind: SourceKind,
    path: &Path,
    line_index: usize,
    value: Value,
    domain: &str,
) -> Option<MemoryRecord> {
    let body = value.get("message").unwrap_or(&value);
    let mut text = ["text", "content", "message", "prompt", "response"]
        .iter()
        .find_map(|key| body.get(*key).and_then(text_from_value))
        .or_else(|| text_from_value(body))?;
    if let Some(title) = body.get("title").and_then(text_from_value)
        && !title.trim().is_empty()
    {
        text = format!("{}\n{}", title.trim(), text.trim());
    }
    if text.trim().is_empty() {
        return None;
    }
    let source = match kind {
        SourceKind::Gemini => "gemini-cli",
        SourceKind::AgentJsonl => "agent-jsonl",
        _ => "jsonl",
    };
    let record_id = ["_id", "uuid", "id", "message_id"]
        .iter()
        .find_map(|key| string_at(&value, &[*key]))
        .unwrap_or_else(|| (line_index + 1).to_string());
    let conversation_id = ["conversation_id", "session_id", "sessionId"]
        .iter()
        .find_map(|key| string_at(&value, &[*key]));
    let source_key = format!(
        "{}#{}#{}",
        path.display(),
        conversation_id.as_deref().unwrap_or("session"),
        record_id
    );
    let mut record = MemoryRecord::new(source, source_key, text);
    record.domain = domain.into();
    record.source_file = Some(path.display().to_string());
    record.source_record_id = Some(record_id);
    record.conversation_id = conversation_id;
    record.turn_index = Some(line_index as u64);
    record.speaker = ["role", "sender", "type"]
        .iter()
        .find_map(|key| string_at(body, &[*key]))
        .or_else(|| string_at(&value, &["type"]));
    record.model = string_at(body, &["model"]).or_else(|| string_at(&value, &["model"]));
    record.parent_id =
        string_at(&value, &["parent_id"]).or_else(|| string_at(&value, &["parentUuid"]));
    record.timestamp = timestamp_at(&value).or_else(|| timestamp_at(body));
    record.attachments = attachment_values(body);
    Some(record)
}

fn semantic_md_record(
    path: &Path,
    line_number: usize,
    block: &str,
    domain: &str,
) -> Result<MemoryRecord> {
    let timestamp_end = block
        .get(4..)
        .and_then(|tail| tail.find("**").map(|offset| offset + 4))
        .context("missing timestamp terminator")?;
    let timestamp_text = block
        .get(4..timestamp_end)
        .context("invalid timestamp slice")?
        .trim();
    let rest = block[timestamp_end + 2..].trim();
    let key_start = rest.find('[').context("missing memory key")?;
    let key_end = rest[key_start + 1..]
        .find(']')
        .map(|offset| key_start + 1 + offset)
        .context("missing memory key terminator")?;
    let key = &rest[key_start + 1..key_end];
    let after_key = rest[key_end + 1..].trim();
    let (agent, text) = if after_key.starts_with('(') {
        let close = after_key.find(')').context("missing agent terminator")?;
        (
            Some(after_key[1..close].trim().to_string()),
            after_key[close + 1..].trim().to_string(),
        )
    } else {
        (None, after_key.to_string())
    };
    if text.is_empty() {
        anyhow::bail!("empty semantic memory");
    }
    let mut record = MemoryRecord::new(
        "semantic-md",
        format!("{}#{key}#{line_number}", path.display()),
        text,
    );
    record.domain = domain.into();
    record.source_file = Some(path.display().to_string());
    record.source_record_id = Some(key.into());
    record.speaker = agent;
    record.timestamp = parse_timestamp(timestamp_text);
    Ok(record)
}

fn resolve_kind(kind: SourceKind, path: &Path) -> SourceKind {
    if !matches!(kind, SourceKind::Auto) {
        return kind;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_lowercase();
    match extension(path).as_str() {
        "jsonl" | "ndjson" => SourceKind::AgentJsonl,
        "md" => SourceKind::SemanticMd,
        "html" => SourceKind::Gemini,
        _ if name.contains("claude") || name == "conversations.json" => SourceKind::Claude,
        _ if name.contains("grok") || name.contains("export_conversations") => SourceKind::Grok,
        _ => SourceKind::Jsonl,
    }
}

fn is_supported_file(path: &Path) -> bool {
    matches!(
        extension(path).as_str(),
        "json" | "jsonl" | "ndjson" | "md" | "html"
    )
}

fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_lowercase()
}

fn string_at(value: &Value, path: &[&str]) -> Option<String> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    match current {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn text_from_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let parts: Vec<_> = items.iter().filter_map(text_from_value).collect();
            (!parts.is_empty()).then(|| parts.join("\n"))
        }
        Value::Object(object) => ["text", "content", "value", "message"]
            .iter()
            .find_map(|key| object.get(*key).and_then(text_from_value)),
        _ => None,
    }
}

fn timestamp_at(value: &Value) -> Option<DateTime<Utc>> {
    for key in [
        "created_at",
        "updated_at",
        "timestamp",
        "create_time",
        "createTime",
        "time",
    ] {
        let Some(raw) = value.get(key) else {
            continue;
        };
        match raw {
            Value::String(text) => {
                if let Some(timestamp) = parse_timestamp(text) {
                    return Some(timestamp);
                }
            }
            Value::Number(number) => {
                if let Some(mut value) = number.as_i64() {
                    if value > 10_000_000_000 {
                        value /= 1000;
                    }
                    if let Some(timestamp) = Utc.timestamp_opt(value, 0).single() {
                        return Some(timestamp);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_timestamp(text: &str) -> Option<DateTime<Utc>> {
    if let Ok(value) = DateTime::parse_from_rfc3339(text) {
        return Some(value.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"] {
        if let Ok(value) = NaiveDateTime::parse_from_str(text, format) {
            return Some(value.and_utc());
        }
    }
    None
}

fn attachment_values(value: &Value) -> Vec<AttachmentRef> {
    ["attachments", "files", "images"]
        .iter()
        .filter_map(|key| value.get(*key).and_then(Value::as_array))
        .flatten()
        .map(|attachment| match attachment {
            Value::String(path) => AttachmentRef {
                path: Some(path.clone()),
                name: None,
                media_type: None,
                sha256: None,
                metadata: Map::new(),
            },
            Value::Object(object) => AttachmentRef {
                path: ["path", "file_path", "url"]
                    .iter()
                    .find_map(|key| object.get(*key).and_then(Value::as_str))
                    .map(str::to_string),
                name: ["name", "file_name"]
                    .iter()
                    .find_map(|key| object.get(*key).and_then(Value::as_str))
                    .map(str::to_string),
                media_type: ["media_type", "mime_type", "type"]
                    .iter()
                    .find_map(|key| object.get(*key).and_then(Value::as_str))
                    .map(str::to_string),
                sha256: object
                    .get("sha256")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                metadata: object.clone(),
            },
            _ => AttachmentRef {
                path: None,
                name: None,
                media_type: None,
                sha256: None,
                metadata: Map::new(),
            },
        })
        .collect()
}

fn stable_value_fragment(value: &Value) -> String {
    crate::record::sha256_hex(value.to_string().as_bytes())[..16].to_string()
}

fn strip_html(html: &str) -> String {
    let mut output = String::new();
    let mut in_tag = false;
    for character in html.chars() {
        match character {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                output.push(' ');
            }
            _ if !in_tag => output.push(character),
            _ => {}
        }
    }
    output
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_claude_array_and_preserves_two_messages() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("claude.json");
        fs::write(
            &path,
            r#"[{"uuid":"c1","name":"test","chat_messages":[
                {"uuid":"m1","sender":"human","text":"hello","created_at":"2026-01-01T00:00:00Z"},
                {"uuid":"m2","sender":"assistant","text":"world","parent_message_uuid":"m1"}
            ]}]"#,
        )
        .unwrap();
        let ingestor = Ingestor::new(temp.path().join("errors.jsonl"));
        let mut records = Vec::new();
        let report = ingestor
            .ingest_path(SourceKind::Claude, &path, "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        assert_eq!(report.emitted, 2);
        assert_eq!(records[0].conversation_id.as_deref(), Some("c1"));
        assert_eq!(records[1].parent_id.as_deref(), Some("m1"));
    }

    #[test]
    fn semantic_markdown_continuations_stay_together() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("memory.md");
        fs::write(
            &path,
            "- **2026-03-31 04:00** [key] (Shep) first line\n  second line\n",
        )
        .unwrap();
        let ingestor = Ingestor::new(temp.path().join("errors.jsonl"));
        let mut records = Vec::new();
        ingestor
            .ingest_path(SourceKind::SemanticMd, &path, "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].text.contains("second line"));
    }

    #[test]
    fn scifact_jsonl_keeps_beir_key_and_title() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("corpus.jsonl");
        fs::write(
            &path,
            r#"{"_id":"42","title":"Stable keys","text":"sort equal scores by key"}"#,
        )
        .unwrap();
        let ingestor = Ingestor::new(temp.path().join("errors.jsonl"));
        let mut records = Vec::new();
        ingestor
            .ingest_path(SourceKind::Jsonl, &path, "scifact", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        assert_eq!(records[0].source_record_id.as_deref(), Some("42"));
        assert_eq!(records[0].text, "Stable keys\nsort equal scores by key");
    }

    #[test]
    fn agent_jsonl_preserves_session_and_model() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("agent.jsonl");
        fs::write(
            &path,
            r#"{"uuid":"m1","session_id":"s1","message":{"role":"assistant","model":"local","content":"hello"}}"#,
        )
        .unwrap();
        let mut records = Vec::new();
        Ingestor::new(temp.path().join("errors.jsonl"))
            .ingest_path(SourceKind::AgentJsonl, &path, "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        assert_eq!(records[0].conversation_id.as_deref(), Some("s1"));
        assert_eq!(records[0].model.as_deref(), Some("local"));
        assert_eq!(records[0].speaker.as_deref(), Some("assistant"));
    }

    #[test]
    fn gemini_takeout_html_drops_container_attributes() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("MyActivity.html");
        fs::write(
            &path,
            r#"<div class="content-cell mdl-cell mdl-cell--6-col">Asked Gemini<br>about splats</div>"#,
        )
        .unwrap();
        let mut records = Vec::new();
        Ingestor::new(temp.path().join("errors.jsonl"))
            .ingest_path(SourceKind::Gemini, &path, "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        assert_eq!(records[0].text, "Asked Gemini about splats");
    }
}
