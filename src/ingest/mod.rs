pub mod extract;

use crate::record::{AttachmentRef, MemoryRecord};
use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use clap::ValueEnum;
use serde::de::{DeserializeSeed, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
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
    /// A whole Grok account export directory: `prod-grok-backend.json` plus the
    /// `prod-mc-asset-server/<uuid>/content` blobs it references. Distinct from `Grok`, which reads
    /// a bare conversations array and cannot resolve attachments.
    GrokExport,
    Gemini,
    AgentJsonl,
    SemanticMd,
    MemoryMd,
    Jsonl,
}

/// Filenames fixed by the export format itself.
const BACKEND_JSON: &str = "prod-grok-backend.json";
const ASSET_DIR: &str = "prod-mc-asset-server";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestReport {
    pub files: usize,
    pub source_items: usize,
    pub emitted: usize,
    pub rejected: usize,
    /// Assets recognised but still awaiting a model to read them (images) or an unpacker
    /// (pdf/zip). Distinct from `rejected`: nothing is wrong with them yet.
    #[serde(default)]
    pub pending: usize,
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
        // An export is one unit, not a tree to walk: the backend JSON is the join table for the
        // asset blobs beside it, so descending into them separately would lose the linkage.
        // Auto-detection keys on the export's own fixed filename, never on a user-chosen one.
        if matches!(kind, SourceKind::GrokExport) || is_grok_export(kind, path) {
            return self.ingest_grok_export(path, domain, report, emit);
        }

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
            SourceKind::MemoryMd => self.ingest_memory_md(path, domain, report, emit),
            SourceKind::Auto => unreachable!("auto source kind must be resolved"),
            SourceKind::GrokExport => unreachable!("grok exports are handled before the walk"),
        }
    }

    /// Read a Grok account export as one unit.
    ///
    /// Two record streams come out, and they are keyed differently on purpose:
    ///
    /// * **Responses** key on `response._id`, which is already a UUID and globally unique. No path
    ///   and no conversation prefix — moving or re-downloading the export must not fork the record.
    /// * **Assets** key on their asset UUID, taken from the directory name under
    ///   `prod-mc-asset-server/`. The flattened `unnested_files/` copies of these blobs carry
    ///   invented filenames that collide, so a name can never be the key.
    ///
    /// Only ~15% of assets are referenced by any response (238 of 1568 in the export this was
    /// written against). The rest are emitted anyway, parentless, and left to settle by meaning
    /// alone — dropping them would discard the bulk of the archive.
    fn ingest_grok_export<F>(
        &self,
        path: &Path,
        domain: &str,
        report: &mut IngestReport,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        let (root, backend) = if path.is_dir() {
            (path.to_path_buf(), path.join(BACKEND_JSON))
        } else {
            let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
            (root, path.to_path_buf())
        };
        let assets_dir = root.join(ASSET_DIR);

        report.files += 1;
        let raw = fs::read_to_string(&backend)
            .with_context(|| format!("failed to read {}", backend.display()))?;
        let export: Value = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {}", backend.display()))?;

        let conversations = export
            .get("conversations")
            .and_then(Value::as_array)
            .context("export has no conversations array")?;

        let mut attached: HashMap<String, AssetOrigin> = HashMap::new();
        for entry in conversations {
            report.source_items += 1;
            match grok_export_conversation(entry, &assets_dir, domain, &mut attached) {
                Ok(records) => {
                    for record in records {
                        emit(record)?;
                        report.emitted += 1;
                    }
                }
                Err(error) => {
                    report.rejected += 1;
                    self.quarantine(&backend, report.source_items, &error.to_string())?;
                }
            }
        }

        self.ingest_assets(&assets_dir, domain, &attached, report, emit)
    }

    /// Emit one record per asset that carries extractable text.
    ///
    /// Images, PDFs and archives yield no text here — they are recorded as pending rather than
    /// emitted, because a memory whose text is empty matches every query weakly and pollutes the
    /// field. They become records once the OCR/unpack pass fills them in.
    fn ingest_assets<F>(
        &self,
        assets_dir: &Path,
        domain: &str,
        attached: &HashMap<String, AssetOrigin>,
        report: &mut IngestReport,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        if !assets_dir.is_dir() {
            return Ok(());
        }
        let mut entries: Vec<_> = fs::read_dir(assets_dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect();
        entries.sort();

        for entry in entries {
            let Some(asset_id) = entry.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(blob) = asset_blob(&entry) else {
                continue;
            };
            report.files += 1;
            report.source_items += 1;

            let extracted = match extract::extract(asset_id, &blob) {
                Ok(extracted) => extracted,
                Err(error) => {
                    report.rejected += 1;
                    self.quarantine(&blob, report.source_items, &error.to_string())?;
                    continue;
                }
            };
            let Some(text) = extracted.text.as_ref().filter(|text| !text.trim().is_empty()) else {
                report.pending += 1;
                continue;
            };

            let mut record = MemoryRecord::new("grok-asset", asset_id, text.clone());
            record.domain = domain.into();
            record.source_file = Some(blob.display().to_string());
            record.source_record_id = Some(asset_id.to_string());
            record.metadata.insert(
                "media_type".into(),
                Value::String(extracted.media_type.clone()),
            );
            record
                .metadata
                .insert("content_sha256".into(), Value::String(extracted.sha256));
            // A referenced asset inherits its response's place in the archive, so the turns that
            // explain the picture stay reachable from the picture.
            match attached.get(asset_id) {
                Some(origin) => {
                    record.conversation_id = Some(origin.conversation_id.clone());
                    record.parent_id = Some(origin.response_id.clone());
                    record.timestamp = origin.timestamp;
                    record.metadata.insert("orphan".into(), Value::Bool(false));
                    if let Some(title) = &origin.title {
                        record
                            .metadata
                            .insert("conversation_title".into(), Value::String(title.clone()));
                    }
                }
                None => {
                    record.metadata.insert("orphan".into(), Value::Bool(true));
                }
            }
            emit(record)?;
            report.emitted += 1;
        }
        Ok(())
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

    /// One memory file becomes exactly one record. These files are written one-fact-per-file by
    /// design, so the file is already the natural retrieval unit — splitting them would break the
    /// fact apart. `MEMORY.md` is skipped: it is an index of pointers to the others, and ingesting
    /// it would create a record that weakly matches every query.
    fn ingest_memory_md<F>(
        &self,
        path: &Path,
        domain: &str,
        report: &mut IngestReport,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(MemoryRecord) -> Result<()>,
    {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("MEMORY.md"))
        {
            return Ok(());
        }

        let mut raw = String::new();
        BufReader::new(File::open(path)?).read_to_string(&mut raw)?;
        report.source_items += 1;
        match memory_md_record(path, &raw, domain) {
            Ok(record) => {
                emit(record)?;
                report.emitted += 1;
            }
            Err(error) => {
                report.rejected += 1;
                self.quarantine(path, 1, &error.to_string())?;
            }
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

/// Where a referenced asset came from, so the blob can inherit its response's context.
///
/// Without this an attached asset lands with no `conversation_id`, which silently disables
/// [`crate::cold_store::ColdStore::context`] for it — the neighbouring turns that explain what the
/// picture *is* would be unreachable from the picture.
#[derive(Debug, Clone)]
struct AssetOrigin {
    conversation_id: String,
    response_id: String,
    timestamp: Option<DateTime<Utc>>,
    title: Option<String>,
}

/// Trim a field and treat blank as absent.
///
/// Grok writes `"model": ""` on 61 of 1061 responses. Left alone that empty string is a *value*,
/// so a `models` recall filter would have to know to ask for `""` to find them.
fn clean(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Speakers additionally case-fold: the same export writes both `assistant` and `ASSISTANT`, and
/// two spellings of one role would split every grouping and filter that touches it.
fn clean_speaker(value: Option<String>) -> Option<String> {
    clean(value).map(|value| value.to_lowercase())
}

/// Locate the payload for one asset directory. The export stores each blob as
/// `<asset-uuid>/content`; a few entries are plain files (profile pictures) instead of directories.
fn asset_blob(entry: &Path) -> Option<PathBuf> {
    if entry.is_dir() {
        let content = entry.join("content");
        return content.is_file().then_some(content);
    }
    entry.is_file().then(|| entry.to_path_buf())
}

/// Turn one `{conversation, responses}` pair into records, recording which assets it claimed.
fn grok_export_conversation(
    entry: &Value,
    assets_dir: &Path,
    domain: &str,
    attached: &mut HashMap<String, AssetOrigin>,
) -> Result<Vec<MemoryRecord>> {
    let conversation = entry
        .get("conversation")
        .context("export entry has no conversation object")?;
    let conversation_id =
        string_at(conversation, &["id"]).context("conversation has no id")?;
    let title = string_at(conversation, &["title"]).filter(|title| !title.trim().is_empty());
    let responses = entry
        .get("responses")
        .and_then(Value::as_array)
        .context("conversation has no responses array")?;

    let mut records = Vec::with_capacity(responses.len());
    for (index, wrapper) in responses.iter().enumerate() {
        let response = wrapper.get("response").unwrap_or(wrapper);
        let Some(text) = response.get("message").and_then(text_from_value) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        // `_id` is a UUID and unique across the whole export, so it stands alone as the key.
        let response_id = string_at(response, &["_id"])
            .or_else(|| string_at(response, &["id"]))
            .unwrap_or_else(|| format!("{conversation_id}/{index}"));

        let timestamp = timestamp_at(response);
        let mut record = MemoryRecord::new("grok-export", response_id.clone(), text);
        record.domain = domain.into();
        record.source_record_id = Some(response_id.clone());
        record.conversation_id = Some(conversation_id.clone());
        record.turn_index = Some(index as u64);
        record.speaker = clean_speaker(string_at(response, &["sender"]));
        record.model = clean(string_at(response, &["model"]));
        record.parent_id = clean(string_at(response, &["parent_response_id"]));
        record.timestamp = timestamp;
        record.attachments = export_attachments(
            response,
            assets_dir,
            attached,
            &AssetOrigin {
                conversation_id: conversation_id.clone(),
                response_id,
                timestamp,
                title: title.clone(),
            },
        );
        if let Some(title) = &title {
            record
                .metadata
                .insert("conversation_title".into(), Value::String(title.clone()));
        }
        records.push(record);
    }
    Ok(records)
}

/// Resolve `file_attachments: ["<asset-uuid>", ...]` against the asset directory.
///
/// The UUID is retained even when the blob is missing from disk (2 of 240 were, in the export this
/// was written against) so the dangling reference stays visible instead of being silently dropped.
fn export_attachments(
    response: &Value,
    assets_dir: &Path,
    attached: &mut HashMap<String, AssetOrigin>,
    origin: &AssetOrigin,
) -> Vec<AttachmentRef> {
    let Some(ids) = response.get("file_attachments").and_then(Value::as_array) else {
        return Vec::new();
    };
    ids.iter()
        .filter_map(Value::as_str)
        .map(|asset_id| {
            // First reference wins: an asset quoted again later belongs to where it first appeared.
            attached
                .entry(asset_id.to_string())
                .or_insert_with(|| origin.clone());
            let blob = asset_blob(&assets_dir.join(asset_id));
            let media_type = blob
                .as_ref()
                .and_then(|path| fs::read(path).ok())
                .map(|bytes| extract::sniff(&bytes).media_type().to_string());
            let mut metadata = Map::new();
            metadata.insert("asset_id".into(), Value::String(asset_id.to_string()));
            if blob.is_none() {
                metadata.insert("missing".into(), Value::Bool(true));
            }
            AttachmentRef {
                path: blob.map(|path| path.display().to_string()),
                name: Some(asset_id.to_string()),
                media_type,
                sha256: None,
                metadata,
            }
        })
        .collect()
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

/// Front-matter reader for the memory-file format: a `---` fenced block of `key: value` lines where
/// two-space-indented lines nest under the preceding top-level key. Deliberately not a general YAML
/// parser — the format is fixed and four fields do not justify a new dependency.
fn parse_front_matter(raw: &str) -> Option<(Map<String, Value>, String)> {
    let mut lines = raw.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    let mut fields = Map::new();
    let mut nested: Option<(String, Map<String, Value>)> = None;
    let mut body = String::new();
    let mut in_body = false;

    for line in lines {
        if in_body {
            body.push_str(line);
            body.push('\n');
            continue;
        }
        if line.trim() == "---" {
            if let Some((key, map)) = nested.take() {
                fields.insert(key, Value::Object(map));
            }
            in_body = true;
            continue;
        }
        let indented = line.starts_with("  ") || line.starts_with('\t');
        let Some((key, value)) = line.trim().split_once(':') else {
            continue;
        };
        let key = key.trim().to_string();
        let value = unquote(value);
        if indented {
            if let Some((_, map)) = nested.as_mut() {
                map.insert(key, Value::String(value));
            }
        } else {
            if let Some((previous, map)) = nested.take() {
                fields.insert(previous, Value::Object(map));
            }
            if value.is_empty() {
                nested = Some((key, Map::new()));
            } else {
                fields.insert(key, Value::String(value));
            }
        }
    }

    in_body.then(|| (fields, body.trim().to_string()))
}

fn unquote(value: &str) -> String {
    let trimmed = value.trim();
    match trimmed.strip_prefix('"').and_then(|v| v.strip_suffix('"')) {
        Some(inner) => inner.replace("\\\"", "\"").replace("\\\\", "\\"),
        None => trimmed.to_string(),
    }
}

/// Pull `[[slug]]` references out of a body. These are the hand-authored edges between memories.
fn wiki_links(body: &str) -> Vec<String> {
    let mut links: Vec<String> = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("[[") {
        let after = &rest[start + 2..];
        let Some(end) = after.find("]]") else { break };
        let link = after[..end].trim();
        if !link.is_empty() && !links.iter().any(|existing| existing == link) {
            links.push(link.to_string());
        }
        rest = &after[end + 2..];
    }
    links
}

fn memory_md_record(path: &Path, raw: &str, domain: &str) -> Result<MemoryRecord> {
    let (fields, body) =
        parse_front_matter(raw).context("memory file has no `---` front-matter block")?;
    if body.is_empty() {
        anyhow::bail!("memory file has front matter but an empty body");
    }

    let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or_default();
    let name = fields
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .unwrap_or(stem)
        .to_string();
    let description = fields
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let metadata = fields.get("metadata").and_then(Value::as_object);

    // `domain` is a physics parameter, not a label: dream-time repulsion is 35% stronger across
    // domains (physics.rs:97). Keying it on the memory type makes user/feedback/project/reference
    // settle into separate basins instead of smearing together.
    let resolved_domain = metadata
        .and_then(|metadata| metadata.get("type"))
        .and_then(Value::as_str)
        .filter(|kind| !kind.is_empty())
        .unwrap_or(domain)
        .to_string();

    // The slug gives BM25 real terms and the description is a hand-written summary, so both belong
    // in the embedded text rather than sitting in metadata where retrieval cannot see them.
    let text = if description.is_empty() {
        format!("{name}\n\n{body}")
    } else {
        format!("{name}: {description}\n\n{body}")
    };

    // source_key is the slug, so re-importing an edited memory updates in place instead of forking.
    let mut record = MemoryRecord::new("claude-memory", name.clone(), text);
    record.domain = resolved_domain;
    record.source_file = Some(path.display().to_string());
    record.source_record_id = Some(name.clone());
    record.speaker = Some("assistant".into());
    record.timestamp = metadata
        .and_then(|metadata| metadata.get("modified"))
        .and_then(Value::as_str)
        .and_then(|stamp| DateTime::parse_from_rfc3339(stamp).ok())
        .map(|stamp| stamp.with_timezone(&Utc));

    record.metadata.insert("memory_name".into(), Value::String(name));
    if !description.is_empty() {
        record
            .metadata
            .insert("description".into(), Value::String(description));
    }
    if let Some(metadata) = metadata {
        for key in ["type", "node_type", "originSessionId"] {
            if let Some(value) = metadata.get(key) {
                record.metadata.insert(key.to_string(), value.clone());
            }
        }
    }
    let links = wiki_links(&body);
    if !links.is_empty() {
        record.metadata.insert(
            "links".into(),
            Value::Array(links.into_iter().map(Value::String).collect()),
        );
    }
    Ok(record)
}

/// Under `Auto`, recognise an export by the one filename the export format fixes.
fn is_grok_export(kind: SourceKind, path: &Path) -> bool {
    if !matches!(kind, SourceKind::Auto) {
        return false;
    }
    if path.is_dir() {
        return path.join(BACKEND_JSON).is_file();
    }
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == BACKEND_JSON)
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
                if let Some(value) = number.as_i64()
                    && let Some(timestamp) = epoch_to_utc(value)
                {
                    return Some(timestamp);
                }
            }
            // Mongo extended JSON: `{"$date": {"$numberLong": "1785272919201"}}`, and the shorter
            // `{"$date": "2026-07-28T21:07:53Z"}`. Every Grok export response timestamp is the first
            // shape, so without this arm the whole conversation history imports undated.
            Value::Object(_) => {
                if let Some(timestamp) = mongo_timestamp(raw) {
                    return Some(timestamp);
                }
            }
            _ => {}
        }
    }
    None
}

/// Seconds or milliseconds since the epoch, disambiguated by magnitude. The cutoff is year 2286 in
/// seconds, which no export here predates or exceeds, so anything larger is milliseconds.
fn epoch_to_utc(value: i64) -> Option<DateTime<Utc>> {
    if value.abs() > 10_000_000_000 {
        Utc.timestamp_millis_opt(value).single()
    } else {
        Utc.timestamp_opt(value, 0).single()
    }
}

fn mongo_timestamp(value: &Value) -> Option<DateTime<Utc>> {
    let inner = value.get("$date")?;
    match inner {
        Value::String(text) => parse_timestamp(text),
        Value::Number(number) => epoch_to_utc(number.as_i64()?),
        // `$numberLong` is a *string* in canonical extended JSON, because the value can exceed
        // what a JSON number is guaranteed to hold.
        Value::Object(_) => {
            let raw = inner.get("$numberLong")?;
            let millis = match raw {
                Value::String(text) => text.parse::<i64>().ok()?,
                Value::Number(number) => number.as_i64()?,
                _ => return None,
            };
            epoch_to_utc(millis)
        }
        _ => None,
    }
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

    /// Uses the real on-disk shape: escaped quotes inside the description, a trailing space after
    /// `metadata:`, and an em dash — all three appear in the actual memory files.
    fn memory_fixture() -> &'static str {
        "---\nname: feedback-no-blades\ndescription: \"The \\\"blade on the end\\\" pattern \
         hurts — say the warm thing plainly\"\nmetadata: \n  node_type: memory\n  type: feedback\
         \n  originSessionId: 549c56fb\n  modified: 2026-07-21T21:06:31.468Z\n---\n\nSay the warm \
         thing and let it end where it ends.\n\nRelated: [[user-jason-working-style]], \
         [[project-narrator-era-rest]], [[user-jason-working-style]]\n"
    }

    #[test]
    fn memory_md_maps_type_to_domain_and_keeps_links() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("feedback_no_blades.md");
        fs::write(&path, memory_fixture()).unwrap();
        let mut records = Vec::new();
        Ingestor::new(temp.path().join("errors.jsonl"))
            .ingest_path(SourceKind::MemoryMd, &path, "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();

        assert_eq!(records.len(), 1, "one memory file is one record");
        let record = &records[0];
        // `type` wins over the CLI-supplied domain because domain drives dream repulsion.
        assert_eq!(record.domain, "feedback");
        assert_eq!(record.source_key, "feedback-no-blades");
        assert!(record.text.contains("blade on the end\" pattern"), "escaped quotes survive");
        assert!(record.text.contains("let it end where it ends"), "body is embedded");
        assert_eq!(
            record.timestamp.map(|ts| ts.to_rfc3339()),
            Some("2026-07-21T21:06:31.468+00:00".into())
        );
        let links = record.metadata.get("links").unwrap().as_array().unwrap();
        assert_eq!(links.len(), 2, "duplicate wiki-links collapse");
    }

    #[test]
    fn memory_md_is_idempotent_and_skips_the_index() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("feedback_no_blades.md"), memory_fixture()).unwrap();
        fs::write(temp.path().join("MEMORY.md"), "- [No blades](feedback_no_blades.md) — hook\n")
            .unwrap();

        let ingest = |records: &mut Vec<MemoryRecord>| {
            Ingestor::new(temp.path().join("errors.jsonl"))
                .ingest_path(SourceKind::MemoryMd, temp.path(), "chat", |record| {
                    records.push(record);
                    Ok(())
                })
                .unwrap()
        };
        let mut first = Vec::new();
        let report = ingest(&mut first);
        let mut second = Vec::new();
        ingest(&mut second);

        // MEMORY.md is an index of pointers; ingesting it would weakly match every query.
        assert_eq!(first.len(), 1, "the index file is skipped");
        assert_eq!(report.rejected, 0);
        assert_eq!(first[0].id, second[0].id, "re-import updates in place");
    }

    /// Every response timestamp in a Grok export is Mongo extended JSON. Before this parsed, all
    /// 1061 responses imported with `timestamp: None` and no error to show for it.
    #[test]
    fn mongo_extended_json_dates_parse() {
        let millis = serde_json::json!({"create_time": {"$date": {"$numberLong": "1785272919201"}}});
        assert_eq!(
            timestamp_at(&millis).map(|ts| ts.to_rfc3339()),
            Some("2026-07-28T21:08:39.201+00:00".into())
        );
        // The short form, and a plain RFC3339 string, must still work.
        let short = serde_json::json!({"create_time": {"$date": "2026-07-28T21:07:53Z"}});
        assert!(timestamp_at(&short).is_some());
        let plain = serde_json::json!({"create_time": "2026-07-28T21:07:53.018584Z"});
        assert!(timestamp_at(&plain).is_some());
        // Seconds vs milliseconds are told apart by magnitude, not by field name.
        let seconds = serde_json::json!({"timestamp": 1785272919});
        assert_eq!(
            timestamp_at(&seconds).map(|ts| ts.to_rfc3339()),
            Some("2026-07-28T21:08:39+00:00".into())
        );
    }

    /// Builds the export layout on disk: backend JSON plus `<asset-uuid>/content` blobs.
    fn export_fixture(root: &Path) {
        let assets = root.join(ASSET_DIR);
        for (id, body) in [
            ("asset-attached", &b"a note that was attached"[..]),
            ("asset-orphan", &b"a note nothing points at"[..]),
        ] {
            fs::create_dir_all(assets.join(id)).unwrap();
            fs::write(assets.join(id).join("content"), body).unwrap();
        }
        fs::create_dir_all(assets.join("asset-image")).unwrap();
        fs::write(
            assets.join("asset-image").join("content"),
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
        )
        .unwrap();

        fs::write(
            root.join(BACKEND_JSON),
            r#"{"conversations":[{
                "conversation":{"id":"conv-1","title":"Physics of Friendship"},
                "responses":[
                  {"response":{"_id":"resp-1","sender":"human","model":"grok-4",
                    "message":"what did we decide",
                    "create_time":{"$date":{"$numberLong":"1785272919201"}},
                    "file_attachments":["asset-attached","asset-gone"]}},
                  {"response":{"_id":"resp-2","sender":"ASSISTANT","model":"",
                    "message":"we decided to keep the cold log","parent_response_id":"resp-1"}}
                ]}]}"#,
        )
        .unwrap();
    }

    #[test]
    fn grok_export_keys_on_uuids_and_resolves_assets() {
        let temp = tempfile::tempdir().unwrap();
        export_fixture(temp.path());
        let mut records = Vec::new();
        let report = Ingestor::new(temp.path().join("errors.jsonl"))
            .ingest_path(SourceKind::GrokExport, temp.path(), "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();

        assert_eq!(report.rejected, 0);
        // Two responses + two text assets; the PNG waits for OCR instead of becoming empty text.
        assert_eq!(report.emitted, 4);
        assert_eq!(report.pending, 1);

        let first = records.iter().find(|r| r.source == "grok-export").unwrap();
        // The key is the bare response UUID: no path, so moving the export cannot fork the record.
        assert_eq!(first.source_key, "resp-1");
        assert_eq!(first.conversation_id.as_deref(), Some("conv-1"));
        assert_eq!(first.model.as_deref(), Some("grok-4"));
        assert!(first.timestamp.is_some(), "mongo dates must survive ingest");
        assert_eq!(
            first.metadata.get("conversation_title").and_then(Value::as_str),
            Some("Physics of Friendship")
        );

        // A referenced-but-missing blob stays visible as a dangling ref rather than vanishing.
        assert_eq!(first.attachments.len(), 2);
        let missing = first
            .attachments
            .iter()
            .find(|a| a.name.as_deref() == Some("asset-gone"))
            .unwrap();
        assert!(missing.path.is_none());
        assert_eq!(missing.metadata.get("missing"), Some(&Value::Bool(true)));

        let resolved = first
            .attachments
            .iter()
            .find(|a| a.name.as_deref() == Some("asset-attached"))
            .unwrap();
        assert_eq!(resolved.media_type.as_deref(), Some("text/plain"));

        // Orphans carry no conversation but are kept — they are the bulk of a real export.
        let orphan = records
            .iter()
            .find(|r| r.source_key == "asset-orphan")
            .unwrap();
        assert_eq!(orphan.source, "grok-asset");
        assert_eq!(orphan.metadata.get("orphan"), Some(&Value::Bool(true)));
        assert!(orphan.conversation_id.is_none());
        // A referenced asset inherits its response's context, so ColdStore::context can still find
        // the turns that explain what the blob is.
        let attached = records
            .iter()
            .find(|r| r.source_key == "asset-attached")
            .unwrap();
        assert_eq!(attached.metadata.get("orphan"), Some(&Value::Bool(false)));
        assert_eq!(attached.conversation_id.as_deref(), Some("conv-1"));
        assert_eq!(attached.parent_id.as_deref(), Some("resp-1"));
        assert_eq!(attached.timestamp, first.timestamp);
    }

    /// The export writes both `assistant` and `ASSISTANT`, and `"model": ""` on a fifth of
    /// responses. Two spellings of one role split every grouping; an empty string is a value a
    /// filter would have to ask for by name.
    #[test]
    fn grok_export_normalizes_speaker_case_and_blank_fields() {
        let temp = tempfile::tempdir().unwrap();
        export_fixture(temp.path());
        let mut records = Vec::new();
        Ingestor::new(temp.path().join("errors.jsonl"))
            .ingest_path(SourceKind::GrokExport, temp.path(), "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();

        let second = records.iter().find(|r| r.source_key == "resp-2").unwrap();
        assert_eq!(second.speaker.as_deref(), Some("assistant"));
        assert_eq!(second.model, None, "blank model is absent, not empty");
    }

    #[test]
    fn grok_export_is_idempotent_and_autodetected() {
        let temp = tempfile::tempdir().unwrap();
        export_fixture(temp.path());
        let ingest = |kind| {
            let mut records = Vec::new();
            Ingestor::new(temp.path().join("errors.jsonl"))
                .ingest_path(kind, temp.path(), "chat", |record| {
                    records.push(record);
                    Ok(())
                })
                .unwrap();
            records
        };
        // Auto must recognise the export by its fixed filename, not by anything user-chosen.
        let explicit = ingest(SourceKind::GrokExport);
        let auto = ingest(SourceKind::Auto);
        assert_eq!(explicit.len(), auto.len());
        let ids: Vec<_> = explicit.iter().map(|r| r.id).collect();
        let again: Vec<_> = auto.iter().map(|r| r.id).collect();
        assert_eq!(ids, again, "re-import must update in place, not fork");
    }

    #[test]
    fn memory_md_without_front_matter_is_quarantined_not_dropped() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("plain.md");
        fs::write(&path, "just a note, no front matter\n").unwrap();
        let mut records = Vec::new();
        let report = Ingestor::new(temp.path().join("errors.jsonl"))
            .ingest_path(SourceKind::MemoryMd, &path, "chat", |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        assert!(records.is_empty());
        assert_eq!(report.rejected, 1);
    }
}
