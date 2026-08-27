use crate::record::{MemoryRecord, RecallContext};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ColdStore {
    path: PathBuf,
}

impl ColdStore {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        if !path.exists() {
            File::create(&path)
                .with_context(|| format!("failed to create cold store {}", path.display()))?;
        }
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn load_all(&self) -> Result<Vec<MemoryRecord>> {
        let file = File::open(&self.path)?;
        let mut records = Vec::new();
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line.with_context(|| format!("failed reading cold line {}", index + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            let record = serde_json::from_str(&line)
                .with_context(|| format!("invalid cold record on line {}", index + 1))?;
            records.push(record);
        }
        Ok(records)
    }

    pub fn ids(&self) -> Result<HashSet<Uuid>> {
        Ok(self
            .load_all()?
            .into_iter()
            .map(|record| record.id)
            .collect())
    }

    pub fn append_new(&self, records: &[MemoryRecord]) -> Result<usize> {
        if records.is_empty() {
            return Ok(0);
        }
        let mut known = self.ids()?;
        let file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)?;
        let mut writer = BufWriter::new(file);
        let mut appended = 0;
        for record in records {
            if !known.insert(record.id) {
                continue;
            }
            serde_json::to_writer(&mut writer, record)?;
            writer.write_all(b"\n")?;
            appended += 1;
        }
        writer.flush()?;
        writer.get_ref().sync_data()?;
        Ok(appended)
    }

    pub fn byte_len(&self) -> Result<u64> {
        Ok(fs::metadata(&self.path)
            .with_context(|| format!("stat {}", self.path.display()))?
            .len())
    }

    /// Read complete JSONL lines appended after `byte_offset`.
    ///
    /// Incomplete last lines (a writer still flushing) are held back so the
    /// next call retries them. If the file shrank, `rewritten` is true and
    /// `records` is a full reload.
    pub fn read_new_since(&self, byte_offset: u64) -> Result<(u64, Vec<MemoryRecord>, bool)> {
        let len = self.byte_len()?;
        if byte_offset > len {
            return Ok((len, self.load_all()?, true));
        }
        if byte_offset == len {
            return Ok((len, Vec::new(), false));
        }
        let mut file = File::open(&self.path)
            .with_context(|| format!("failed to open {}", self.path.display()))?;
        file.seek(SeekFrom::Start(byte_offset))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let complete_end = if buf.ends_with(&[b'\n']) {
            buf.len()
        } else {
            buf.iter()
                .rposition(|&b| b == b'\n')
                .map(|i| i + 1)
                .unwrap_or(0)
        };
        if complete_end == 0 {
            return Ok((byte_offset, Vec::new(), false));
        }
        let consumed = byte_offset + complete_end as u64;
        let mut records = Vec::new();
        for line in buf[..complete_end].split(|&b| b == b'\n') {
            if line.is_empty() {
                continue;
            }
            let line = std::str::from_utf8(line)
                .with_context(|| format!("cold store {} is not utf-8", self.path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            records.push(serde_json::from_str(line).with_context(|| {
                format!("invalid cold record while tailing {}", self.path.display())
            })?);
        }
        Ok((consumed, records, false))
    }

    pub fn record_map(&self) -> Result<HashMap<Uuid, MemoryRecord>> {
        Ok(self
            .load_all()?
            .into_iter()
            .map(|record| (record.id, record))
            .collect())
    }

    pub fn context(&self, target: &MemoryRecord, radius: usize) -> Result<RecallContext> {
        let Some(conversation_id) = &target.conversation_id else {
            return Ok(RecallContext::default());
        };
        let mut conversation: Vec<_> = self
            .load_all()?
            .into_iter()
            .filter(|record| record.conversation_id.as_ref() == Some(conversation_id))
            .collect();
        conversation.sort_by(|a, b| {
            a.turn_index
                .cmp(&b.turn_index)
                .then_with(|| a.timestamp.cmp(&b.timestamp))
                .then_with(|| a.id.cmp(&b.id))
        });
        let Some(position) = conversation
            .iter()
            .position(|record| record.id == target.id)
        else {
            return Ok(RecallContext::default());
        };
        let start = position.saturating_sub(radius);
        let end = (position + radius + 1).min(conversation.len());
        Ok(RecallContext {
            before: conversation[start..position].to_vec(),
            after: conversation[position + 1..end].to_vec(),
        })
    }

    pub fn verify_append_only(before: &[MemoryRecord], after: &[MemoryRecord]) -> bool {
        after.len() >= before.len()
            && before
                .iter()
                .zip(after.iter())
                .all(|(old, new)| old.id == new.id && old.content_sha256 == new.content_sha256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_is_idempotent_and_order_preserving() {
        let temp = tempfile::tempdir().unwrap();
        let store = ColdStore::open(temp.path().join("memories.jsonl")).unwrap();
        let first = MemoryRecord::new("test", "1", "one");
        let second = MemoryRecord::new("test", "2", "two");
        assert_eq!(
            store.append_new(&[first.clone(), second.clone()]).unwrap(),
            2
        );
        assert_eq!(store.append_new(std::slice::from_ref(&first)).unwrap(), 0);
        let loaded = store.load_all().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, first.id);
        assert!(ColdStore::verify_append_only(&[first], &loaded));
    }

    #[test]
    fn tail_read_sees_rows_appended_after_open() {
        let temp = tempfile::tempdir().unwrap();
        let store = ColdStore::open(temp.path().join("memories.jsonl")).unwrap();
        let first = MemoryRecord::new("cli-session", "1", "yesetday");
        assert_eq!(store.append_new(std::slice::from_ref(&first)).unwrap(), 1);
        let offset = store.byte_len().unwrap();
        let (end, none, rewritten) = store.read_new_since(offset).unwrap();
        assert_eq!(end, offset);
        assert!(none.is_empty());
        assert!(!rewritten);

        let second = MemoryRecord::new("cli-session", "2", "ogarinzing");
        assert_eq!(store.append_new(std::slice::from_ref(&second)).unwrap(), 1);
        let (end, new, rewritten) = store.read_new_since(offset).unwrap();
        assert!(!rewritten);
        assert_eq!(end, store.byte_len().unwrap());
        assert_eq!(new.len(), 1);
        assert_eq!(new[0].id, second.id);
        assert!(new[0].text.contains("ogarinzing"));
    }
}
