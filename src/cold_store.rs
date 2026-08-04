use crate::record::{MemoryRecord, RecallContext};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
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
}
