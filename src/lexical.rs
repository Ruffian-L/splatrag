use crate::record::MemoryRecord;
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT};
use tantivy::{Document, Index, IndexReader, ReloadPolicy, TantivyDocument, doc};
use uuid::Uuid;

pub struct LexicalIndex {
    index: Index,
    reader: IndexReader,
    id: Field,
    text: Field,
    path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LexicalHit {
    pub id: Uuid,
    pub score: f32,
}

impl LexicalIndex {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        fs::create_dir_all(&path)?;
        let (index, id, text) = if path.join("meta.json").exists() {
            let index = Index::open_in_dir(&path)?;
            let schema = index.schema();
            let id = schema.get_field("id")?;
            let text = schema.get_field("text")?;
            (index, id, text)
        } else {
            let mut builder = Schema::builder();
            let id = builder.add_text_field("id", STRING | STORED);
            let text = builder.add_text_field("text", TEXT | STORED);
            builder.add_text_field("domain", STRING | STORED);
            builder.add_text_field("model", STRING | STORED);
            builder.add_text_field("conversation_id", STRING | STORED);
            let schema = builder.build();
            let index = Index::create_in_dir(&path, schema)?;
            (index, id, text)
        };
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;
        Ok(Self {
            index,
            reader,
            id,
            text,
            path,
        })
    }

    pub fn add_records(&self, records: &[MemoryRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let schema = self.index.schema();
        let domain = schema.get_field("domain")?;
        let model = schema.get_field("model")?;
        let conversation_id = schema.get_field("conversation_id")?;
        let mut writer = self.index.writer(64_000_000)?;
        for record in records {
            writer.add_document(doc!(
                self.id => record.id.to_string(),
                self.text => record.text.clone(),
                domain => record.domain.clone(),
                model => record.model.clone().unwrap_or_default(),
                conversation_id => record.conversation_id.clone().unwrap_or_default(),
            ))?;
        }
        writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<LexicalHit>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.text]);
        let parsed = parser
            .parse_query(query)
            .with_context(|| format!("invalid lexical query {query:?}"))?;
        let results = searcher.search(&parsed, &TopDocs::with_limit(limit).order_by_score())?;
        let schema = self.index.schema();
        let mut hits = Vec::with_capacity(results.len());
        for (score, address) in results {
            let document: TantivyDocument = searcher.doc(address)?;
            let value: serde_json::Value = serde_json::from_str(&document.to_json(&schema))?;
            let Some(id) = value
                .get("id")
                .and_then(|value| value.get(0))
                .and_then(serde_json::Value::as_str)
                .and_then(|value| Uuid::parse_str(value).ok())
            else {
                continue;
            };
            hits.push(LexicalHit { id, score });
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        Ok(hits)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_returns_keyed_memory() {
        let temp = tempfile::tempdir().unwrap();
        let index = LexicalIndex::open(temp.path().join("tantivy")).unwrap();
        let record = MemoryRecord::new("test", "one", "anisotropic gaussian memory");
        index.add_records(std::slice::from_ref(&record)).unwrap();
        let hits = index.search("anisotropic", 10).unwrap();
        assert_eq!(hits[0].id, record.id);
    }
}
