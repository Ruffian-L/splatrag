use crate::record::MemoryRecord;
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions,
};
use tantivy::tokenizer::{
    Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, StopWordFilter, TextAnalyzer,
};
use tantivy::{Document, Index, IndexReader, ReloadPolicy, TantivyDocument, Term, doc};
use uuid::Uuid;

/// Tantivy's default tokenizer only lowercases and splits. That leaves `blades` unable to match
/// `blade`, and indexes `the`/`is`/`no` as ordinary terms — so BM25 ends up ranking on stopword
/// coincidence and document length instead of on the rare word that carries the query's intent.
/// Stemming plus English stopword removal is what makes the lexical arm discriminate.
const TEXT_TOKENIZER: &str = "memory_en";

/// Tantivy's English stopword list is tuned for keyword search and keeps every interrogative —
/// `what`, `how`, `where` all survive it. This store is queried in natural questions, so those
/// words appear in most queries *and* most documents while carrying no topical signal. Left in,
/// they score: a document matching only `what` ties a document matching the one rare term the
/// question was actually about.
const QUESTION_WORDS: &[&str] = &[
    "what", "when", "where", "who", "whom", "whose", "which", "why", "how", "did", "do", "does",
    "done", "can", "could", "would", "should", "shall", "tell", "know",
];

fn register_tokenizer(index: &Index) -> Result<()> {
    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(64))
        .filter(LowerCaser)
        .filter(
            StopWordFilter::new(Language::English)
                .context("tantivy has no English stopword list")?,
        )
        .filter(StopWordFilter::remove(
            QUESTION_WORDS.iter().map(|word| word.to_string()),
        ))
        .filter(Stemmer::new(Language::English))
        .build();
    index.tokenizers().register(TEXT_TOKENIZER, analyzer);
    Ok(())
}

fn stemmed_text_options() -> TextOptions {
    TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TEXT_TOKENIZER)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored()
}

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
            let text = builder.add_text_field("text", stemmed_text_options());
            builder.add_text_field("domain", STRING | STORED);
            builder.add_text_field("model", STRING | STORED);
            builder.add_text_field("conversation_id", STRING | STORED);
            let schema = builder.build();
            let index = Index::create_in_dir(&path, schema)?;
            (index, id, text)
        };
        // Must happen on both paths: an index reopened from disk records the tokenizer by name, and
        // querying without registering it back returns an "unknown tokenizer" error at search time.
        register_tokenizer(&index)?;
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
        // Built from analyzed terms rather than QueryParser. Recall receives arbitrary prose —
        // "alpha (+)- thalassemia", "RIGHT NOW: [REQUEST: LOCK]", "grok-4-1-non-thinking" — and
        // QueryParser reads +, -, (), :, ", ^, ~ as operators, so real text raises a syntax error
        // that fails the entire recall including the semantic arm. Going through the index
        // analyzer removes that whole class of failure and guarantees query-time and index-time
        // stemming cannot drift apart.
        let terms = self.analyzed_terms(query)?;
        // A query of only stopwords ("what is that?") analyzes to nothing. That is not an error:
        // the lexical arm simply abstains and the semantic arm answers.
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let searcher = self.reader.searcher();
        let clauses = terms
            .into_iter()
            .map(|text| {
                let term = Term::from_field_text(self.text, &text);
                let query: Box<dyn Query> =
                    Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs));
                (Occur::Should, query)
            })
            .collect::<Vec<_>>();
        let parsed = BooleanQuery::new(clauses);
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

    /// The terms `query` reduces to under the same analyzer the index uses — i.e. what survives
    /// stopword removal and stemming. Duplicates are kept so a repeated word carries the extra
    /// weight it would under a parsed query.
    fn analyzed_terms(&self, query: &str) -> Result<Vec<String>> {
        let mut analyzer = self
            .index
            .tokenizers()
            .get(TEXT_TOKENIZER)
            .context("memory tokenizer was not registered on this index")?;
        let mut stream = analyzer.token_stream(query);
        let mut terms = Vec::new();
        while stream.advance() {
            terms.push(stream.token().text.clone());
        }
        Ok(terms)
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

    /// The real miss this tokenizer was added to fix. Two failures stacked: `blades` could not
    /// match `blade` without a stemmer, and the decoy — which deliberately shares the query's
    /// `what` and nothing else — matched that interrogative with the same IDF as the one rare
    /// term that carries the question, producing a tie the wrong document could win.
    #[test]
    fn plural_query_matches_singular_body_via_stemming() {
        let temp = tempfile::tempdir().unwrap();
        let index = LexicalIndex::open(temp.path().join("tantivy")).unwrap();
        let target = MemoryRecord::new("test", "blades", "a kind sentence with a blade on the end");
        let decoy = MemoryRecord::new("test", "decoy", "what is the guard on the backup drive");
        index.add_records(&[target.clone(), decoy]).unwrap();

        let hits = index.search("what is the no blades rule", 10).unwrap();
        assert_eq!(hits[0].id, target.id, "stemmed rare term must beat stopwords");
    }

    #[test]
    fn stopwords_alone_retrieve_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let index = LexicalIndex::open(temp.path().join("tantivy")).unwrap();
        let record = MemoryRecord::new("test", "one", "a kind sentence with a blade on the end");
        index.add_records(std::slice::from_ref(&record)).unwrap();
        assert!(
            index.search("what is the", 10).unwrap().is_empty(),
            "an all-stopword query carries no signal and must not rank documents"
        );
    }

    /// Real prose reaches recall verbatim. Every one of these crashed the previous QueryParser
    /// path with a syntax error that failed the whole recall, semantic arm included.
    #[test]
    fn query_syntax_characters_do_not_fail_recall() {
        let temp = tempfile::tempdir().unwrap();
        let index = LexicalIndex::open(temp.path().join("tantivy")).unwrap();
        let record = MemoryRecord::new(
            "test",
            "one",
            "homozygous alpha thalassemia trait raises anemia vulnerability",
        );
        index.add_records(std::slice::from_ref(&record)).unwrap();

        for query in [
            "A high microerythrocyte count raises vulnerability to severe anemia in \
             homozygous alpha (+)- thalassemia trait subjects.",
            "RIGHT NOW: [REQUEST: LOCK]",
            "grok-4-1-non-thinking",
            "sigma = s^2 I + (lambda - s^2) uu^T",
            "\"unbalanced quote",
            "trailing operator +",
        ] {
            index
                .search(query, 10)
                .unwrap_or_else(|error| panic!("query {query:?} failed: {error:#}"));
        }

        let hits = index.search("alpha (+)- thalassemia", 10).unwrap();
        assert_eq!(hits[0].id, record.id, "punctuation must not suppress the match");
    }

    /// Reopening must re-register the analyzer or every subsequent search fails.
    #[test]
    fn reopened_index_can_still_search() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("tantivy");
        let record = MemoryRecord::new("test", "one", "persistent homology barcodes");
        LexicalIndex::open(&dir)
            .unwrap()
            .add_records(std::slice::from_ref(&record))
            .unwrap();

        let hits = LexicalIndex::open(&dir).unwrap().search("barcode", 10).unwrap();
        assert_eq!(hits[0].id, record.id);
    }
}
