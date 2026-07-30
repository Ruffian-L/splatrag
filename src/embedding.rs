use crate::config::{EmbeddingConfig, LabelingConfig};
use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

#[derive(Clone)]
pub struct EmbeddingClient {
    client: Client,
    config: EmbeddingConfig,
}

#[derive(Clone)]
pub struct LabelingClient {
    client: Client,
    config: LabelingConfig,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

impl EmbeddingClient {
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()?;
        Ok(Self { client, config })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed_batch(&[text.to_string()]).await?;
        vectors.pop().context("embedding server returned no vector")
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let bounded: Vec<String> = texts.iter().map(|text| bound_for_embedding(text)).collect();
        let url = format!("{}/v1/embeddings", self.config.url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .json(&EmbeddingRequest {
                model: &self.config.model,
                input: &bounded,
                encoding_format: "float",
            })
            .send()
            .await
            .context("failed to reach local embedding server")?
            .error_for_status()
            .context("local embedding server rejected request")?
            .json::<EmbeddingResponse>()
            .await
            .context("invalid local embedding response")?;

        if response.data.len() != texts.len() {
            anyhow::bail!(
                "embedding server returned {} vectors for {} inputs",
                response.data.len(),
                texts.len()
            );
        }
        let mut ordered: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for mut item in response.data {
            if item.embedding.len() != self.config.dimensions {
                anyhow::bail!(
                    "embedding dimension mismatch: expected {}, got {}",
                    self.config.dimensions,
                    item.embedding.len()
                );
            }
            normalize(&mut item.embedding);
            if item.index >= ordered.len() {
                anyhow::bail!("embedding response index {} is out of range", item.index);
            }
            ordered[item.index] = Some(item.embedding);
        }
        ordered
            .into_iter()
            .enumerate()
            .map(|(index, vector)| {
                vector.with_context(|| format!("embedding response omitted item {index}"))
            })
            .collect()
    }

    pub async fn doctor(&self) -> Result<()> {
        self.embed("SplatRAG embedding health check").await?;
        Ok(())
    }

    pub fn batch_size(&self) -> usize {
        self.config.batch_size.max(1)
    }
}

impl LabelingClient {
    pub fn new(config: LabelingConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()?;
        Ok(Self { client, config })
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub async fn label_basin(&self, representatives: &[String]) -> Result<BasinLabelDraft> {
        if !self.config.enabled {
            anyhow::bail!("basin labeling is disabled");
        }
        // 8 x 400 chars keeps the prompt near ~1k tokens, well inside a 4096-token server context
        // with room for the reply. 12 x 700 could reach ~2.1k and crowd out the response budget.
        let evidence = representatives
            .iter()
            .take(8)
            .enumerate()
            .map(|(index, text)| format!("{}. {}", index + 1, truncate(text, 400)))
            .collect::<Vec<_>>()
            .join("\n");
        // The word cap is load-bearing: small local models write expansive summaries and run past
        // max_tokens mid-object, which reaches the parser as unterminated JSON.
        let prompt = format!(
            "Name this AI-memory basin from its representative messages.\n\
             Return only JSON with string fields label, path, summary.\n\
             label: 2-6 concrete words. path: slash-separated hierarchy. \
             summary: one sentence of at most 25 words, grounded in the messages.\n\n{evidence}"
        );
        let url = format!(
            "{}/v1/chat/completions",
            self.config.url.trim_end_matches('/')
        );
        let response: Value = self
            .client
            .post(url)
            .json(&label_request(&self.config.model, prompt))
            .send()
            .await
            .context("failed to reach local labeling model")?
            .error_for_status()?
            .json()
            .await?;
        let content = response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .context("labeling model response omitted message content")?;
        parse_label_json(content)
    }
}

/// What the vision model is asked of every picture in the archive.
///
/// Transcription first, description only as a fallback: most of these images are screenshots of
/// terminals, code and chat, where the literal text *is* the memory. Asking for a description of a
/// screenshot instead would throw away the thing worth keeping.
const OCR_PROMPT: &str = "Transcribe all text visible in this image verbatim, preserving line \
breaks and structure. If the image contains no readable text, instead describe what it shows in \
one or two sentences. Output only the transcription or description, with no preamble.";

impl LabelingClient {
    /// Read an image with the local vision model.
    ///
    /// Requires `llama-server` to have been started with a matching `--mmproj` projector; without
    /// one the server accepts the request and simply never sees the image.
    pub async fn read_image(&self, bytes: &[u8], media_type: &str) -> Result<String> {
        if !self.config.enabled {
            anyhow::bail!("labeling/vision model is disabled");
        }
        let url = format!(
            "{}/v1/chat/completions",
            self.config.url.trim_end_matches('/')
        );
        let data_url = format!(
            "data:{media_type};base64,{}",
            crate::ingest::extract::base64_encode(bytes)
        );
        let request = serde_json::json!({
            "model": self.config.model,
            "temperature": 0.0,
            // Long enough for a dense screenshot; the cost of truncation is a clipped memory.
            "max_tokens": 900,
            // Same reason as label_request: Gemma 4 thinks by default in llama.cpp and will spend
            // the entire budget on hidden reasoning, returning an empty `content`.
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}]
        });

        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await
            .context("failed to reach local vision server")?
            .error_for_status()
            .context("local vision server rejected request")?
            .json::<Value>()
            .await
            .context("invalid local vision response")?;

        let content = response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_string();
        if content.is_empty() {
            anyhow::bail!(
                "vision model returned empty content (is --mmproj loaded, and thinking disabled?)"
            );
        }
        Ok(content)
    }
}

fn label_request(model: &str, prompt: String) -> Value {
    serde_json::json!({
        "model": model,
        "temperature": 0.1,
        // Headroom for a complete JSON object. Too tight a budget truncates mid-object, which the
        // parser can only report as malformed JSON rather than as the length limit it actually is.
        "max_tokens": 512,
        // Gemma 4 enables thinking by default in llama.cpp. Basin labels are a
        // short structured task, so spending the response budget on hidden
        // reasoning can leave `message.content` empty.
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": prompt}]
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinLabelDraft {
    pub label: String,
    pub path: String,
    pub summary: String,
}

fn parse_label_json(content: &str) -> Result<BasinLabelDraft> {
    let start = content
        .find('{')
        .with_context(|| format!("label response has no JSON object: {:?}", truncate(content, 200)))?;
    // An opening brace with no closing one is the signature of a response cut off at max_tokens,
    // so echo the tail — "malformed JSON" alone sends you looking in the wrong place.
    let end = content.rfind('}').with_context(|| {
        format!(
            "label response was cut off before closing the JSON object (likely max_tokens): {:?}",
            truncate(content, 200)
        )
    })?;
    let draft: BasinLabelDraft = serde_json::from_str(&content[start..=end])?;
    if draft.label.trim().is_empty() || draft.path.trim().is_empty() {
        anyhow::bail!("label response contains empty label or path");
    }
    Ok(draft)
}

/// Characters of a memory that actually reach the embedding server.
///
/// The archive contains pasted logs and code dumps; the largest single memory in it tokenizes to
/// **43,771 tokens**, past the server's whole 40,960-token context. That request comes back as a
/// 400 and takes the entire ingest run down with it, which is how the Grok import silently stopped
/// at 896 of 1819 records.
///
/// Bounding the input is not data loss. The cold log is authoritative and keeps every byte; the
/// embedding is derived state, and a mean-pooled vector over 40k tokens of log spew would not have
/// carried meaning anyway. Roughly 4 chars per token, so this sits near 25k tokens — comfortably
/// inside the context with room for whatever the server adds.
const EMBED_CHAR_BUDGET: usize = 100_000;

/// Clip to [`EMBED_CHAR_BUDGET`] on a char boundary. Cheap no-op for the overwhelming majority.
fn bound_for_embedding(text: &str) -> String {
    if text.len() <= EMBED_CHAR_BUDGET {
        return text.to_string();
    }
    text.chars().take(EMBED_CHAR_BUDGET).collect()
}

pub fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for value in vector {
            *value /= norm;
        }
    }
}

pub fn matryoshka64(vector: &[f32]) -> Result<Vec<f32>> {
    if vector.len() < 64 {
        anyhow::bail!("embedding must have at least 64 dimensions");
    }
    let mut sliced = vector[..64].to_vec();
    normalize(&mut sliced);
    Ok(sliced)
}

fn truncate(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One oversized memory must not be able to take an ingest run down.
    #[test]
    fn oversized_text_is_bounded_but_ordinary_text_is_untouched() {
        let ordinary = "a normal conversational turn about splat memory";
        assert_eq!(bound_for_embedding(ordinary), ordinary);

        // Larger than the real 43,771-token record that aborted the Grok import.
        let huge = "x".repeat(EMBED_CHAR_BUDGET * 3);
        let bounded = bound_for_embedding(&huge);
        assert_eq!(bounded.chars().count(), EMBED_CHAR_BUDGET);

        // Multi-byte input must clip on a char boundary, not mid-codepoint.
        let wide = "日".repeat(EMBED_CHAR_BUDGET + 500);
        let bounded = bound_for_embedding(&wide);
        assert_eq!(bounded.chars().count(), EMBED_CHAR_BUDGET);
    }

    #[test]
    fn matryoshka_vector_is_normalized() {
        let full = vec![2.0; 4096];
        let sliced = matryoshka64(&full).unwrap();
        let norm = sliced.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn label_parser_tolerates_code_fences() {
        let label = parse_label_json(
            "```json\n{\"label\":\"Rust memory\",\"path\":\"ai/rust\",\"summary\":\"Work.\"}\n```",
        )
        .unwrap();
        assert_eq!(label.path, "ai/rust");
    }

    #[test]
    fn label_request_disables_thinking_for_structured_output() {
        let request = label_request("gemma-4", "label this".into());
        assert_eq!(
            request.pointer("/chat_template_kwargs/enable_thinking"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            request.pointer("/messages/0/content"),
            Some(&Value::String("label this".into()))
        );
    }
}
