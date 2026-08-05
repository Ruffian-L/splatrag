//! Shell out to `jlens-gguf readout` for verbalizable (text_bridge) words.
//!
//! First-thought mode (default): wraps the query in a Gemma chat hinge so the
//! last residual is the *model's first generation token distribution* — not
//! mid-sentence continuation of raw prose (which surfaces function words).
//!
//! Env:
//! - `JLENS_BIN` — path to jlens-gguf (default: ~/jlens-gguf-repo/target/release/jlens-gguf)
//! - `JLENS_MODEL` — generative GGUF (default: gemma-3-4b-it-q4_0)
//! - `JLENS_TOKENIZER` — optional tokenizer.json
//! - `JLENS_BASELINE` — residual baseline for dim_signature (text_bridge stays raw unembed)
//! - `JLENS_TOP_K` — default 8
//! - `JLENS_LAYERS` — default `last`
//! - `JLENS_RAW=1` — skip chat wrap; unembed raw prompt end (continuation lens)
//! - `JLENS_FORCE_CUDA=1` — only when CUDA jlens build is known good

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JlensToken {
    pub id: u32,
    pub text: String,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JlensReadout {
    pub ok: bool,
    pub prompt: String,
    pub text_bridge: String,
    pub text_bridge_hash: Option<u64>,
    pub top_tokens: Vec<JlensToken>,
    pub layer: Option<usize>,
    pub position: Option<usize>,
    pub model_path: String,
    pub note: String,
    pub error: Option<String>,
    pub raw_stderr_tail: Option<String>,
}

fn default_bin() -> PathBuf {
    if let Ok(p) = std::env::var("JLENS_BIN") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    // Common local build layouts; override with JLENS_BIN.
    for rel in [
        "jlens-gguf-repo/target/release/jlens-gguf",
        "jlens-gguf/target/release/jlens-gguf",
    ] {
        let p = PathBuf::from(&home).join(rel);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from(home).join("jlens-gguf-repo/target/release/jlens-gguf")
}

fn default_model() -> PathBuf {
    if let Ok(p) = std::env::var("JLENS_MODEL") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    for rel in [
        "models/gemma-3-4b-it-q4_0.gguf",
        ".cache/models/gemma-3-4b-it-q4_0.gguf",
    ] {
        let p = PathBuf::from(&home).join(rel);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from(home).join("models/gemma-3-4b-it-q4_0.gguf")
}

fn default_tokenizer() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("JLENS_TOKENIZER") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    for rel in [
        "models/tokenizer.json",
        ".cache/models/tokenizer.json",
    ] {
        let p = PathBuf::from(&home).join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Position-matched residual baseline. Without it, top tokens are model furniture.
fn default_baseline() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("JLENS_BASELINE") {
        let pb = PathBuf::from(p);
        return if pb.exists() { Some(pb) } else { None };
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    for rel in [
        "jlens-gguf-repo/baselines/gemma3-4b-commit-last.baseline",
        ".cache/jlens/gemma3-4b-commit-last.baseline",
    ] {
        let p = PathBuf::from(&home).join(rel);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Wrap user text as a Gemma-3 IT first-turn hinge so residual@-1 is first thought.
fn first_thought_prompt(user_text: &str) -> String {
    let body = user_text.trim();
    // Already a chat transcript — pass through.
    if body.contains("<start_of_turn>") {
        return body.to_string();
    }
    format!(
        "<start_of_turn>user\n{body}\n<end_of_turn>\n<start_of_turn>model\n"
    )
}

/// Run jlens-gguf readout once; parse last JSONL record from stdout (or emit file).
pub async fn readout_verbalizable(prompt: &str, top_k: usize, verify: bool) -> Result<JlensReadout> {
    let bin = default_bin();
    let model = default_model();
    if !bin.exists() {
        bail!(
            "jlens binary missing at {} — build with: cargo build --release -p jlens-gguf --no-default-features",
            bin.display()
        );
    }
    if !model.exists() {
        bail!("jlens model missing at {}", model.display());
    }

    let layers = std::env::var("JLENS_LAYERS").unwrap_or_else(|_| "last".into());
    let top_k = top_k.max(1).min(32);
    let raw_mode = std::env::var("JLENS_RAW").ok().as_deref() == Some("1");
    let lens_prompt = if raw_mode {
        prompt.to_string()
    } else {
        first_thought_prompt(prompt)
    };
    let tmp = std::env::temp_dir().join(format!(
        "jlens_viz_{}.jsonl",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    ));

    let mut cmd = Command::new(&bin);
    cmd.arg("readout")
        .arg("--model")
        .arg(&model)
        .arg("--prompt")
        .arg(&lens_prompt)
        .arg("--layers")
        .arg(&layers)
        .arg("--positions")
        .arg("-1")
        .arg("--top-k")
        .arg(top_k.to_string())
        .arg("--emit")
        .arg(&tmp)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    if let Some(tok) = default_tokenizer() {
        cmd.arg("--tokenizer").arg(tok);
    }
    if let Some(bl) = default_baseline() {
        cmd.arg("--baseline").arg(bl);
    }
    // Default CPU: this machine's nvcc cannot target the live GPU arch for candle-kernels.
    // Set JLENS_FORCE_CUDA=1 only when a CUDA build of jlens-gguf is known good.
    if std::env::var("JLENS_FORCE_CUDA").is_err() {
        cmd.arg("--cpu");
    }
    if verify {
        cmd.arg("--verify");
    }

    let output = cmd
        .output()
        .await
        .with_context(|| format!("spawn {}", bin.display()))?;

    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stderr_tail: String = stderr
        .lines()
        .rev()
        .take(12)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("\n");

    if !output.status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Ok(JlensReadout {
            ok: false,
            prompt: prompt.into(),
            text_bridge: String::new(),
            text_bridge_hash: None,
            top_tokens: vec![],
            layer: None,
            position: None,
            model_path: model.display().to_string(),
            note: "jlens readout failed".into(),
            error: Some(format!(
                "exit {:?} — {}",
                output.status.code(),
                stderr_tail
            )),
            raw_stderr_tail: Some(stderr_tail),
        });
    }

    let body = if tmp.exists() {
        std::fs::read_to_string(&tmp).unwrap_or_default()
    } else {
        String::from_utf8_lossy(&output.stdout).to_string()
    };
    let _ = std::fs::remove_file(&tmp);

    let last = body
        .lines()
        .rev()
        .find(|l| l.trim().starts_with('{'))
        .unwrap_or("");
    if last.is_empty() {
        return Ok(JlensReadout {
            ok: false,
            prompt: prompt.into(),
            text_bridge: String::new(),
            text_bridge_hash: None,
            top_tokens: vec![],
            layer: None,
            position: None,
            model_path: model.display().to_string(),
            note: "no jsonl record from jlens".into(),
            error: Some(stderr_tail.clone()),
            raw_stderr_tail: Some(stderr_tail),
        });
    }

    let v: serde_json::Value =
        serde_json::from_str(last).with_context(|| format!("parse jlens json: {last}"))?;

    let top_tokens: Vec<JlensToken> = v
        .get("top_tokens")
        .and_then(|t| t.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    Some(JlensToken {
                        id: t.get("id")?.as_u64()? as u32,
                        text: t.get("text")?.as_str()?.to_string(),
                        logit: t.get("logit")?.as_f64()? as f32,
                    })
                })
                .collect::<Vec<JlensToken>>()
        })
        .unwrap_or_default();

    let text_bridge = v
        .get("text_bridge")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let text_bridge = if text_bridge.is_empty() {
        top_tokens
            .iter()
            .map(|t| t.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        text_bridge
    };

    Ok(JlensReadout {
        ok: true,
        prompt: prompt.into(),
        text_bridge,
        text_bridge_hash: v.get("text_bridge_hash").and_then(|x| x.as_u64()),
        top_tokens,
        layer: v.get("layer").and_then(|x| x.as_u64()).map(|u| u as usize),
        position: v
            .get("position")
            .and_then(|x| x.as_u64())
            .map(|u| u as usize),
        model_path: model.display().to_string(),
        note: if raw_mode {
            "jlens raw continuation lens (JLENS_RAW=1): residual@prompt-end → unembed".into()
        } else {
            "jlens first thought: Gemma chat hinge → residual@model-start → unembed top tokens"
                .into()
        },
        error: None,
        raw_stderr_tail: if stderr_tail.is_empty() {
            None
        } else {
            Some(stderr_tail)
        },
    })
}

#[allow(dead_code)]
pub fn bin_exists() -> bool {
    Path::new(&default_bin()).exists()
}
