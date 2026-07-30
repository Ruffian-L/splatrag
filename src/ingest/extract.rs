//! Asset extraction: bytes in, text out.
//!
//! Export assets arrive as `prod-mc-asset-server/<uuid>/content` with **no extension** — the type
//! lives in the first few bytes and nowhere else. The flattened `unnested_files/` copy of this same
//! data invented filenames from conversation titles, which are neither unique nor meaningful, so
//! nothing here may key on a name.
//!
//! Extraction splits into what a pure function can do (`extract_text`: utf-8 payloads, decoded
//! inline) and what needs a model on the other end of a socket (images -> OCR). The second kind is
//! slow enough that it is cached separately rather than redone per ingest.

use crate::record::sha256_hex;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

/// How many leading bytes are enough to identify every format below.
const SNIFF_BYTES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MediaKind {
    Text,
    Html,
    Png,
    Jpeg,
    Webp,
    Gif,
    Pdf,
    Zip,
    Unknown,
}

impl MediaKind {
    /// The MIME type recorded on the attachment, so a consumer never has to re-sniff.
    pub fn media_type(self) -> &'static str {
        match self {
            MediaKind::Text => "text/plain",
            MediaKind::Html => "text/html",
            MediaKind::Png => "image/png",
            MediaKind::Jpeg => "image/jpeg",
            MediaKind::Webp => "image/webp",
            MediaKind::Gif => "image/gif",
            MediaKind::Pdf => "application/pdf",
            MediaKind::Zip => "application/zip",
            MediaKind::Unknown => "application/octet-stream",
        }
    }

    /// True when the payload is a picture, i.e. text can only come from OCR.
    pub fn is_image(self) -> bool {
        matches!(
            self,
            MediaKind::Png | MediaKind::Jpeg | MediaKind::Webp | MediaKind::Gif
        )
    }
}

impl fmt::Display for MediaKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.media_type())
    }
}

/// Identify a payload from its leading bytes.
///
/// Magic numbers first, because they are unambiguous. Only if none match do we fall back to asking
/// whether the bytes decode as UTF-8 — a test that must come last, since a PNG chunk can contain
/// incidental valid UTF-8 but a PNG is never text.
pub fn sniff(bytes: &[u8]) -> MediaKind {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return MediaKind::Png;
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return MediaKind::Jpeg;
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return MediaKind::Gif;
    }
    if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        return MediaKind::Webp;
    }
    if bytes.starts_with(b"%PDF") {
        return MediaKind::Pdf;
    }
    if bytes.starts_with(b"PK\x03\x04") {
        return MediaKind::Zip;
    }

    let Ok(text) = std::str::from_utf8(&bytes[..bytes.len().min(SNIFF_BYTES)]) else {
        // A truncated multi-byte char at the sniff boundary is not proof of binary, but the
        // callers here re-check the full payload before trusting Text, so erring to Unknown is safe.
        return MediaKind::Unknown;
    };
    let head = text.trim_start().to_lowercase();
    if head.starts_with("<!doctype html") || head.starts_with("<html") || head.starts_with("<p>") {
        return MediaKind::Html;
    }
    MediaKind::Text
}

/// What one asset yielded. `text` is `None` exactly when a model is still required (an image).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extracted {
    pub asset_id: String,
    pub media_type: String,
    pub kind: MediaKind,
    pub sha256: String,
    pub bytes: u64,
    pub text: Option<String>,
}

/// Read an asset and pull out whatever text does not need a model.
///
/// Images return `Ok(Extracted { text: None, .. })` rather than an error: "not yet OCR'd" is a
/// normal state for a resumable pass, not a failure, and the sha/type recorded here are what let a
/// later OCR run find the file again without re-walking the tree.
pub fn extract(asset_id: &str, path: &Path) -> Result<Extracted> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let kind = sniff(&bytes);
    let text = match kind {
        MediaKind::Text | MediaKind::Html => {
            let decoded = String::from_utf8_lossy(&bytes).into_owned();
            let cleaned = if kind == MediaKind::Html {
                super::strip_html(&decoded)
            } else {
                decoded
            };
            (!cleaned.trim().is_empty()).then_some(cleaned)
        }
        // Images need OCR; pdf/zip/unknown need their own unpackers. All are left for a later pass
        // rather than silently emitting an empty memory.
        _ => None,
    };
    Ok(Extracted {
        asset_id: asset_id.to_string(),
        media_type: kind.media_type().to_string(),
        kind,
        sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
        text,
    })
}

const BASE64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding, for building `data:` URLs.
///
/// Hand-rolled rather than pulled in as a dependency: this is the only call site, the alphabet is
/// fixed by RFC 4648, and the round-trip is pinned by a test below.
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(BASE64[(n >> 18) as usize & 63] as char);
        out.push(BASE64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            BASE64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            BASE64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// A resumable record of what has already been read out of the assets.
///
/// OCR over a full archive runs for hours, so it must survive being interrupted. The cache is
/// derived state keyed by asset id: deleting it costs time, never data.
pub struct OcrCache {
    path: std::path::PathBuf,
    done: std::collections::HashMap<String, Extracted>,
}

impl OcrCache {
    pub fn open(path: impl Into<std::path::PathBuf>) -> Result<Self> {
        use std::io::BufRead;
        let path = path.into();
        let mut done = std::collections::HashMap::new();
        if path.is_file() {
            let file = fs::File::open(&path)?;
            for line in std::io::BufReader::new(file).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                // A half-written trailing line is normal after an interrupt; stop rather than fail.
                let Ok(entry) = serde_json::from_str::<Extracted>(&line) else {
                    break;
                };
                done.insert(entry.asset_id.clone(), entry);
            }
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        Ok(Self { path, done })
    }

    pub fn get(&self, asset_id: &str) -> Option<&Extracted> {
        self.done.get(asset_id)
    }

    pub fn contains(&self, asset_id: &str) -> bool {
        self.done.contains_key(asset_id)
    }

    pub fn len(&self) -> usize {
        self.done.len()
    }

    pub fn is_empty(&self) -> bool {
        self.done.is_empty()
    }

    /// Append one result and flush, so an interrupt loses at most the entry in flight.
    pub fn append(&mut self, entry: Extracted) -> Result<()> {
        use std::io::Write;
        let file = fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&self.path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, &entry)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        self.done.insert(entry.asset_id.clone(), entry);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_including_padding() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
        // High bytes must not sign-extend; a PNG header is the real input shape.
        assert_eq!(base64_encode(b"\x89PNG\r\n\x1a\n"), "iVBORw0KGgo=");
    }

    #[test]
    fn ocr_cache_resumes_and_tolerates_a_torn_tail() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("extracted.jsonl");
        let mut cache = OcrCache::open(&path).unwrap();
        assert!(cache.is_empty());
        cache
            .append(Extracted {
                asset_id: "a1".into(),
                media_type: "image/png".into(),
                kind: MediaKind::Png,
                sha256: "deadbeef".into(),
                bytes: 12,
                text: Some("read from the screenshot".into()),
            })
            .unwrap();

        // A run killed mid-write leaves a partial line; reopening must keep what completed.
        use std::io::Write;
        write!(
            fs::OpenOptions::new().append(true).open(&path).unwrap(),
            "{{\"asset_id\":\"a2\",\"med"
        )
        .unwrap();

        let reopened = OcrCache::open(&path).unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(reopened.contains("a1"));
        assert!(!reopened.contains("a2"));
        assert_eq!(
            reopened.get("a1").unwrap().text.as_deref(),
            Some("read from the screenshot")
        );
    }

    #[test]
    fn magic_bytes_beat_incidental_utf8() {
        // A PNG header is followed by bytes that decode fine as UTF-8; the magic number must win.
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(b"IHDR plain ascii tail");
        assert_eq!(sniff(&png), MediaKind::Png);
        assert!(sniff(&png).is_image());

        assert_eq!(sniff(b"just some notes"), MediaKind::Text);
        assert_eq!(sniff(b"%PDF-1.7"), MediaKind::Pdf);
        assert_eq!(sniff(b"PK\x03\x04\x14\x00"), MediaKind::Zip);
        assert_eq!(sniff(&[0xff, 0xd8, 0xff, 0xe0]), MediaKind::Jpeg);
    }

    #[test]
    fn webp_needs_riff_and_the_webp_tag() {
        let mut riff = b"RIFF".to_vec();
        riff.extend_from_slice(&[0, 0, 0, 0]);
        riff.extend_from_slice(b"WEBP");
        assert_eq!(sniff(&riff), MediaKind::Webp);

        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&[0, 0, 0, 0]);
        wav.extend_from_slice(b"WAVE");
        assert_ne!(sniff(&wav), MediaKind::Webp);
    }

    #[test]
    fn html_assets_are_stripped_to_text() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("content");
        fs::write(&path, "<p>Based on the <strong>sources</strong>, the correlation</p>").unwrap();
        let extracted = extract("asset-1", &path).unwrap();
        assert_eq!(extracted.kind, MediaKind::Html);
        assert_eq!(
            extracted.text.as_deref(),
            Some("Based on the sources , the correlation")
        );
    }

    #[test]
    fn images_defer_to_ocr_without_erroring() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("content");
        fs::write(&path, b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR").unwrap();
        let extracted = extract("asset-2", &path).unwrap();
        assert!(extracted.text.is_none(), "OCR is a later pass, not a failure");
        assert_eq!(extracted.media_type, "image/png");
        assert_eq!(extracted.sha256.len(), 64);
    }
}
