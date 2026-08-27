//! 64D memory packets — wire format for the steering lane.
//!
//! Stack (already proven in niodv4 juice; this is the SplatRAG joint only):
//!   semantics [64]  →  optional VQ →  Unicode PUA (U+E000+)  →  string
//!
//! Raw float pack is lossless. Unicode pack is lossy VQ (exact re-encode match
//! after quantize, same as niodv4 M7.5 contract).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use uuid::Uuid;

pub const PACKET_DIM: usize = 64;
/// BMP Private Use Area base — same as niodv4 `unicode_tokenizer.PUA_BASE`.
pub const PUA_BASE: u32 = 0xE000;
pub const PUA_CAPACITY: usize = 0xF8FF - 0xE000 + 1; // 6400

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPacket {
    pub version: u32,
    pub dim: usize,
    pub memory_id: Option<Uuid>,
    /// Unit (or near-unit) 64D semantics.
    pub semantics_64: Vec<f32>,
    /// Little-endian f32 bytes, standard base64 — handy for connectors that hate long JSON arrays.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantics_b64: Option<String>,
    /// Steering α, 0.0 = unsteered. Absent in a hand-written packet means "don't steer",
    /// which is why the default is zero and not mass's 1.0.
    #[serde(default)]
    pub gain: f32,
    #[serde(default = "default_mass")]
    pub mass: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basin_id: Option<String>,
    #[serde(default)]
    pub basin_locked: bool,
    /// Optional VQ Unicode transport (one char per subsampled step; single vector → one char).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unicode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unicode_codebook: Option<String>,
    /// Cosine(raw, VQ centroid) when unicode was filled; absent on raw-only packs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant_cosine: Option<f32>,
}

fn default_mass() -> f32 {
    1.0
}

impl MemoryPacket {
    pub fn from_semantics(
        memory_id: Option<Uuid>,
        semantics: &[f32],
        gain: f32,
        mass: f32,
        basin_id: Option<String>,
        basin_locked: bool,
    ) -> Result<Self> {
        ensure_dim64(semantics)?;
        Ok(Self {
            version: 1,
            dim: PACKET_DIM,
            memory_id,
            semantics_64: semantics.to_vec(),
            semantics_b64: Some(encode_f32_le_b64(semantics)),
            gain,
            mass,
            basin_id,
            basin_locked,
            unicode: None,
            unicode_codebook: None,
            quant_cosine: None,
        })
    }

    /// Attach VQ unicode using a loaded codebook. Replaces `unicode` fields; keeps raw floats.
    pub fn with_unicode(mut self, codebook: &VqCodebook) -> Result<Self> {
        let (ch, cos) = codebook.encode_one(&self.semantics_64)?;
        self.unicode = Some(ch.to_string());
        self.unicode_codebook = Some(codebook.label.clone());
        self.quant_cosine = Some(cos);
        Ok(self)
    }

    /// Prefer raw `semantics_64`; if empty, try b64; if still empty and unicode+codebook, decode VQ.
    pub fn resolve_semantics(&self, codebook: Option<&VqCodebook>) -> Result<Vec<f32>> {
        if self.semantics_64.len() == PACKET_DIM {
            return Ok(self.semantics_64.clone());
        }
        if let Some(b64) = &self.semantics_b64 {
            let v = decode_f32_le_b64(b64)?;
            ensure_dim64(&v)?;
            return Ok(v);
        }
        if let (Some(uni), Some(cb)) = (&self.unicode, codebook) {
            return cb.decode_string(uni);
        }
        bail!("packet has no recoverable 64D semantics (need array, b64, or unicode+codebook)");
    }

    pub fn validate_raw(&self) -> Result<()> {
        if self.dim != PACKET_DIM && self.semantics_64.len() != PACKET_DIM {
            // allow dim field to match semantics
        }
        if !self.semantics_64.is_empty() {
            ensure_dim64(&self.semantics_64)?;
        }
        Ok(())
    }
}

/// VQ codebook: K centroids in 64D → PUA chars U+E000+i.
#[derive(Debug, Clone)]
pub struct VqCodebook {
    pub label: String,
    pub centroids: Vec<Vec<f32>>,
}

impl VqCodebook {
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read codebook {}", path.display()))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("invalid codebook json {}", path.display()))?;
        let k = value.get("K").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let d = value.get("D").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
        if d != PACKET_DIM {
            bail!("codebook D={d}, expected {PACKET_DIM}");
        }
        let cents = value
            .get("centroids")
            .and_then(|v| v.as_array())
            .context("codebook missing centroids array")?;
        let mut centroids = Vec::with_capacity(cents.len());
        for (i, c) in cents.iter().enumerate() {
            let arr = c
                .as_array()
                .with_context(|| format!("centroid {i} not an array"))?;
            let mut v = Vec::with_capacity(PACKET_DIM);
            for (j, x) in arr.iter().take(PACKET_DIM).enumerate() {
                let f = x
                    .as_f64()
                    .with_context(|| format!("centroid {i}[{j}] not a number"))?
                    as f32;
                v.push(f);
            }
            if v.len() != PACKET_DIM {
                bail!("centroid {i} has dim {}, need {PACKET_DIM}", v.len());
            }
            centroids.push(v);
        }
        if centroids.is_empty() {
            bail!("codebook has zero centroids");
        }
        if k > 0 && centroids.len() != k {
            // tolerate mismatch; use actual length
        }
        if centroids.len() > PUA_CAPACITY {
            bail!(
                "codebook size {} exceeds PUA capacity {PUA_CAPACITY}",
                centroids.len()
            );
        }
        let label = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("codebook")
            .to_string();
        Ok(Self { label, centroids })
    }

    /// Synthetic codebook for tests (no file).
    pub fn synthetic(k: usize, seed: u32) -> Self {
        let mut centroids = Vec::with_capacity(k);
        for i in 0..k {
            let mut v = vec![0.0f32; PACKET_DIM];
            // Distinct unit-ish axes + mix so nearest-neighbour is stable.
            let base = (i as f32 + 1.0) * 0.17 + seed as f32 * 0.01;
            for (j, slot) in v.iter_mut().enumerate() {
                *slot = ((j as f32 + 1.0) * base).sin() * 0.5
                    + ((j as f32 + 3.0) * base * 0.3).cos() * 0.5;
            }
            let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
            for slot in &mut v {
                *slot /= n;
            }
            centroids.push(v);
        }
        Self {
            label: format!("synthetic_k{k}"),
            centroids,
        }
    }

    pub fn encode_one(&self, semantics: &[f32]) -> Result<(char, f32)> {
        ensure_dim64(semantics)?;
        let mut best_i = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let d = l2_sq(semantics, c);
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        let cos = cosine(semantics, &self.centroids[best_i]);
        Ok((index_to_char(best_i)?, cos))
    }

    pub fn decode_char(&self, c: char) -> Result<Vec<f32>> {
        let idx = char_to_index(c)?;
        if idx >= self.centroids.len() {
            bail!(
                "PUA index {idx} out of codebook range {}",
                self.centroids.len()
            );
        }
        Ok(self.centroids[idx].clone())
    }

    pub fn decode_string(&self, s: &str) -> Result<Vec<f32>> {
        let mut chars = s.chars();
        let Some(c) = chars.next() else {
            bail!("empty unicode string");
        };
        // Single-vector packets are one char; multi-char trajectories average (rare for G1).
        let mut acc = self.decode_char(c)?;
        let mut n = 1usize;
        for c in chars {
            let v = self.decode_char(c)?;
            for (a, b) in acc.iter_mut().zip(&v) {
                *a += *b;
            }
            n += 1;
        }
        if n > 1 {
            let inv = 1.0 / n as f32;
            for a in &mut acc {
                *a *= inv;
            }
        }
        Ok(acc)
    }

    /// Quantize then re-encode; must match for the M7.5-style contract on one vector.
    pub fn reencode_match(&self, semantics: &[f32]) -> Result<bool> {
        let (ch, _) = self.encode_one(semantics)?;
        let decoded = self.decode_char(ch)?;
        let (ch2, _) = self.encode_one(&decoded)?;
        Ok(ch == ch2)
    }
}

pub fn index_to_char(idx: usize) -> Result<char> {
    if idx >= PUA_CAPACITY {
        bail!("index {idx} exceeds PUA capacity");
    }
    char::from_u32(PUA_BASE + idx as u32).context("invalid codepoint")
}

pub fn char_to_index(c: char) -> Result<usize> {
    let cp = c as u32;
    if !(PUA_BASE..=0xF8FF).contains(&cp) {
        bail!("char U+{cp:04X} not in BMP PUA");
    }
    Ok((cp - PUA_BASE) as usize)
}

pub fn ensure_dim64(v: &[f32]) -> Result<()> {
    if v.len() != PACKET_DIM {
        bail!("expected {PACKET_DIM}-d vector, got {}", v.len());
    }
    Ok(())
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let d = na * nb;
    if d < 1e-12 {
        0.0
    } else {
        (dot / d).clamp(-1.0, 1.0)
    }
}

/// Centroid-removal strength [`combine`] uses when the caller does not name one.
///
/// 0.2 is the measured setting, not a round number — see the table on [`combine`].
pub const DEFAULT_COMBINE_T: f32 = 0.2;

/// The mean direction of the field, as a unit axis, and how many splats it was measured on.
///
/// Stride-sample the store the way [`crate::pick::NullModel`] is sampled, and for the same reason:
/// the answer must not drift with how much of the store happened to be scanned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldCentroid {
    pub axis: Vec<f32>,
    pub samples: usize,
}

impl FieldCentroid {
    /// `None` when nothing usable was sampled, or when the sample has no net direction at all —
    /// which would be a store with no centre, not a store whose centre is zero.
    pub fn estimate(vectors: &[&[f32]]) -> Option<Self> {
        let mut sum = vec![0.0f32; PACKET_DIM];
        let mut used = 0usize;
        for v in vectors {
            if v.len() != PACKET_DIM {
                continue;
            }
            for (slot, x) in sum.iter_mut().zip(v.iter()) {
                *slot += *x;
            }
            used += 1;
        }
        if used == 0 {
            return None;
        }
        normalize(&mut sum)?;
        Some(Self {
            axis: sum,
            samples: used,
        })
    }
}

/// In-place L2 normalize. `None` when there is no direction left to keep.
fn normalize(v: &mut [f32]) -> Option<()> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm < 1e-9 {
        return None;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Some(())
}

/// Combine 64D states without landing on the generic centre of the store.
///
/// The obvious operator, `normalize(a + b)`, is on-manifold and wrong. Measured 2026-08-22 on 250
/// packets sampled from the 48,068-memory personal store: naive sums sit at median
/// nearest-neighbour cosine 0.7860 against a 0.2620 random floor, so they *look* like real
/// memories — but they carry a top-5 hub share of 23.2% where real memories carry 13.2%, and a
/// cosine-to-centroid of 0.8011 where real memories sit at 0.6883. Addition drags toward the mean
/// direction of the field, so a naive combine samples the most generic region there is. A
/// behavioural run on that would read "generic" as "emergent".
///
/// Subtracting `t` of the component along the field centroid buys that back:
///
/// | t             | NN cosine | top-5 hub share | cos → centroid |
/// |---------------|-----------|-----------------|----------------|
/// | 0.0 (naive)   | 0.7860    | 23.2%           | 0.8011         |
/// | 0.2           | 0.7500    | 16.0%           | 0.7308         |
/// | 0.4           | 0.6981    | 10.2%           | 0.6262         |
/// | 1.0           | 0.3498    | 13.2%           | 0.0000         |
/// | real memories | 0.7487    | 13.2%           | 0.6883         |
///
/// [`DEFAULT_COMBINE_T`] = 0.2 is the setting that matches real nearest-neighbour density (0.7500
/// against 0.7487) while cutting hub share by a third. t = 0.4 goes *below* the real hub share,
/// which is what to reach for when the point is exploration rather than fidelity.
///
/// t = 1.0 is off-manifold — 0.3498, against a 0.2620 random floor. The centroid direction is
/// load-bearing and cannot simply be projected out, which is why this is a knob and not a fix.
///
/// Component-wise max was measured on the same sample and is strictly worse (0.7483 / 19.0%).
/// Slerp at t = 0.5 reproduces the naive sum exactly, which is the harness check that says these
/// numbers are real and not an artefact of the sampling.
pub fn combine(vectors: &[&[f32]], centroid: &FieldCentroid, t: f32) -> Result<Vec<f32>> {
    if vectors.is_empty() {
        bail!("combine needs at least one vector");
    }
    ensure_dim64(&centroid.axis).context("field centroid is not 64D")?;
    let mut sum = vec![0.0f32; PACKET_DIM];
    for v in vectors {
        ensure_dim64(v)?;
        for (slot, x) in sum.iter_mut().zip(v.iter()) {
            *slot += *x;
        }
    }
    let along: f32 = sum.iter().zip(&centroid.axis).map(|(s, c)| s * c).sum();
    for (slot, c) in sum.iter_mut().zip(&centroid.axis) {
        *slot -= t * along * c;
    }
    normalize(&mut sum)
        .context("combined vector collapsed to zero length — inputs cancelled, or t overshot")?;
    Ok(sum)
}

/// What [`combine`] actually did, so a run can be read back without re-deriving it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombineReceipt {
    pub packet: MemoryPacket,
    /// Centroid-removal strength applied.
    pub t: f32,
    pub inputs: Vec<Uuid>,
    /// Splats the centroid was measured on.
    pub centroid_samples: usize,
    /// Cosine to the field centroid at t = 0, i.e. what the naive sum would have been,
    pub cosine_to_centroid_naive: f32,
    /// and after removal. The gap between these two is the whole point of the knob.
    pub cosine_to_centroid: f32,
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Standard base64 of little-endian f32 words (no external crate).
pub fn encode_f32_le_b64(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    crate::ingest::extract::base64_encode(&bytes)
}

pub fn decode_f32_le_b64(b64: &str) -> Result<Vec<f32>> {
    let bytes = base64_decode(b64)?;
    if bytes.len() % 4 != 0 {
        bail!("b64 float payload length {} not multiple of 4", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn base64_decode(input: &str) -> Result<Vec<u8>> {
    // Minimal decoder matching extract::base64_encode (standard alphabet + padding).
    const TABLE: &[u8; 256] = &{
        let mut t = [255u8; 256];
        let alpha = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            t[alpha[i] as usize] = i as u8;
            i += 1;
        }
        t
    };
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if !clean.len().is_multiple_of(4) {
        bail!("invalid base64 length");
    }
    let mut out = Vec::with_capacity(clean.len() / 4 * 3);
    for chunk in clean.chunks_exact(4) {
        let mut vals = [0u8; 4];
        let mut pad = 0;
        for i in 0..4 {
            if chunk[i] == b'=' {
                vals[i] = 0;
                pad += 1;
            } else {
                let v = TABLE[chunk[i] as usize];
                if v == 255 {
                    bail!("invalid base64 char");
                }
                vals[i] = v;
            }
        }
        let n = ((vals[0] as u32) << 18)
            | ((vals[1] as u32) << 12)
            | ((vals[2] as u32) << 6)
            | (vals[3] as u32);
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field with a real centre: every vector leans on axis 0, and differs after that.
    fn leaning_field(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; PACKET_DIM];
                v[0] = 0.8;
                v[1 + (i % (PACKET_DIM - 1))] = 0.6;
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect()
    }

    #[test]
    fn the_knob_walks_a_combined_state_off_the_centroid() {
        let field = leaning_field(64);
        let refs: Vec<&[f32]> = field.iter().map(|v| v.as_slice()).collect();
        let centroid = FieldCentroid::estimate(&refs).expect("leaning field has a centre");

        let pair = [field[0].as_slice(), field[9].as_slice()];
        let naive = combine(&pair, &centroid, 0.0).unwrap();
        let knobbed = combine(&pair, &centroid, DEFAULT_COMBINE_T).unwrap();

        let naive_c = cosine(&naive, &centroid.axis);
        let knobbed_c = cosine(&knobbed, &centroid.axis);
        assert!(
            naive_c > knobbed_c,
            "naive sum must sit closer to the centre than the knobbed one: {naive_c} vs {knobbed_c}"
        );
        // Monotone in t is the property the sweep table depends on being true.
        let mut last = f32::INFINITY;
        for step in 0..=5 {
            let t = step as f32 / 5.0;
            let c = cosine(&combine(&pair, &centroid, t).unwrap(), &centroid.axis);
            assert!(
                c <= last + 1e-5,
                "cos to centroid must fall with t, broke at t={t}"
            );
            last = c;
        }
    }

    #[test]
    fn t_zero_is_exactly_the_naive_sum() {
        let field = leaning_field(32);
        let refs: Vec<&[f32]> = field.iter().map(|v| v.as_slice()).collect();
        let centroid = FieldCentroid::estimate(&refs).unwrap();

        let mut expected: Vec<f32> = field[0].iter().zip(&field[5]).map(|(a, b)| a + b).collect();
        normalize(&mut expected).unwrap();

        let got = combine(&[field[0].as_slice(), field[5].as_slice()], &centroid, 0.0).unwrap();
        assert!(
            cosine(&got, &expected) > 0.999_999,
            "t = 0 must leave the operator alone"
        );
    }

    #[test]
    fn full_removal_leaves_no_centroid_component() {
        let field = leaning_field(48);
        let refs: Vec<&[f32]> = field.iter().map(|v| v.as_slice()).collect();
        let centroid = FieldCentroid::estimate(&refs).unwrap();
        let stripped =
            combine(&[field[2].as_slice(), field[11].as_slice()], &centroid, 1.0).unwrap();
        let c = cosine(&stripped, &centroid.axis);
        assert!(
            c.abs() < 1e-5,
            "t = 1 must be orthogonal to the centre, got {c}"
        );
    }

    #[test]
    fn combining_a_vector_with_its_own_negation_is_an_error_not_a_zero_packet() {
        let field = leaning_field(16);
        let refs: Vec<&[f32]> = field.iter().map(|v| v.as_slice()).collect();
        let centroid = FieldCentroid::estimate(&refs).unwrap();
        let flipped: Vec<f32> = field[0].iter().map(|x| -x).collect();
        assert!(
            combine(&[field[0].as_slice(), flipped.as_slice()], &centroid, 0.0).is_err(),
            "a cancelled sum has no direction and must not be dressed up as a packet"
        );
    }

    #[test]
    fn raw_pack_roundtrip_cosine_is_one() {
        let mut sem = vec![0.0f32; 64];
        sem[0] = 0.6;
        sem[1] = -0.3;
        sem[7] = 0.4;
        let n = (sem.iter().map(|x| x * x).sum::<f32>()).sqrt();
        for s in &mut sem {
            *s /= n;
        }
        let packet = MemoryPacket::from_semantics(
            Some(Uuid::nil()),
            &sem,
            -0.2,
            -1.0,
            Some("basin-a".into()),
            true,
        )
        .unwrap();
        let got = packet.resolve_semantics(None).unwrap();
        assert!((cosine(&sem, &got) - 1.0).abs() < 1e-5);
        assert!((packet.gain + 0.2).abs() < 1e-6);
        assert!(packet.mass < 0.0);
        assert!(packet.basin_locked);

        // b64 path
        let mut stripped = packet.clone();
        stripped.semantics_64.clear();
        let got2 = stripped.resolve_semantics(None).unwrap();
        assert!((cosine(&sem, &got2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn unicode_vq_reencode_matches() {
        let cb = VqCodebook::synthetic(32, 7);
        let mut sem = cb.centroids[3].clone();
        // jitter still nearest to 3 for mild noise
        sem[0] += 0.01;
        let n = (sem.iter().map(|x| x * x).sum::<f32>()).sqrt();
        for s in &mut sem {
            *s /= n;
        }
        assert!(cb.reencode_match(&sem).unwrap());

        let packet = MemoryPacket::from_semantics(None, &sem, 1.0, 1.0, None, false)
            .unwrap()
            .with_unicode(&cb)
            .unwrap();
        assert!(packet.unicode.is_some());
        let decoded = cb.decode_string(packet.unicode.as_ref().unwrap()).unwrap();
        let (ch1, _) = cb.encode_one(&sem).unwrap();
        let (ch2, _) = cb.encode_one(&decoded).unwrap();
        assert_eq!(ch1, ch2);
    }

    #[test]
    fn pua_index_roundtrip() {
        for i in [0usize, 1, 255, 1000] {
            let c = index_to_char(i).unwrap();
            assert_eq!(char_to_index(c).unwrap(), i);
        }
    }

    #[test]
    fn b64_float_roundtrip() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let b = encode_f32_le_b64(&v);
        let back = decode_f32_le_b64(&b).unwrap();
        for (a, b) in v.iter().zip(&back) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
