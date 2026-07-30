//! Ontological inversion operators on concept vectors.
//!
//! Ported from the measured OI baseline (`ontological-inversion`) and the Rust
//! primitives in `niodoo-connector` — not re-derived.
//!
//! Claim (measured): negative steering of a concept does not erase it; under a
//! coherent axis it moves toward a structured opposite. Sweet band α ≈ 0.15–0.30;
//! collapse past ~0.4. Default op is polarity-aware inversion (matches niodoo).
//!
//! These ops are dim-agnostic: any equal-length unit (or near-unit) vectors work.
//! SplatRAG applies them on 64-d matryoshka semantics.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Measured sweet-band floor / ceiling for |α| when inverting.
pub const ALPHA_MIN: f32 = 0.05;
pub const ALPHA_MAX: f32 = 0.35;
/// Past this |gain|, generations in the original OI work collapsed — still legal,
/// but callers should treat it as collapse-risk.
pub const COLLAPSE_ONSET: f32 = 0.40;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InversionOp {
    /// Set signed projection to −α·|p| (niodoo polarity_aware_inversion).
    #[default]
    Polarity,
    /// True Householder reflection, blended by strength: most stable operator.
    Householder,
    /// Fixed push opposite the concept direction: h − strength·‖h‖·d̂.
    NegativeGain,
}

impl InversionOp {
    pub fn parse(name: &str) -> Result<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "polarity" | "projection_polarity" | "polarity_aware" => Ok(Self::Polarity),
            "householder" | "reflect" => Ok(Self::Householder),
            "negative_gain" | "neg" | "gain" => Ok(Self::NegativeGain),
            other => bail!("unknown inversion op '{other}' (polarity|householder|negative_gain)"),
        }
    }
}

/// Clamp |gain| into the measured inversion sweet band. Sign is preserved.
pub fn clamp_alpha(gain: f32) -> f32 {
    let sign = if gain < 0.0 { -1.0 } else { 1.0 };
    let mag = gain.abs().clamp(ALPHA_MIN, ALPHA_MAX);
    sign * mag
}

pub fn collapse_risk(gain: f32) -> bool {
    gain.abs() >= COLLAPSE_ONSET
}

pub fn dot(a: &[f32], b: &[f32]) -> Result<f32> {
    ensure_same_len(a, b)?;
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

pub fn norm_sq(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum()
}

pub fn norm(v: &[f32]) -> f32 {
    norm_sq(v).sqrt()
}

pub fn cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    let denom = norm(a) * norm(b);
    if denom <= f32::EPSILON {
        return Ok(0.0);
    }
    Ok(dot(a, b)? / denom)
}

pub fn normalize(v: &[f32]) -> Result<Vec<f32>> {
    let n = norm(v);
    if n <= f32::EPSILON {
        bail!("cannot normalize zero vector");
    }
    Ok(v.iter().map(|x| x / n).collect())
}

pub fn projection_scalar(h: &[f32], target: &[f32]) -> Result<f32> {
    let denom = norm_sq(target);
    if denom <= f32::EPSILON {
        bail!("target vector has zero norm");
    }
    Ok(dot(h, target)? / denom)
}

/// h − (1+α)·p·target  — flips a positive projection through zero when α≥0.
pub fn naive_projection_flip(h: &[f32], target: &[f32], alpha: f32) -> Result<Vec<f32>> {
    ensure_same_len(h, target)?;
    let p = projection_scalar(h, target)?;
    Ok(h
        .iter()
        .zip(target)
        .map(|(hv, tv)| *hv - (1.0 + alpha) * p * *tv)
        .collect())
}

/// Set the signed component on `target` to −α·|p_before| (or −α if p≈0).
pub fn polarity_aware_inversion(h: &[f32], target: &[f32], alpha: f32) -> Result<Vec<f32>> {
    ensure_same_len(h, target)?;
    let p = projection_scalar(h, target)?;
    let target_p = polarity_target_projection(p, alpha);
    let correction = target_p - p;
    Ok(h
        .iter()
        .zip(target)
        .map(|(hv, tv)| *hv + correction * *tv)
        .collect())
}

pub fn polarity_target_projection(p_before: f32, alpha: f32) -> f32 {
    if p_before.abs() <= f32::EPSILON {
        -alpha
    } else {
        -alpha * p_before.abs()
    }
}

/// Φ(h) = (1−s)·h + s·(h − 2 (h·d̂) d̂). At s=1 this is a true involution.
pub fn householder_blend(h: &[f32], direction: &[f32], strength: f32) -> Result<Vec<f32>> {
    ensure_same_len(h, direction)?;
    let d = normalize(direction)?;
    let p = dot(h, &d)?;
    let s = strength.clamp(0.0, 1.0);
    Ok(h
        .iter()
        .zip(&d)
        .map(|(hv, dv)| {
            let reflected = *hv - 2.0 * p * *dv;
            (1.0 - s) * *hv + s * reflected
        })
        .collect())
}

/// Fixed push opposite the unit concept direction, scaled by ‖h‖·strength.
pub fn negative_gain_push(h: &[f32], direction: &[f32], strength: f32) -> Result<Vec<f32>> {
    ensure_same_len(h, direction)?;
    let d = normalize(direction)?;
    let scale = -strength * norm(h);
    Ok(h
        .iter()
        .zip(&d)
        .map(|(hv, dv)| *hv + scale * *dv)
        .collect())
}

pub fn householder_reflect(x: &[f32], normal: &[f32]) -> Result<Vec<f32>> {
    ensure_same_len(x, normal)?;
    let denom = norm_sq(normal);
    if denom <= f32::EPSILON {
        bail!("householder normal has zero norm");
    }
    let scale = 2.0 * dot(x, normal)? / denom;
    Ok(x
        .iter()
        .zip(normal)
        .map(|(xv, nv)| *xv - scale * *nv)
        .collect())
}

/// Apply an inversion / amplify operator to a concept vector.
///
/// **Gain is not mass.** Two independent knobs:
/// - `gain < 0` → **invert** semantics (ontological inversion / sorrowful flip)
/// - `gain > 0` → **amplify** along the concept axis
/// - `mass < 0` (elsewhere) → **repel** in the dream force law
///
/// Negative gain never sets mass. After a flip the new semantics may attract or
/// repel by cosine alone; bollard repulsion is an explicit negative-mass choice.
///
/// `axis` is the concept direction. Self-axis = pass the vector as both args.
pub fn apply_steering(
    semantics: &[f32],
    axis: &[f32],
    gain: f32,
    op: InversionOp,
) -> Result<Vec<f32>> {
    ensure_same_len(semantics, axis)?;
    if gain == 0.0 {
        return Ok(semantics.to_vec());
    }
    if gain > 0.0 {
        // Amplify along axis — still not a mass change.
        let d = normalize(axis)?;
        let scale = clamp_alpha(gain).abs() * norm(semantics);
        let mut out: Vec<f32> = semantics
            .iter()
            .zip(&d)
            .map(|(hv, dv)| *hv + scale * *dv)
            .collect();
        let n = norm(&out);
        if n > f32::EPSILON {
            for v in &mut out {
                *v /= n;
            }
        }
        return Ok(out);
    }

    // Negative gain: invert. Strength α = clamp(|gain|) into the measured sweet band.
    let alpha = clamp_alpha(gain).abs();
    let mut out = match op {
        InversionOp::Polarity => {
            let target = normalize(axis)?;
            polarity_aware_inversion(semantics, &target, alpha)?
        }
        InversionOp::Householder => householder_blend(semantics, axis, alpha.min(1.0))?,
        InversionOp::NegativeGain => negative_gain_push(semantics, axis, alpha)?,
    };
    let n = norm(&out);
    if n > f32::EPSILON {
        for v in &mut out {
            *v /= n;
        }
    }
    Ok(out)
}

/// Dream mass magnitude from radiance, **preserving** any explicit mass sign.
/// Negative mass stays negative (repels); positive stays positive. Gain is ignored.
pub fn mass_from_radiance(current_mass: f32, radiance: f32) -> f32 {
    let sign = if current_mass < 0.0 { -1.0 } else { 1.0 };
    sign * radiance.abs().max(0.1)
}

fn ensure_same_len(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        bail!("dimension mismatch: {} vs {}", a.len(), b.len());
    }
    if a.is_empty() {
        bail!("empty vectors are not valid");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polarity_sets_requested_signed_component() {
        let target = vec![1.0, 0.0];
        let h = vec![-0.25, 0.75];
        let inverted = polarity_aware_inversion(&h, &target, 2.0).unwrap();
        let p = projection_scalar(&inverted, &target).unwrap();
        assert!((p + 0.5).abs() < 1e-6);
    }

    #[test]
    fn householder_reflection_is_self_inverse() {
        let normal = vec![0.7, -0.2, 0.4];
        let x = vec![0.1, 0.5, -0.8];
        let once = householder_reflect(&x, &normal).unwrap();
        let twice = householder_reflect(&once, &normal).unwrap();
        for (actual, expected) in twice.iter().zip(&x) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn negative_gain_self_axis_reduces_self_cosine() {
        let mut sem = vec![0.0f32; 64];
        sem[0] = 1.0;
        let flipped = apply_steering(&sem, &sem, -0.2, InversionOp::Polarity).unwrap();
        let cos = cosine(&sem, &flipped).unwrap();
        assert!(
            cos < 0.5,
            "sorrowful flip should move off the original axis, cos={cos}"
        );
    }

    /// **The self-axis is degenerate, and α is inert on it.** Pinned so it cannot be mistaken for
    /// working steering, and so that anyone who makes α responsive sees this test fail on purpose.
    ///
    /// These operators act only on the component of `h` along the axis. When the axis *is* `h`, that
    /// component is the whole vector, and normalization discards the only quantity α scaled:
    ///
    /// - polarity: `out = h + (−α|p| − p)·ĥ = −α·h` → normalizes to exactly `−ĥ` for every α
    /// - householder: `out = h − 2α(h·ĥ)ĥ = (1 − 2α)·h` → `+ĥ` for all α < 0.5
    /// - negative_gain: `out = h − α‖h‖ĥ = (1 − α)·h` → `+ĥ`
    ///
    /// So `steer --gain` on a self-axis is a full 180° flip (polarity) or a no-op (the other two),
    /// with nothing in between — the measured α ≈ 0.15–0.30 sweet band cannot be expressed. A
    /// non-degenerate reference direction (basin centroid, field mean — the μ in the OI prior art's
    /// `Φ_c(h) = μ + (I − 2P_c)(h − μ)`) is required for α to mean anything.
    #[test]
    fn self_axis_steering_is_degenerate_and_alpha_does_nothing() {
        let mut sem = vec![0.0f32; 64];
        for (i, value) in sem.iter_mut().enumerate() {
            *value = ((i as f32) * 0.7).sin();
        }
        let sem = normalize(&sem).unwrap();

        for alpha in [-0.05f32, -0.1, -0.2, -0.3, -0.35, -0.9] {
            let flipped = apply_steering(&sem, &sem, alpha, InversionOp::Polarity).unwrap();
            let cos = cosine(&sem, &flipped).unwrap();
            assert!(
                (cos + 1.0).abs() < 1e-4,
                "polarity self-axis is always the exact antipode; α={alpha} gave {cos}"
            );

            for op in [InversionOp::Householder, InversionOp::NegativeGain] {
                let out = apply_steering(&sem, &sem, alpha, op).unwrap();
                let cos = cosine(&sem, &out).unwrap();
                // |α| is clamped to <= 0.35, so (1 - 2α) and (1 - α) both stay positive.
                assert!(
                    (cos - 1.0).abs() < 1e-4,
                    "{op:?} self-axis is a no-op; α={alpha} gave {cos}"
                );
            }
        }
    }

    /// With a reference direction the memory genuinely projects onto, α becomes responsive.
    ///
    /// This is the behaviour the sweet band was measured against, and the reason the fix for the
    /// degeneracy above is "choose a real axis", not "change the operators".
    #[test]
    fn a_distinct_axis_makes_alpha_monotonic() {
        let mut sem = vec![0.0f32; 64];
        sem[0] = 1.0;
        // Substantially aligned but not identical — what a basin centroid looks like.
        let mut axis = vec![0.0f32; 64];
        axis[0] = 0.8;
        axis[1] = 0.6;
        let axis = normalize(&axis).unwrap();
        assert!(cosine(&sem, &axis).unwrap() > 0.5, "axis must share a component");

        let mut previous = 1.0f32;
        for alpha in [-0.05f32, -0.15, -0.25, -0.35] {
            let out = apply_steering(&sem, &axis, alpha, InversionOp::NegativeGain).unwrap();
            let cos = cosine(&sem, &out).unwrap();
            assert!(
                cos < previous,
                "α={alpha} did not move further than the previous step ({cos} vs {previous})"
            );
            previous = cos;
        }
        // And it stays a rotation, not a collapse to the antipode.
        assert!(previous > -0.5, "sweet-band α should not reach the antipode, got {previous}");
    }

    #[test]
    fn positive_gain_stays_aligned() {
        let mut sem = vec![0.0f32; 64];
        sem[0] = 1.0;
        let boosted = apply_steering(&sem, &sem, 0.2, InversionOp::Polarity).unwrap();
        let cos = cosine(&sem, &boosted).unwrap();
        assert!(cos > 0.99, "amplify should stay on-axis, cos={cos}");
    }

    #[test]
    fn mass_from_radiance_preserves_sign_not_gain() {
        assert!(mass_from_radiance(-1.0, 2.0) < 0.0);
        assert!(mass_from_radiance(1.0, 2.0) > 0.0);
        // Positive mass + high radiance stays positive even if someone later sets gain < 0.
        assert!(mass_from_radiance(0.5, 3.0) > 0.0);
    }

    #[test]
    fn clamp_alpha_respects_sweet_band() {
        assert!((clamp_alpha(-0.01).abs() - ALPHA_MIN).abs() < 1e-6);
        assert!((clamp_alpha(-0.9).abs() - ALPHA_MAX).abs() < 1e-6);
        assert!(clamp_alpha(-0.2) < 0.0);
    }
}
