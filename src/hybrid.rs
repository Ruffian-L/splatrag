//! Ant-slime hybrid confidence for memory retrieval.
//!
//! `radiance` is connectivity at dream time; `Splat.usage` is retrieval count (τ).
//! Wired into `pick` only when `PickConfig.hybrid` is set. Retrieval / SciFact stay off.
//!
//! Distance uses a flux ceiling, not a tiny epsilon:
//! `L_eff = max(0, 1 - cosine) + 1/f_max`, so `F = D / L_eff` is bounded by
//! `D * f_max`. Evaporation is `rho / (F + kappa)` and capped. τ ranking,
//! evaporation, and D-updates only run inside the query's cosine neighborhood
//! so unrelated traffic cannot starve a holdout or export habit to another topic.

use crate::geometry::Splat;
use crate::record::RecallHit;
use std::collections::HashMap;
use uuid::Uuid;

/// Ranking + evaporation knobs. Extra fields exist because global `rho/F`
/// wipes a distant holdout (assert 3) and raw `dD/dt = F - γD` still grows
/// the long bridge when F is only a few units.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Weight of usage habit (τ).
    pub alpha: f32,
    /// Weight of contextual flux (F).
    pub beta: f32,
    /// Perfect-match flux multiplier at D=1. Defines `epsilon = 1/f_max`.
    pub f_max: f32,
    /// Evaporation pressure.
    pub rho: f32,
    /// Evaporation floor. Even F = F_max still forgets `rho / (f_max + kappa)`.
    pub kappa: f32,
    /// Conductivity decay in `D ← (1-γ)D + F/(F+f_max)`.
    pub gamma: f32,
    /// Max fraction of τ evaporated in one step.
    pub evap_cap: f32,
    /// Cosine floor. Below this, this query does not evaporate, adapt D,
    /// or let τ rank — habit stays in the active neighborhood.
    pub neighborhood: f32,
    /// Pheromone deposited on the chosen tube.
    pub delta_tau: f32,
    pub d_min: f32,
    pub d_max: f32,
    /// If finite, τ with I_R above this collapses this cycle (Easyway).
    /// Default `INFINITY` = off, so the cage can be measured first.
    pub trap_awareness_threshold: f32,
    /// Multiply τ by this on collapse. 0.05 keeps 5%.
    pub sensory_collapse_factor: f32,
    /// Hard ceiling after Easyway collapse. Must sit below a fresh `delta_tau`
    /// (1.0) so leftover habit cannot beat a new pick on usage alone.
    pub collapse_tau_cap: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            // Meaning must outrank a small leftover-τ gap. β=1 already does that
            // on the live 0.90 vs 0.32 cosine split; 2.0 leaves margin.
            beta: 2.0,
            f_max: 100.0,
            rho: 0.2,
            kappa: 1.0,
            // Must sit between 1/(f_max L_long) and 1/(f_max L_holdout) so a
            // 0.55 habit atrophies while a 0.90 holdout can still thicken.
            gamma: 0.05,
            evap_cap: 0.25,
            // 64d live long-bridge cosine sat at 0.325; 0.40 zeroed τ and the cage
            // could not close. Holdout C on Q1 is 0.10, so 0.25 still localizes.
            neighborhood: 0.25,
            delta_tau: 1.0,
            d_min: 0.05,
            d_max: 8.0,
            trap_awareness_threshold: f32::INFINITY,
            sensory_collapse_factor: 0.05,
            collapse_tau_cap: 0.5,
        }
    }
}

impl HybridConfig {
    /// Habit-only: Deneubourg lock-in. Flux is computed but does not rank.
    pub fn trap() -> Self {
        Self {
            beta: 0.0,
            ..Self::default()
        }
    }

    /// Cage-breaker: flux on, high I_R shatters τ this cycle, leftover capped < 1.
    pub fn easyway() -> Self {
        Self {
            trap_awareness_threshold: 3.0,
            sensory_collapse_factor: 0.05,
            collapse_tau_cap: 0.5,
            ..Self::default()
        }
    }

    /// Live pick knobs. `beta = 0` is the cage. Easyway is the shatter and
    /// needs flux (`beta = 1`); otherwise collapsed τ still ranks first.
    pub fn for_pick(beta: f32, easyway: bool) -> Self {
        if easyway {
            Self::easyway()
        } else {
            Self {
                beta: beta.max(0.0),
                ..Self::default()
            }
        }
    }

    fn f_max(&self) -> f32 {
        self.f_max.max(1.0)
    }
}

/// One memory pathway. Cosine is per-query, not stored on the tube.
#[derive(Debug, Clone)]
pub struct Tube {
    pub id: String,
    pub tau: f32,
    pub conductivity: f32,
}

#[derive(Debug, Clone)]
pub struct Scored {
    pub id: String,
    pub cosine: f32,
    pub flux: f32,
    pub confidence: f32,
}

pub fn epsilon(cfg: &HybridConfig) -> f32 {
    1.0 / cfg.f_max()
}

pub fn length(cosine: f32, cfg: &HybridConfig) -> f32 {
    let cos = cosine.clamp(0.0, 1.0);
    (1.0 - cos).max(0.0) + epsilon(cfg)
}

pub fn flux(conductivity: f32, cosine: f32, cfg: &HybridConfig) -> f32 {
    conductivity.max(0.0) / length(cosine, cfg)
}

pub fn confidence(tau: f32, flux: f32, cfg: &HybridConfig) -> f32 {
    let x = cfg.alpha * tau.max(0.0) + cfg.beta * flux.max(0.0);
    x * x
}

/// Usage only ranks inside the query neighborhood. Otherwise a tube that
/// won many unrelated queries carries τ into a topic it does not own.
pub fn effective_tau(tau: f32, cosine: f32, cfg: &HybridConfig) -> f32 {
    if cosine >= cfg.neighborhood {
        tau.max(0.0)
    } else {
        0.0
    }
}

/// Habit over truth. High I_R is the Big Monster: score is τ, not flux.
pub fn illusion_ratio(tau: f32, flux: f32, cfg: &HybridConfig) -> f32 {
    (cfg.alpha * tau.max(0.0)) / (cfg.beta * flux.max(0.0) + cfg.kappa.max(1e-6))
}

pub fn should_collapse(tau: f32, flux: f32, cfg: &HybridConfig) -> bool {
    cfg.trap_awareness_threshold.is_finite()
        && illusion_ratio(tau, flux, cfg) > cfg.trap_awareness_threshold
}

/// `min(ρ/(F+κ), evap_cap)` so a tiny F cannot dump the whole τ in one step.
pub fn evaporation_rate(flux: f32, cfg: &HybridConfig) -> f32 {
    let denom = flux.max(0.0) + cfg.kappa.max(1e-6);
    let raw = cfg.rho.max(0.0) / denom;
    raw.clamp(0.0, cfg.evap_cap.clamp(0.0, 1.0))
}

/// Saturating drive in `[0, 1)`. Equals 0.5 when F = F_max.
pub fn conductivity_drive(flux: f32, cfg: &HybridConfig) -> f32 {
    let f = flux.max(0.0);
    f / (f + cfg.f_max())
}

pub fn next_conductivity(d: f32, flux: f32, cfg: &HybridConfig) -> f32 {
    let gamma = cfg.gamma.clamp(0.0, 0.999);
    let next = (1.0 - gamma) * d.max(0.0) + conductivity_drive(flux, cfg);
    next.clamp(cfg.d_min.max(0.0), cfg.d_max.max(cfg.d_min.max(0.0)))
}

pub fn score_tubes(
    tubes: &[Tube],
    cosines: &HashMap<String, f32>,
    cfg: &HybridConfig,
) -> Vec<Scored> {
    tubes
        .iter()
        .map(|tube| {
            let cosine = cosines.get(&tube.id).copied().unwrap_or(0.0);
            let flux = flux(tube.conductivity, cosine, cfg);
            Scored {
                id: tube.id.clone(),
                cosine,
                flux,
                confidence: confidence(effective_tau(tube.tau, cosine, cfg), flux, cfg),
            }
        })
        .collect()
}

pub fn choose<'a>(scored: &'a [Scored]) -> Option<&'a Scored> {
    scored.iter().max_by(|a, b| {
        a.confidence
            .total_cmp(&b.confidence)
            .then_with(|| a.id.cmp(&b.id))
    })
}

/// Shatter high-I_R τ before ranking. Returns ids that collapsed this cycle.
pub fn collapse_tubes(
    tubes: &mut [Tube],
    cosines: &HashMap<String, f32>,
    cfg: &HybridConfig,
) -> HashMap<String, ()> {
    let mut collapsed = HashMap::<String, ()>::new();
    for tube in tubes.iter_mut() {
        let cosine = cosines.get(&tube.id).copied().unwrap_or(0.0);
        if cosine < cfg.neighborhood {
            continue;
        }
        let f = flux(tube.conductivity, cosine, cfg);
        if should_collapse(tube.tau, f, cfg) {
            let shrunk = tube.tau.max(0.0) * cfg.sensory_collapse_factor.clamp(0.0, 1.0);
            tube.tau = shrunk.min(cfg.collapse_tau_cap.max(0.0));
            collapsed.insert(tube.id.clone(), ());
        }
    }
    collapsed
}

/// One query: optional Easyway collapse, then score, then mild evap/deposit.
pub fn step(
    tubes: &mut [Tube],
    cosines: &HashMap<String, f32>,
    cfg: &HybridConfig,
) -> Option<String> {
    let collapsed = collapse_tubes(tubes, cosines, cfg);
    let scored = score_tubes(tubes, cosines, cfg);
    let chosen = choose(&scored)?.id.clone();
    for tube in tubes.iter_mut() {
        let cosine = cosines.get(&tube.id).copied().unwrap_or(0.0);
        let f = flux(tube.conductivity, cosine, cfg);
        let in_hood = cosine >= cfg.neighborhood || tube.id == chosen;
        if in_hood {
            if !collapsed.contains_key(&tube.id) {
                tube.tau = ((1.0 - evaporation_rate(f, cfg)) * tube.tau).max(0.0);
            }
            tube.conductivity = next_conductivity(tube.conductivity, f, cfg);
        }
        if tube.id == chosen {
            tube.tau += cfg.delta_tau.max(0.0);
        }
    }
    Some(chosen)
}

fn live_conductivity(d: f32) -> f32 {
    if d > 0.0 { d } else { 1.0 }
}

fn tube_from_hit(hit: &RecallHit, splat: Option<&Splat>) -> Tube {
    Tube {
        id: hit.memory.id.to_string(),
        tau: splat.map(|s| s.usage.max(0.0)).unwrap_or(0.0),
        conductivity: splat
            .map(|s| live_conductivity(s.conductivity))
            .unwrap_or(1.0),
    }
}

fn cosine_map(hits: &[RecallHit]) -> HashMap<String, f32> {
    hits.iter()
        .map(|hit| (hit.memory.id.to_string(), hit.scores.cosine.clamp(0.0, 1.0)))
        .collect()
}

/// Increment usage on one splat. Never touches radiance.
pub fn bump_usage(splat: &mut Splat, delta: f32) {
    splat.usage = (splat.usage + delta.max(0.0)).max(0.0);
}

/// Fill ANN-miss cosine (0) from splat 64-d vs the query. Pick-only — recall
/// leaves the 0 on purpose for SciFact. Friendship-size pools otherwise drop
/// a BM25 gold into L≈1 and hybrid buries it.
pub fn fill_missing_cosine(hits: &mut [RecallHit], splats: &[Splat], query_64: &[f32]) {
    if query_64.is_empty() {
        return;
    }
    let splat_by_id: HashMap<Uuid, &Splat> = splats
        .iter()
        .map(|splat| (splat.memory_id, splat))
        .collect();
    for hit in hits.iter_mut() {
        if hit.scores.cosine > 0.0 {
            continue;
        }
        let Some(splat) = splat_by_id.get(&hit.memory.id) else {
            continue;
        };
        if splat.semantics.is_empty() {
            continue;
        }
        hit.scores.cosine = crate::geometry::cosine(query_64, &splat.semantics).clamp(0.0, 1.0);
    }
}

/// Re-rank a pick/recall pool by hybrid confidence. Leaves BM25/cosine/radiance
/// breakdowns in place; only `final_score` (and order) change.
pub fn rerank_hits(hits: &mut [RecallHit], splats: &[Splat], cfg: &HybridConfig) {
    if hits.is_empty() {
        return;
    }
    let splat_by_id: HashMap<Uuid, &Splat> = splats
        .iter()
        .map(|splat| (splat.memory_id, splat))
        .collect();
    let mut tubes: Vec<Tube> = hits
        .iter()
        .map(|hit| tube_from_hit(hit, splat_by_id.get(&hit.memory.id).copied()))
        .collect();
    let cosines = cosine_map(hits);
    collapse_tubes(&mut tubes, &cosines, cfg);
    let scored = score_tubes(&tubes, &cosines, cfg);
    let conf: HashMap<String, f32> = scored
        .iter()
        .map(|row| (row.id.clone(), row.confidence))
        .collect();
    for hit in hits.iter_mut() {
        if let Some(&c) = conf.get(&hit.memory.id.to_string()) {
            hit.scores.final_score = c;
        }
    }
    hits.sort_by(|a, b| {
        b.scores
            .final_score
            .total_cmp(&a.scores.final_score)
            .then_with(|| a.memory.id.cmp(&b.memory.id))
    });
}

/// One hybrid query against hot splats: evaporate/adapt neighborhood, deposit on winner.
/// Radiance is not written.
pub fn apply_step(splats: &mut [Splat], hits: &[RecallHit], cfg: &HybridConfig) -> Option<Uuid> {
    if hits.is_empty() {
        return None;
    }
    let splat_by_id: HashMap<Uuid, usize> = splats
        .iter()
        .enumerate()
        .map(|(i, splat)| (splat.memory_id, i))
        .collect();
    let mut tubes: Vec<Tube> = hits
        .iter()
        .map(|hit| {
            let splat = splat_by_id.get(&hit.memory.id).map(|&i| &splats[i]);
            tube_from_hit(hit, splat)
        })
        .collect();
    let chosen = step(&mut tubes, &cosine_map(hits), cfg)?;
    let chosen_id = Uuid::parse_str(&chosen).ok()?;
    let updated: HashMap<String, (f32, f32)> = tubes
        .iter()
        .map(|tube| (tube.id.clone(), (tube.tau, tube.conductivity)))
        .collect();
    for splat in splats.iter_mut() {
        if let Some(&(tau, d)) = updated.get(&splat.memory_id.to_string()) {
            splat.usage = tau;
            splat.conductivity = d;
        }
    }
    Some(chosen_id)
}

#[cfg(test)]
fn tube_by_id<'a>(tubes: &'a [Tube], id: &str) -> &'a Tube {
    tubes
        .iter()
        .find(|t| t.id == id)
        .unwrap_or_else(|| panic!("missing tube {id}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_bridge() -> Vec<Tube> {
        vec![
            Tube {
                id: "A".into(),
                tau: 20.0,
                conductivity: 1.0,
            },
            Tube {
                id: "B".into(),
                tau: 0.0,
                conductivity: 1.0,
            },
        ]
    }

    /// Long bridge A (habit, worse match), short bridge B (new, better),
    /// holdout C (right for Q2), distractor E.
    fn store() -> Vec<Tube> {
        vec![
            Tube {
                id: "A".into(),
                tau: 20.0,
                conductivity: 1.0,
            },
            Tube {
                id: "B".into(),
                tau: 0.0,
                conductivity: 1.0,
            },
            Tube {
                id: "C".into(),
                tau: 20.0,
                conductivity: 1.0,
            },
            Tube {
                id: "E".into(),
                tau: 0.0,
                conductivity: 1.0,
            },
        ]
    }

    fn q1() -> HashMap<String, f32> {
        HashMap::from([
            ("A".into(), 0.55),
            ("B".into(), 0.99),
            ("C".into(), 0.10),
            ("E".into(), 0.08),
        ])
    }

    fn q2() -> HashMap<String, f32> {
        HashMap::from([
            ("A".into(), 0.12),
            ("B".into(), 0.11),
            ("C".into(), 0.90),
            ("E".into(), 0.50),
        ])
    }

    fn winner_id(tubes: &[Tube], q: &HashMap<String, f32>, cfg: &HybridConfig) -> String {
        choose(&score_tubes(tubes, q, cfg))
            .expect("non-empty")
            .id
            .clone()
    }

    /// Assert 1. Habit-only ranking prefers the saturated worse match.
    #[test]
    fn trap_habit_only_prefers_the_long_bridge() {
        let tubes = two_bridge();
        let cfg = HybridConfig::trap();
        assert_eq!(winner_id(&tubes, &q1(), &cfg), "A");
        let scored = score_tubes(&tubes, &q1(), &cfg);
        let a = scored.iter().find(|s| s.id == "A").unwrap();
        let b = scored.iter().find(|s| s.id == "B").unwrap();
        assert!(
            a.confidence > b.confidence,
            "A {} vs B {}",
            a.confidence,
            b.confidence
        );
        assert!(a.confidence.is_finite() && b.confidence.is_finite());
        // Same τ on an off-topic holdout ties A; flux is what breaks the tie.
        assert_ne!(winner_id(&store(), &q1(), &cfg), "B");
    }

    /// Assert 2. Flux on the short path beats the saturated long path.
    #[test]
    fn escape_flux_prefers_the_short_bridge() {
        let tubes = store();
        let cfg = HybridConfig::default();
        assert_eq!(winner_id(&tubes, &q1(), &cfg), "B");
        let scored = score_tubes(&tubes, &q1(), &cfg);
        let a = scored.iter().find(|s| s.id == "A").unwrap();
        let b = scored.iter().find(|s| s.id == "B").unwrap();
        assert!(
            b.confidence > a.confidence,
            "B {} vs A {}",
            b.confidence,
            a.confidence
        );
    }

    /// Assert 3. Q traffic must not starve the holdout that still owns Q2.
    #[test]
    fn holdout_survives_unrelated_traffic() {
        let mut tubes = store();
        let cfg = HybridConfig::default();
        let tau_c0 = tube_by_id(&tubes, "C").tau;
        let d_c0 = tube_by_id(&tubes, "C").conductivity;
        for _ in 0..24 {
            let chosen = step(&mut tubes, &q1(), &cfg).unwrap();
            assert_eq!(chosen, "B");
        }
        let c = tube_by_id(&tubes, "C");
        assert!(
            (c.tau - tau_c0).abs() < 1e-5,
            "Q traffic moved holdout tau {} -> {}",
            tau_c0,
            c.tau
        );
        assert!(
            (c.conductivity - d_c0).abs() < 1e-5,
            "Q traffic moved holdout D {} -> {}",
            d_c0,
            c.conductivity
        );
        assert_eq!(winner_id(&tubes, &q2(), &cfg), "C");
        let scored = score_tubes(&tubes, &q2(), &cfg);
        let c_s = scored.iter().find(|s| s.id == "C").unwrap();
        let e_s = scored.iter().find(|s| s.id == "E").unwrap();
        assert!(
            c_s.confidence > e_s.confidence,
            "C {} vs E {}",
            c_s.confidence,
            e_s.confidence
        );
    }

    /// Assert 4. Short tube thickens; long tube atrophies; holdout waits.
    #[test]
    fn atrophy_is_relative() {
        let mut tubes = store();
        let cfg = HybridConfig::default();
        let d_a0 = tube_by_id(&tubes, "A").conductivity;
        let d_b0 = tube_by_id(&tubes, "B").conductivity;
        let d_c0 = tube_by_id(&tubes, "C").conductivity;
        for _ in 0..24 {
            step(&mut tubes, &q1(), &cfg);
        }
        let d_a = tube_by_id(&tubes, "A").conductivity;
        let d_b = tube_by_id(&tubes, "B").conductivity;
        let d_c = tube_by_id(&tubes, "C").conductivity;
        assert!(d_b > d_b0, "short D {d_b0} -> {d_b}");
        assert!(d_a < d_a0, "long D {d_a0} -> {d_a}");
        assert!(d_b > d_a, "short {d_b} should exceed long {d_a}");
        assert!((d_c - d_c0).abs() < 1e-5, "holdout D moved {d_c0} -> {d_c}");
        step(&mut tubes, &q2(), &cfg);
        let d_c1 = tube_by_id(&tubes, "C").conductivity;
        assert!(d_c1 > d_c0, "Q2 should thicken holdout {d_c0} -> {d_c1}");
    }

    /// Brick 1 control: without τ in the score, cosine already prefers B.
    #[test]
    fn shipped_score_without_tau_prefers_the_short_bridge() {
        let q = q1();
        let a = q["A"];
        let b = q["B"];
        assert!(b > a, "cosine B {b} should beat A {a}");
    }

    #[test]
    fn f_max_caps_exact_match_and_keeps_ordinality() {
        let cfg = HybridConfig::default();
        let f_exact = flux(1.0, 1.0, &cfg);
        let f_near = flux(1.0, 0.99, &cfg);
        let f_good = flux(1.0, 0.95, &cfg);
        assert!((f_exact - cfg.f_max).abs() < 1e-4, "exact F {f_exact}");
        assert!((f_near - 50.0).abs() < 1e-3, "0.99 F {f_near}");
        assert!(f_good > 16.0 && f_good < 17.0, "0.95 F {f_good}");
        assert!(f_exact > f_near && f_near > f_good);
        assert!(f_exact.is_finite() && f_near.is_finite());
        let c_exact = confidence(0.0, f_exact, &cfg);
        let c_near = confidence(0.0, f_near, &cfg);
        assert!(c_exact.is_finite() && c_exact < 1.0e10);
        assert!(c_exact > c_near);

        let rate_exact = evaporation_rate(f_exact, &cfg);
        let rate_zero = evaporation_rate(0.0, &cfg);
        assert!(rate_exact > 0.0, "perfect match must still evaporate");
        assert!(rate_exact < rate_zero);
        assert!(rate_zero <= cfg.evap_cap + 1e-6);
    }

    /// Why neighborhood + cap exist: global ρ/F eats the holdout.
    #[test]
    fn unlocalized_evaporation_starves_the_holdout() {
        let mut tubes = store();
        let cfg = HybridConfig {
            neighborhood: 0.0,
            evap_cap: 1.0,
            ..HybridConfig::default()
        };
        let tau_c0 = tube_by_id(&tubes, "C").tau;
        for _ in 0..24 {
            step(&mut tubes, &q1(), &cfg);
        }
        let tau_c = tube_by_id(&tubes, "C").tau;
        assert!(
            tau_c < 0.2 * tau_c0,
            "expected starve, tau {tau_c0} -> {tau_c}"
        );
    }

    fn two_bridge_hits() -> (Vec<crate::record::RecallHit>, crate::geometry::HotState) {
        use crate::geometry::{HotState, Splat};
        use crate::record::{MemoryRecord, RecallContext, RecallHit, ScoreBreakdown};
        let a = RecallHit {
            memory: MemoryRecord::new("test", "A", "old habit"),
            context: RecallContext::default(),
            basin_id: None,
            basin_label: None,
            scores: ScoreBreakdown {
                bm25: 2.0,
                cosine: 0.55,
                radiance: 0.9,
                radiance_weight: 5.0,
                final_score: 0.55,
            },
        };
        let b = RecallHit {
            memory: MemoryRecord::new("test", "B", "new relevant"),
            context: RecallContext::default(),
            basin_id: None,
            basin_label: None,
            scores: ScoreBreakdown {
                bm25: 8.0,
                cosine: 0.99,
                radiance: 0.1,
                radiance_weight: 5.0,
                final_score: 0.99,
            },
        };
        let hits = vec![a, b];
        let mut hot = HotState::default();
        for hit in &hits {
            let mut splat = Splat::from_embedding(&hit.memory, &vec![0.125; 64]).unwrap();
            if hit.memory.source_key == "A" {
                splat.usage = 20.0;
                splat.radiance = 9.0;
            }
            hot.splats.push(splat);
        }
        (hits, hot)
    }

    #[test]
    fn bump_usage_does_not_touch_radiance() {
        let (_hits, mut hot) = two_bridge_hits();
        let splat = hot.splats.iter_mut().find(|s| s.usage > 0.0).unwrap();
        let radiance = splat.radiance;
        let usage = splat.usage;
        bump_usage(splat, 1.0);
        assert!((splat.usage - usage - 1.0).abs() < 1e-6);
        assert_eq!(splat.radiance, radiance);
    }

    #[test]
    fn rerank_trap_then_escape_on_pick_pool() {
        let (mut hits, hot) = two_bridge_hits();
        let a_id = hits[0].memory.id;
        let b_id = hits[1].memory.id;
        rerank_hits(&mut hits, &hot.splats, &HybridConfig::trap());
        assert_eq!(
            hits[0].memory.id, a_id,
            "habit-only must pick the long bridge"
        );
        rerank_hits(&mut hits, &hot.splats, &HybridConfig::default());
        assert_eq!(hits[0].memory.id, b_id, "flux must pick the short bridge");
    }

    #[test]
    fn apply_step_writes_usage_not_radiance() {
        let (hits, mut hot) = two_bridge_hits();
        let rad_before: Vec<f32> = hot.splats.iter().map(|s| s.radiance).collect();
        let chosen = apply_step(&mut hot.splats, &hits, &HybridConfig::default()).unwrap();
        let b_id = hits
            .iter()
            .find(|h| h.memory.source_key == "B")
            .unwrap()
            .memory
            .id;
        assert_eq!(chosen, b_id);
        let b = hot.splats.iter().find(|s| s.memory_id == b_id).unwrap();
        assert!(b.usage > 0.0, "winner should receive Δτ");
        let rad_after: Vec<f32> = hot.splats.iter().map(|s| s.radiance).collect();
        assert_eq!(rad_before, rad_after);
    }

    /// Massive τ would cage even with flux; Easyway shatters it before ranking.
    #[test]
    fn easyway_shatters_the_cage_in_one_step() {
        let mut tubes = two_bridge();
        tubes[0].tau = 100.0;
        let cage = HybridConfig::default();
        assert_eq!(winner_id(&tubes, &q1(), &cage), "A");
        let chosen = step(&mut tubes, &q1(), &HybridConfig::easyway()).unwrap();
        assert_eq!(chosen, "B");
        assert!(
            tube_by_id(&tubes, "A").tau <= 0.5 + 1e-5,
            "collapsed τ must sit below a fresh pick, got {}",
            tube_by_id(&tubes, "A").tau
        );
    }

    #[test]
    fn rerank_easyway_picks_the_short_bridge() {
        let (mut hits, hot) = two_bridge_hits();
        let a_id = hits[0].memory.id;
        let b_id = hits[1].memory.id;
        rerank_hits(&mut hits, &hot.splats, &HybridConfig::trap());
        assert_eq!(hits[0].memory.id, a_id, "cage must still pick A");
        rerank_hits(&mut hits, &hot.splats, &HybridConfig::easyway());
        assert_eq!(hits[0].memory.id, b_id, "easyway must pick B this cycle");
    }

    #[test]
    fn easyway_with_beta_zero_does_not_hand_the_query_to_b() {
        let mut tubes = two_bridge();
        tubes[0].tau = 24.0;
        let cfg = HybridConfig {
            beta: 0.0,
            trap_awareness_threshold: 3.0,
            sensory_collapse_factor: 0.05,
            ..HybridConfig::default()
        };
        assert_eq!(
            step(&mut tubes, &q1(), &cfg).unwrap(),
            "A",
            "collapsed leftover τ still ranks if flux is off"
        );
    }

    #[test]
    fn easyway_write_caps_tau_below_a_fresh_pick() {
        let (hits, mut hot) = two_bridge_hits();
        apply_step(&mut hot.splats, &hits, &HybridConfig::easyway()).unwrap();
        let a = hot
            .splats
            .iter()
            .find(|s| s.memory_id == hits[0].memory.id)
            .unwrap();
        let b = hot
            .splats
            .iter()
            .find(|s| s.memory_id == hits[1].memory.id)
            .unwrap();
        assert!(
            a.usage <= 0.5 + 1e-5,
            "A leftover usage {}, cap 0.5",
            a.usage
        );
        assert!(
            b.usage >= 1.0 - 1e-5,
            "B should receive Δτ, got {}",
            b.usage
        );
        let mut again = hits.clone();
        rerank_hits(&mut again, &hot.splats, &HybridConfig::trap());
        assert_eq!(
            again[0].memory.id, hits[1].memory.id,
            "next habit-only pick must be B"
        );
    }

    #[test]
    fn fill_missing_cosine_uses_splat_semantics() {
        let (mut hits, hot) = two_bridge_hits();
        hits[0].scores.cosine = 0.0;
        let query = vec![0.125_f32; 64];
        fill_missing_cosine(&mut hits, &hot.splats, &query);
        assert!(
            hits[0].scores.cosine > 0.5,
            "ANN-miss should fill from splat, got {}",
            hits[0].scores.cosine
        );
    }

    #[test]
    fn small_usage_gap_cannot_beat_a_wide_cosine_gap() {
        let tubes = vec![
            Tube {
                id: "A".into(),
                tau: 1.2,
                conductivity: 1.0,
            },
            Tube {
                id: "B".into(),
                tau: 1.0,
                conductivity: 1.0,
            },
        ];
        let q = HashMap::from([("A".into(), 0.325), ("B".into(), 0.90)]);
        assert_eq!(
            winner_id(&tubes, &q, &HybridConfig::default()),
            "B",
            "0.2 leftover τ must not beat 0.90 vs 0.32 cosine"
        );
    }
}
