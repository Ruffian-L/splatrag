use crate::ann::KeyedAnn;
use crate::config::PhysicsConfig;
use crate::geometry::{HotState, cosine};
use crate::inversion::mass_from_radiance;
use crate::topology::{PersistenceInterval, discover_basins};
use anyhow::Result;
use chrono::Utc;
use rayon::prelude::*;
use std::collections::HashMap;

/// Floor on pair distance. Newtonian attraction diverges as separation approaches zero, so two
/// near-identical memories would otherwise produce an unbounded force and fling the field apart.
const MIN_SEPARATION: f32 = 0.05;

/// Floor on the similarity weight. Gravity must never reach exactly zero, or unrelated memories
/// detach from the field entirely and drift instead of settling into their own basin.
const MIN_ATTRACTION_WEIGHT: f32 = 0.05;

/// Per-splat movement in one step below which the field counts as settled.
const SETTLED_DISPLACEMENT_PER_SPLAT: f32 = 1e-6;

#[derive(Debug, Clone)]
pub struct DreamReport {
    pub steps: usize,
    pub kinetic_energy: f32,
    pub basins: usize,
    pub persistence_intervals: Vec<PersistenceInterval>,
}

pub fn dream(state: &mut HotState, ann: &KeyedAnn, config: &PhysicsConfig) -> Result<DreamReport> {
    dream_with_progress(state, ann, config, 0, |_, _, _| {})
}

/// Same physics as [`dream`], but calls `on_step(step, kinetic_energy, state)` every `emit_every`
/// completed steps (and once on the final step). `emit_every == 0` disables mid-run hooks
/// (final hook still fires).
pub fn dream_with_progress<F>(
    state: &mut HotState,
    ann: &KeyedAnn,
    config: &PhysicsConfig,
    emit_every: usize,
    mut on_step: F,
) -> Result<DreamReport>
where
    F: FnMut(usize, f32, &HotState),
{
    if state.splats.is_empty() {
        return Ok(DreamReport {
            steps: 0,
            kinetic_energy: 0.0,
            basins: 0,
            persistence_intervals: Vec::new(),
        });
    }
    // Fit the semantic basis before any force is computed. A dream run on bootstrap positions is
    // simulating a coordinate system that does not reflect meaning.
    state.ensure_projection();
    let index_by_id: HashMap<_, _> = state
        .splats
        .iter()
        .enumerate()
        .map(|(index, splat)| (splat.memory_id, index))
        .collect();
    let semantic_neighbors: Vec<Vec<usize>> = state
        .splats
        .iter()
        .map(|splat| {
            ann.neighbors_for_key(splat.memory_id)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|id| index_by_id.get(&id).copied())
                .take(32)
                .collect()
        })
        .collect();
    for (splat, neighbors) in state.splats.iter_mut().zip(&semantic_neighbors) {
        splat.radiance = 1.0 + (neighbors.len() as f32 + 1.0).ln();
        // Magnitude from connectivity; sign is an independent physics knob.
        // Negative mass → repels. Negative gain is not consulted here (that inverts
        // semantics at steer time, already baked into splat.semantics).
        splat.mass = mass_from_radiance(splat.mass, splat.radiance);
        // Domain/state colors so the live field is readable (SciFact vs Urban, basin tint).
        splat.color_rgba = domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0);
    }

    let mut energy = f32::INFINITY;
    let mut steps_taken = 0;
    for _ in 0..config.steps {
        // One O(N^2) pass, as the original dream did. Attraction is Newtonian and therefore has
        // no range limit, so it cannot be answered with a neighbour list or a spatial grid: every
        // pair pulls at every step, and that is precisely what makes the field consolidate.
        let forces: Vec<[f32; 3]> = (0..state.splats.len())
            .into_par_iter()
            .map(|index| {
                let splat = &state.splats[index];
                let mut force = [
                    -splat.position[0] * config.origin_pull,
                    -splat.position[1] * config.origin_pull,
                    -splat.position[2] * config.origin_pull,
                ];
                for (other_index, other) in state.splats.iter().enumerate() {
                    if other_index == index {
                        continue;
                    }
                    let delta = subtract(other.position, splat.position);
                    let distance = length(delta).max(MIN_SEPARATION);

                    // Semantics set the SIGN of the pair force, distance only its magnitude.
                    // That ordering is the whole point. A gravity-style law (G*m*m/d^2) is
                    // dominated by 1/d^2, which spans orders of magnitude while cosine spans less
                    // than 20x — so it clumps by proximity and erases meaning as it runs. Measured
                    // on these memories, gravity took the semantic correlation from -0.85 to
                    // -0.00 over 300 steps. Sign-from-semantics keeps it near -0.80 and improves
                    // separation the longer it runs.
                    //
                    // The cross-domain penalty is subtracted from similarity rather than applied
                    // as a multiplier on repulsion, so a foreign domain can flip an otherwise
                    // attracting pair into a repelling one. That is what lets a junk domain be
                    // quarantined into its own basin instead of merely sitting loosely nearby.
                    //
                    // Negative mass is a separate bollard: it forces repulsion regardless of
                    // cosine. Negative gain does not live here — it already inverted semantics.
                    let mut weight = cosine(&splat.semantics, &other.semantics)
                        - config.semantic_threshold;
                    if splat.domain != other.domain {
                        weight -= config.cross_domain_repulsion;
                    }
                    if splat.mass < 0.0 || other.mass < 0.0 {
                        weight = weight.min(-0.05);
                    }

                    if weight > 0.0 {
                        // Attraction grows with distance, so a similar memory left far away is
                        // always pulled home.
                        add_scaled(&mut force, delta, config.attraction * weight);
                    } else {
                        // Repulsion decays as 1/(1+d^2). A non-decaying repulsion pushes with
                        // constant strength forever and ejects outliers to infinity — with the
                        // earlier law one splat reached 987 units out while the median sat at 4.
                        let falloff = 1.0 + distance * distance;
                        let strength = config.semantic_repulsion * (-weight) / falloff;
                        add_scaled(&mut force, delta, -strength / distance);
                    }

                    // Hard-core separation: keeps a settled cluster from collapsing onto a point.
                    if distance < config.repulsion_radius {
                        let strength = (config.repulsion_radius - distance) * config.repulsion;
                        add_scaled(&mut force, delta, -strength / distance);
                    }
                }
                force
            })
            .collect();

        energy = 0.0;
        let mut displacement = 0.0;
        for (splat, force) in state.splats.iter_mut().zip(forces) {
            // Inertial mass is |m|; sign already lived in the force (bollard rule above).
            let inertia = splat.mass.abs().max(0.1);
            let acceleration = [
                force[0] / inertia,
                force[1] / inertia,
                force[2] / inertia,
            ];
            for (axis, acceleration) in acceleration.iter().enumerate() {
                splat.velocity[axis] =
                    (splat.velocity[axis] + acceleration * config.dt) * config.damping;
                let delta = splat.velocity[axis] * config.dt;
                splat.position[axis] += delta;
                displacement += delta.abs();
            }
            energy += 0.5 * inertia * length(splat.velocity).powi(2);
        }
        steps_taken += 1;
        // Speed-tint: faster stars glow hotter so motion is visible in the live feed.
        if emit_every > 0 && steps_taken % emit_every == 0 {
            for splat in &mut state.splats {
                let speed = length(splat.velocity);
                splat.color_rgba =
                    domain_state_color(&splat.domain, splat.basin_id.as_deref(), speed);
            }
            on_step(steps_taken, energy, state);
        }
        // Settle on total movement rather than a fixed step count: a field still collapsing keeps
        // running, and one that has already found its shape stops instead of burning the budget.
        // The threshold scales with the corpus so it means "per splat", not "in total".
        if displacement < SETTLED_DISPLACEMENT_PER_SPLAT * state.splats.len() as f32 {
            break;
        }
    }

    let previous = state.basins.clone();
    let (basins, persistence_intervals) = discover_basins(&mut state.splats, &previous, config);
    state.basins = basins;
    // Re-tint after basin assignment so basin hue lands on settled membership.
    for splat in &mut state.splats {
        splat.color_rgba = domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0);
    }
    state.dream_cycle += 1;
    state.last_dream_at = Some(Utc::now());
    state.kinetic_energy = energy;
    on_step(steps_taken, energy, state);
    Ok(DreamReport {
        steps: steps_taken,
        kinetic_energy: energy,
        basins: state.basins.len(),
        persistence_intervals,
    })
}

/// Domain = primary hue family. Basin id = within-family tint. Speed = live brightness.
/// SciFact → cool cyan/blue. Urban (poison) → warm amber/magenta. Other → violet.
pub fn domain_state_color(domain: &str, basin_id: Option<&str>, speed: f32) -> [u8; 4] {
    let d = domain.to_ascii_lowercase();
    let (base_h, sat) = if d.contains("scifact") || d == "science" {
        (205.0, 0.72)
    } else if d.contains("urban") || d.contains("poison") || d.contains("junk") {
        (28.0, 0.82)
    } else if d.contains("chat") || d.contains("agent") {
        (275.0, 0.55)
    } else {
        (160.0, 0.45)
    };
    let basin_jitter = basin_id
        .map(|id| {
            let mut h = 0u32;
            for b in id.bytes() {
                h = h.wrapping_mul(31).wrapping_add(b as u32);
            }
            (h % 40) as f32 - 20.0
        })
        .unwrap_or(0.0);
    let hue = (base_h + basin_jitter).rem_euclid(360.0);
    let speed_boost = (speed * 8.0).clamp(0.0, 0.45);
    let value = (0.55 + speed_boost).clamp(0.35, 1.0);
    let alpha = (200.0 + speed_boost * 55.0).clamp(160.0, 255.0) as u8;
    let (r, g, b) = hsv_to_rgb(hue, sat, value);
    [r, g, b, alpha]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (rp, gp, bp) = match h as i32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((rp + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((gp + m) * 255.0).round().clamp(0.0, 255.0) as u8,
        ((bp + m) * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn length(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn add_scaled(target: &mut [f32; 3], vector: [f32; 3], scale: f32) {
    for axis in 0..3 {
        target[axis] += vector[axis] * scale;
    }
}
