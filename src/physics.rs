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
    state.dream_cycle += 1;
    state.last_dream_at = Some(Utc::now());
    state.kinetic_energy = energy;
    Ok(DreamReport {
        steps: steps_taken,
        kinetic_energy: energy,
        basins: state.basins.len(),
        persistence_intervals,
    })
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
