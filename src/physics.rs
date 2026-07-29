use crate::ann::KeyedAnn;
use crate::config::PhysicsConfig;
use crate::geometry::{HotState, cosine};
use crate::topology::{PersistenceInterval, discover_basins};
use anyhow::Result;
use chrono::Utc;
use rayon::prelude::*;
use std::collections::HashMap;

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
        splat.mass = splat.radiance.max(0.1);
    }

    let mut energy = f32::INFINITY;
    let mut steps_taken = 0;
    for step in 0..config.steps {
        let cells = spatial_cells(&state.splats, config.spatial_cell_size);
        let forces: Vec<[f32; 3]> = (0..state.splats.len())
            .into_par_iter()
            .map(|index| {
                let splat = &state.splats[index];
                let mut force = [
                    -splat.position[0] * config.origin_pull,
                    -splat.position[1] * config.origin_pull,
                    -splat.position[2] * config.origin_pull,
                ];
                for &other_index in &semantic_neighbors[index] {
                    if other_index == index {
                        continue;
                    }
                    let other = &state.splats[other_index];
                    let similarity = cosine(&splat.semantics, &other.semantics);
                    if similarity < config.semantic_threshold {
                        continue;
                    }
                    let delta = subtract(other.position, splat.position);
                    let distance = length(delta).max(0.001);
                    if distance <= config.neighbor_radius * 4.0 {
                        let strength = config.attraction
                            * similarity
                            * (splat.mass * other.mass).sqrt()
                            * distance.min(config.neighbor_radius);
                        add_scaled(&mut force, delta, strength / distance);
                    }
                }
                let origin = cell(splat.position, config.spatial_cell_size);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(neighbors) =
                                cells.get(&(origin.0 + dx, origin.1 + dy, origin.2 + dz))
                            {
                                for &other_index in neighbors {
                                    if other_index == index {
                                        continue;
                                    }
                                    let other = &state.splats[other_index];
                                    let delta = subtract(other.position, splat.position);
                                    let distance = length(delta).max(0.001);
                                    if distance < config.repulsion_radius {
                                        let domain_multiplier = if splat.domain == other.domain {
                                            1.0
                                        } else {
                                            1.0 + config.cross_domain_repulsion
                                        };
                                        let strength = (config.repulsion_radius - distance)
                                            * config.repulsion
                                            * domain_multiplier;
                                        add_scaled(&mut force, delta, -strength / distance);
                                    }
                                }
                            }
                        }
                    }
                }
                force
            })
            .collect();

        energy = 0.0;
        let mut displacement = 0.0;
        for (splat, force) in state.splats.iter_mut().zip(forces) {
            let acceleration = [
                force[0] / splat.mass.max(0.1),
                force[1] / splat.mass.max(0.1),
                force[2] / splat.mass.max(0.1),
            ];
            for (axis, acceleration) in acceleration.iter().enumerate() {
                splat.velocity[axis] =
                    (splat.velocity[axis] + acceleration * config.dt) * config.damping;
                let delta = splat.velocity[axis] * config.dt;
                splat.position[axis] += delta;
                displacement += delta.abs();
            }
            energy += 0.5 * splat.mass * length(splat.velocity).powi(2);
        }
        steps_taken = step + 1;
        if displacement < 1e-5 {
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

fn spatial_cells(
    splats: &[crate::geometry::Splat],
    size: f32,
) -> HashMap<(i32, i32, i32), Vec<usize>> {
    let mut cells = HashMap::new();
    for (index, splat) in splats.iter().enumerate() {
        cells
            .entry(cell(splat.position, size))
            .or_insert_with(Vec::new)
            .push(index);
    }
    cells
}

fn cell(position: [f32; 3], size: f32) -> (i32, i32, i32) {
    let size = size.max(0.001);
    (
        (position[0] / size).floor() as i32,
        (position[1] / size).floor() as i32,
        (position[2] / size).floor() as i32,
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
