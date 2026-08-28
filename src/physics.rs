use crate::ann::KeyedAnn;
use crate::config::PhysicsConfig;
use crate::geometry::{HotState, Splat, cosine};
use crate::inversion::mass_from_radiance;
use crate::topology::{PersistenceInterval, discover_basins};
use anyhow::Result;
use chrono::Utc;
use rayon::prelude::*;
use std::collections::HashMap;

/// Floor on pair distance. Newtonian attraction diverges as separation approaches zero, so two
/// near-identical memories would otherwise produce an unbounded force and fling the field apart.
const MIN_SEPARATION: f32 = 0.05;

/// Per-splat movement in one step below which the field counts as settled.
const SETTLED_DISPLACEMENT_PER_SPLAT: f32 = 1e-6;

/// Above this splat count, dream uses ANN semantic neighbors instead of full O(N²) pairs.
/// Full pairwise force is correct but freezes the live GUI past ~10k (48k² per step).
/// Neighbor mode keeps the same force *law* on the local semantic graph so the field can
/// still flutter and consolidate at personal-store scale.
const NEIGHBOR_FORCE_AT: usize = 4_096;

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
    // Mute does not uncollapse the well those copies already packed. If one basin still
    // holds most of the live field, throw away that geometry and start from unmuted PCA.
    if state.live_field_is_soup() {
        let active = state.splats.iter().filter(|splat| !splat.muted).count();
        let largest = state
            .basins
            .iter()
            .map(|basin| basin.member_ids.len())
            .max()
            .unwrap_or(0);
        eprintln!(
            "dream: live field is soup (largest basin {largest} / {active} active) — reprojecting from unmuted PCA"
        );
        state.reproject();
    }
    let index_by_id: HashMap<_, _> = state
        .splats
        .iter()
        .enumerate()
        .map(|(index, splat)| (splat.memory_id, index))
        .collect();
    // Duplicate collapse (see `crate::dedupe`) marks byte-identical copies muted. They
    // stay in cold and in the ANN index but are out of the physics entirely.
    let state_muted: Vec<bool> = state.splats.iter().map(|s| s.muted).collect();
    let semantic_neighbors: Vec<Vec<usize>> = state
        .splats
        .par_iter()
        .map(|splat| {
            ann.neighbors_for_key(splat.memory_id)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|id| index_by_id.get(&id).copied())
                .filter(|&i| !state_muted[i])
                .take(32)
                .collect()
        })
        .collect();
    for (splat, neighbors) in state.splats.iter_mut().zip(&semantic_neighbors) {
        if splat.muted {
            continue;
        }
        splat.radiance = 1.0 + (neighbors.len() as f32 + 1.0).ln();
        // Magnitude from connectivity; sign is an independent physics knob.
        // Negative mass → repels. Negative gain is not consulted here (that inverts
        // semantics at steer time, already baked into splat.semantics).
        splat.mass = mass_from_radiance(splat.mass, splat.radiance);
        // Basin-first colors so personal fields (mostly one domain) stay readable.
        splat.color_rgba = domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0);
    }

    let n_splats = state.splats.len();
    // "Per splat" means per splat that can actually move; muted copies would otherwise
    // inflate the budget and stop the dream early.
    let neighbor_mode = n_splats > NEIGHBOR_FORCE_AT;
    if neighbor_mode {
        eprintln!(
            "dream: neighbor force mode (n={n_splats} > {NEIGHBOR_FORCE_AT}) — ANN attract + spatial repel"
        );
    }
    let (steps_taken, energy) = run_force_steps(
        state,
        &semantic_neighbors,
        config,
        neighbor_mode,
        emit_every,
        &mut on_step,
    );

    let previous = state.basins.clone();
    let (basins, persistence_intervals) = discover_basins(
        &mut state.splats,
        &previous,
        config,
        Some(ann),
        Some(&semantic_neighbors),
    );
    state.basins = basins;
    {
        let counts: HashMap<String, usize> = state
            .basins
            .iter()
            .map(|basin| (basin.id.clone(), basin.member_ids.len()))
            .collect();
        for splat in &mut state.splats {
            if splat.muted {
                continue;
            }
            let n = splat
                .basin_id
                .as_ref()
                .and_then(|id| counts.get(id).copied())
                .unwrap_or(1)
                .max(1);
            let major = 0.85 + (n as f32 + 1.0).ln() * 0.22;
            splat.scale = [major, major * 0.42, major * 0.42];
        }
    }
    // Re-tint after basin assignment so basin hue lands on settled membership.
    for splat in &mut state.splats {
        splat.color_rgba = domain_state_color(&splat.domain, splat.basin_id.as_deref(), 0.0);
    }
    state.dream_cycle += 1;
    state.last_dream_at = Some(Utc::now());
    // A 0-step rediscover must not clobber KE with the unset INF sentinel
    // (serde then writes `null` and the next load can fail).
    if steps_taken > 0 {
        state.kinetic_energy = energy;
    }
    on_step(steps_taken, energy, state);
    Ok(DreamReport {
        steps: steps_taken,
        kinetic_energy: energy,
        basins: state.basins.len(),
        persistence_intervals,
    })
}

fn want_cuda(config: &PhysicsConfig) -> bool {
    !config.backend.eq_ignore_ascii_case("cpu")
}

fn run_force_steps<F>(
    state: &mut HotState,
    semantic_neighbors: &[Vec<usize>],
    config: &PhysicsConfig,
    neighbor_mode: bool,
    emit_every: usize,
    on_step: &mut F,
) -> (usize, f32)
where
    F: FnMut(usize, f32, &HotState),
{
    if want_cuda(config) {
        match crate::cuda_dream::run_force_loop(
            state,
            semantic_neighbors,
            config,
            neighbor_mode,
            emit_every,
            on_step,
        ) {
            Ok(result) => return (result.steps, result.energy),
            Err(error) => {
                if config.backend.eq_ignore_ascii_case("cuda") {
                    eprintln!("dream: CUDA failed ({error:#}); falling back to CPU");
                } else {
                    eprintln!("dream: CUDA skipped ({error:#}); CPU force loop");
                }
            }
        }
    }
    cpu_force_loop(
        state,
        semantic_neighbors,
        config,
        neighbor_mode,
        emit_every,
        on_step,
    )
}

/// Force on one splat for the current step: origin pull plus the pair law against its
/// neighbors. Neighbor mode attracts through the ANN semantic graph and bounces off the
/// spatial-hash halo; full mode runs the original O(N²) pair law against everything.
fn splat_force(
    state: &HotState,
    semantic_neighbors: &[Vec<usize>],
    spatial: &[Vec<usize>],
    config: &PhysicsConfig,
    neighbor_mode: bool,
    index: usize,
) -> [f32; 3] {
    let splat = &state.splats[index];
    if splat.muted {
        return [0.0, 0.0, 0.0];
    }
    let mut force = [
        -splat.position[0] * config.origin_pull,
        -splat.position[1] * config.origin_pull,
        -splat.position[2] * config.origin_pull,
    ];
    let apply_pair = |force: &mut [f32; 3], other: &Splat, allow_attract: bool| {
        if other.muted {
            return;
        }
        let delta = subtract(other.position, splat.position);
        let distance = length(delta).max(MIN_SEPARATION);

        // Semantics set the SIGN of the pair force, distance only its magnitude.
        // A gravity-style law (G*m*m/d^2) is dominated by 1/d^2 and erases meaning
        // (measured: semantic correlation -0.85 → 0 over 300 steps). Sign-from-
        // semantics keeps structure.
        let mut weight = cosine(&splat.semantics, &other.semantics) - config.semantic_threshold;
        if splat.domain != other.domain {
            weight -= config.cross_domain_repulsion;
        }
        if splat.mass < 0.0 || other.mass < 0.0 {
            weight = weight.min(-0.05);
        }

        if weight > 0.0 {
            if allow_attract {
                add_scaled(force, delta, config.attraction * weight);
            }
        } else {
            let falloff = 1.0 + distance * distance;
            let strength = config.semantic_repulsion * (-weight) / falloff;
            add_scaled(force, delta, -strength / distance);
        }

        if distance < config.repulsion_radius {
            let strength = (config.repulsion_radius - distance) * config.repulsion;
            add_scaled(force, delta, -strength / distance);
        }
    };

    if neighbor_mode {
        for &other_index in &semantic_neighbors[index] {
            if other_index == index {
                continue;
            }
            apply_pair(&mut force, &state.splats[other_index], true);
        }
        for &other_index in &spatial[index] {
            if other_index == index {
                continue;
            }
            if semantic_neighbors[index]
                .iter()
                .any(|&semantic| semantic == other_index)
            {
                continue;
            }
            // Spatial halo: bounce only. Attraction through the grid would
            // re-glue the soup (moderate cosine + packed 3-d).
            apply_pair(&mut force, &state.splats[other_index], false);
        }
    } else {
        for (other_index, other) in state.splats.iter().enumerate() {
            if other_index == index {
                continue;
            }
            apply_pair(&mut force, other, true);
        }
    }
    force
}

/// One semi-implicit Euler pass over the field. Inertial mass is |m|; the sign already
/// lived in the force. Returns (total displacement, kinetic energy) for the step.
fn integrate_step(
    splats: &mut [Splat],
    forces: Vec<[f32; 3]>,
    config: &PhysicsConfig,
) -> (f32, f32) {
    let mut energy = 0.0;
    let mut displacement = 0.0;
    for (splat, force) in splats.iter_mut().zip(forces) {
        if splat.muted {
            continue;
        }
        // Inertial mass is |m|; sign already lived in the force (bollard rule above).
        let inertia = splat.mass.abs().max(0.1);
        let acceleration = [force[0] / inertia, force[1] / inertia, force[2] / inertia];
        for (axis, acceleration) in acceleration.iter().enumerate() {
            splat.velocity[axis] =
                (splat.velocity[axis] + acceleration * config.dt) * config.damping;
            let delta = splat.velocity[axis] * config.dt;
            splat.position[axis] += delta;
            displacement += delta.abs();
        }
        energy += 0.5 * inertia * length(splat.velocity).powi(2);
    }
    (displacement, energy)
}

/// Speed-tint: faster stars glow hotter so motion is visible in the live feed.
fn tint_by_speed(splats: &mut [Splat]) {
    for splat in splats {
        let speed = length(splat.velocity);
        splat.color_rgba = domain_state_color(&splat.domain, splat.basin_id.as_deref(), speed);
    }
}

fn cpu_force_loop<F>(
    state: &mut HotState,
    semantic_neighbors: &[Vec<usize>],
    config: &PhysicsConfig,
    neighbor_mode: bool,
    emit_every: usize,
    on_step: &mut F,
) -> (usize, f32)
where
    F: FnMut(usize, f32, &HotState),
{
    let n_splats = state.splats.len();
    let active_splats = state.splats.iter().filter(|s| !s.muted).count().max(1);
    let mut energy = 0.0f32;
    let mut steps_taken = 0usize;
    for _ in 0..config.steps {
        // Small stores: full O(N²) pairs (original consolidate path).
        // Large stores: ANN semantic neighbors for attraction (and their pair law), plus a
        // spatial-hash halo for repulsion. ANN-only was wrong at personal scale: the 32
        // nearest are almost all similar, so dissimilar memories that share a 3-d cell
        // never bounce, origin-pull packs the cloud, and basin discovery finds one component.
        // `neighbor_radius` is that spatial halo (config had it; physics never read it).
        let spatial = if neighbor_mode {
            spatial_neighbors(&state.splats, config.neighbor_radius)
        } else {
            Vec::new()
        };
        let forces: Vec<[f32; 3]> = (0..n_splats)
            .into_par_iter()
            .map(|index| {
                splat_force(
                    state,
                    semantic_neighbors,
                    &spatial,
                    config,
                    neighbor_mode,
                    index,
                )
            })
            .collect();

        let (displacement, step_energy) = integrate_step(&mut state.splats, forces, config);
        energy = step_energy;
        steps_taken += 1;
        if emit_every > 0 && steps_taken % emit_every == 0 {
            tint_by_speed(&mut state.splats);
            on_step(steps_taken, energy, state);
        }
        // Settle on total movement rather than a fixed step count: a field still collapsing keeps
        // running, and one that has already found its shape stops instead of burning the budget.
        // The threshold scales with the corpus so it means "per splat", not "in total".
        if displacement < SETTLED_DISPLACEMENT_PER_SPLAT * active_splats as f32 {
            break;
        }
    }
    (steps_taken, energy)
}

/// Color the field for *personal* viewing.
///
/// Basin membership is the primary signal (each basin gets a stable, widely spaced hue).
/// Domain is no longer a color axis — scifact/urban/poison were eval/test corpora and do not
/// belong in a personal chat store. Speed still brightens during live dream motion.
pub fn domain_state_color(_domain: &str, basin_id: Option<&str>, speed: f32) -> [u8; 4] {
    let (hue, sat, base_v) = match basin_id {
        Some(id) => {
            let mut h = 0u32;
            for b in id.bytes() {
                h = h.wrapping_mul(31).wrapping_add(b as u32);
            }
            // Golden-angle spacing so neighboring basin ids still land far apart on the wheel.
            let hue = (h as f32 * 137.508) % 360.0;
            // Keep saturation high enough that clusters read at a glance, not pastel soup.
            let sat = 0.62 + ((h >> 8) % 20) as f32 * 0.01;
            (hue, sat.clamp(0.55, 0.82), 0.62)
        }
        // Unassigned / not yet basined — dim cool slate, not competing with real clusters.
        None => (210.0, 0.18, 0.42),
    };
    let speed_boost = (speed * 8.0).clamp(0.0, 0.45);
    let value = (base_v + speed_boost).clamp(0.35, 1.0);
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

/// Spatial-hash neighbors within `radius`. Muted copies are skipped so a collapsed
/// duplicate clump cannot seed cells.
fn spatial_neighbors(splats: &[Splat], radius: f32) -> Vec<Vec<usize>> {
    let n = splats.len();
    let mut adj = vec![Vec::new(); n];
    if radius <= 0.0 {
        return adj;
    }
    let cell_size = radius.max(0.001);
    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (index, splat) in splats.iter().enumerate() {
        if splat.muted {
            continue;
        }
        cells
            .entry(cell(splat.position, cell_size))
            .or_default()
            .push(index);
    }
    for (index, splat) in splats.iter().enumerate() {
        if splat.muted {
            continue;
        }
        let origin = cell(splat.position, cell_size);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(neighbors) = cells.get(&(origin.0 + dx, origin.1 + dy, origin.2 + dz))
                    else {
                        continue;
                    };
                    for &other in neighbors {
                        if other <= index {
                            continue;
                        }
                        if length(subtract(splat.position, splats[other].position)) <= radius {
                            adj[index].push(other);
                            adj[other].push(index);
                        }
                    }
                }
            }
        }
    }
    adj
}

fn cell(position: [f32; 3], size: f32) -> (i32, i32, i32) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PhysicsConfig;
    use crate::geometry::HotState;
    use uuid::Uuid;

    fn splat(id: u128, position: [f32; 3], muted: bool) -> Splat {
        Splat {
            memory_id: Uuid::from_u128(id),
            semantics: vec![0.125; 64],
            position,
            velocity: [0.0; 3],
            scale: [1.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [255; 4],
            mass: 1.0,
            radiance: 1.0,
            usage: 0.0,
            conductivity: 1.0,
            gain: 0.0,
            basin_locked: false,
            muted,
            duplicate_count: 1,
            domain: "chat".into(),
            basin_id: None,
            lineage: vec![],
        }
    }

    #[test]
    fn spatial_neighbors_link_nearby_and_skip_muted() {
        let splats = vec![
            splat(1, [0.0, 0.0, 0.0], false),
            splat(2, [0.1, 0.0, 0.0], false),
            splat(3, [5.0, 0.0, 0.0], false),
            splat(4, [0.05, 0.0, 0.0], true),
        ];
        let adj = spatial_neighbors(&splats, 0.5);
        assert!(adj[0].contains(&1));
        assert!(adj[1].contains(&0));
        assert!(!adj[0].contains(&2));
        assert!(adj[3].is_empty(), "muted must not seed or receive edges");
        assert!(!adj[0].contains(&3));
    }

    fn toy_state() -> HotState {
        let mut splats = Vec::new();
        for i in 0..8u128 {
            let mut s = splat(i + 1, [i as f32 * 0.4, (i % 3) as f32 * 0.3, 0.1], false);
            s.semantics = (0..64)
                .map(|k| ((i as f32 + 1.0) * 0.02 + k as f32 * 0.001).sin())
                .collect();
            s.domain = if i % 2 == 0 {
                "chat".into()
            } else {
                "note".into()
            };
            s.mass = 1.0 + (i as f32) * 0.05;
            s.velocity = [0.01 * i as f32, 0.0, 0.0];
            splats.push(s);
        }
        HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats,
            basins: Vec::new(),
            projection: None,
        }
    }

    fn max_pos_delta(a: &HotState, b: &HotState) -> f32 {
        a.splats
            .iter()
            .zip(&b.splats)
            .map(|(x, y)| {
                (0..3)
                    .map(|k| (x.position[k] - y.position[k]).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max)
    }

    fn compare_backends(steps: usize, neighbor_mode: bool, tol: f32) {
        if !crate::cuda_dream::available() {
            eprintln!("skip cuda_matches_cpu: no CUDA device");
            return;
        }
        let mut cpu = toy_state();
        let mut gpu = cpu.clone();
        let nbrs = vec![Vec::new(); cpu.splats.len()];
        let mut cfg = PhysicsConfig::default();
        cfg.steps = steps;
        cfg.backend = "cpu".into();
        let (cpu_steps, _) =
            cpu_force_loop(&mut cpu, &nbrs, &cfg, neighbor_mode, 0, &mut |_, _, _| {});
        cfg.backend = "cuda".into();
        let gpu_result = crate::cuda_dream::run_force_loop(
            &mut gpu,
            &nbrs,
            &cfg,
            neighbor_mode,
            0,
            &mut |_, _, _| {},
        )
        .expect("CUDA force loop");
        let gpu_steps = gpu_result.steps;
        assert_eq!(cpu_steps, gpu_steps);
        let delta = max_pos_delta(&cpu, &gpu);
        assert!(
            delta < tol,
            "CPU vs CUDA max |Δpos|={delta} after {steps} steps neighbor_mode={neighbor_mode}"
        );
    }

    #[test]
    fn cuda_matches_cpu_full_pair_1_10_50() {
        compare_backends(1, false, 2e-4);
        compare_backends(10, false, 5e-4);
        compare_backends(50, false, 2e-3);
    }

    #[test]
    fn cuda_matches_cpu_neighbor_mode_10() {
        compare_backends(10, true, 5e-3);
    }
}
