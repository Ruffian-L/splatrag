use crate::ann::KeyedAnn;
use crate::config::PhysicsConfig;
use crate::geometry::{Basin, Splat, cosine};
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

const BASIN_NAMESPACE: Uuid = Uuid::from_u128(0x9b51_03af_dca2_4f97_a628_c1e4_7720_6b3d);

#[derive(Debug, Clone)]
pub struct PersistenceInterval {
    pub birth: f32,
    pub death: f32,
    pub survivor: Uuid,
    pub absorbed: Uuid,
}

pub fn discover_basins(
    splats: &mut [Splat],
    previous: &[Basin],
    config: &PhysicsConfig,
    ann: Option<&KeyedAnn>,
    semantic_neighbors: Option<&[Vec<usize>]>,
) -> (Vec<Basin>, Vec<PersistenceInterval>) {
    // Pack to 64d geometry. 3D radius percolation was gluing the PCA picture into one well.
    let edges = hidden_knn_edges(splats, config, ann, semantic_neighbors);
    let mut union = UnionFind::new(splats.len());
    let mut intervals = Vec::new();
    let min_size = config.min_basin_size.max(1);
    let core = config.basin_core_cosine.clamp(0.0, 1.0);
    for edge in &edges {
        let left_root = union.find(edge.left);
        let right_root = union.find(edge.right);
        if left_root == right_root {
            continue;
        }
        let left_size = union.size[left_root];
        let right_size = union.size[right_root];
        let cosine = (1.0 - edge.distance).clamp(0.0, 1.0);
        let crumb = left_size < min_size || right_size < min_size;
        // Two real wells stay two wells unless this edge is a tight 64d core, not a 0.55 bridge.
        if !crumb && cosine < core {
            continue;
        }
        let (survivor, absorbed) = union.union(left_root, right_root);
        intervals.push(PersistenceInterval {
            birth: 0.0,
            death: edge.distance,
            survivor: splats[survivor].memory_id,
            absorbed: splats[absorbed].memory_id,
        });
    }

    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for index in 0..splats.len() {
        let root = union.find(index);
        components.entry(root).or_default().push(index);
    }
    let previous_members: Vec<HashSet<Uuid>> = previous
        .iter()
        .map(|basin| basin.member_ids.iter().copied().collect())
        .collect();
    let mut claimed_previous = HashSet::new();
    let mut basins = Vec::new();
    // Locked members keep their basin_id even if the free graph would reassign them.
    // Collect them first so free components can still form around unlocked points.
    let locked: Vec<(usize, String)> = splats
        .iter()
        .enumerate()
        .filter_map(|(index, splat)| {
            if splat.basin_locked {
                splat.basin_id.clone().map(|id| (index, id))
            } else {
                None
            }
        })
        .collect();

    for mut members in components.into_values() {
        if members.len() < config.min_basin_size {
            for index in members {
                if !splats[index].basin_locked {
                    splats[index].basin_id = None;
                }
            }
            continue;
        }
        members.sort_by_key(|index| splats[*index].memory_id);
        // Cap is bookkeeping (split_overloaded), not a meaning unit — see research log 2026-08-07.
        // `basin_max_members = 0` means no axis-split bags: keep the geometric component intact.
        let max_members = if config.basin_max_members == 0 {
            usize::MAX
        } else {
            config.basin_max_members.max(config.min_basin_size)
        };
        for chunk in split_overloaded(&members, splats, max_members) {
            let ids: Vec<_> = chunk.iter().map(|index| splats[*index].memory_id).collect();
            let id_set: HashSet<_> = ids.iter().copied().collect();
            let mut best_previous = None;
            let mut best_overlap = 0.0_f32;
            for (index, old) in previous_members.iter().enumerate() {
                if claimed_previous.contains(&index) {
                    continue;
                }
                let intersection = id_set.intersection(old).count();
                let union_size = id_set.union(old).count().max(1);
                let overlap = intersection as f32 / union_size as f32;
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_previous = Some(index);
                }
            }
            let reused = best_previous.filter(|_| best_overlap >= 0.5);
            if let Some(index) = reused {
                claimed_previous.insert(index);
            }
            let basin_id = reused
                .map(|index| previous[index].id.clone())
                .unwrap_or_else(|| stable_basin_id(&ids));
            for index in &chunk {
                if !splats[*index].basin_locked {
                    splats[*index].basin_id = Some(basin_id.clone());
                }
            }
            let centroid = centroid(&chunk, splats);
            let spread = chunk
                .iter()
                .map(|index| distance(splats[*index].position, centroid))
                .sum::<f32>()
                / chunk.len() as f32;
            let stability = (1.0 - spread / (config.basin_radius * 3.0)).clamp(0.0, 1.0);
            let mut representatives = chunk.clone();
            representatives.sort_by(|left, right| {
                splats[*right]
                    .radiance
                    .total_cmp(&splats[*left].radiance)
                    .then_with(|| splats[*left].memory_id.cmp(&splats[*right].memory_id))
            });
            let representative_ids = representatives
                .into_iter()
                .take(12)
                .map(|index| splats[index].memory_id)
                .collect();
            let old = reused.map(|index| &previous[index]);
            basins.push(Basin {
                id: basin_id.clone(),
                parent_id: old.and_then(|basin| basin.parent_id.clone()),
                label: old
                    .map(|basin| basin.label.clone())
                    .unwrap_or_else(|| format!("unlabeled-{}", &basin_id[..8])),
                path: old
                    .map(|basin| basin.path.clone())
                    .unwrap_or_else(|| "memory/unlabeled".into()),
                summary: old.map(|basin| basin.summary.clone()).unwrap_or_default(),
                // A reused basin keeps whatever state it had. Promoting to `stable` on reuse alone
                // marked basins stable that were never successfully labeled — labeling only visits
                // `pending` basins, so a placeholder `unlabeled-<id>` name became permanent after
                // one dream cycle, and any labeling failure was unrecoverable.
                label_state: old
                    .map(|basin| basin.label_state.clone())
                    .unwrap_or_else(|| "pending".into()),
                stability,
                centroid,
                member_ids: ids,
                representative_ids,
                updated_at: Utc::now(),
            });
        }
    }
    // Re-attach locked members to their declared basins (create a shell basin if needed).
    for (index, basin_id) in &locked {
        splats[*index].basin_id = Some(basin_id.clone());
        if let Some(basin) = basins.iter_mut().find(|b| b.id == *basin_id) {
            if !basin.member_ids.contains(&splats[*index].memory_id) {
                basin.member_ids.push(splats[*index].memory_id);
            }
        } else {
            basins.push(Basin {
                id: basin_id.clone(),
                parent_id: None,
                label: format!("locked-{}", &basin_id[..basin_id.len().min(8)]),
                path: "memory/locked".into(),
                summary: String::new(),
                label_state: "locked".into(),
                stability: 1.0,
                centroid: splats[*index].position,
                member_ids: vec![splats[*index].memory_id],
                representative_ids: vec![splats[*index].memory_id],
                updated_at: Utc::now(),
            });
        }
    }

    attach_crumbs(splats, &mut basins, config, ann, semantic_neighbors);

    basins.sort_by(|a, b| a.id.cmp(&b.id));
    (basins, intervals)
}

/// Gaussian falloff: a crumb joins the nearest existing well. Never unions two wells.
fn attach_crumbs(
    splats: &mut [Splat],
    basins: &mut [Basin],
    config: &PhysicsConfig,
    ann: Option<&KeyedAnn>,
    semantic_neighbors: Option<&[Vec<usize>]>,
) {
    let attach = config.basin_attach_cosine;
    if attach <= 0.0 || basins.is_empty() {
        return;
    }
    let mut basin_index: HashMap<String, usize> = HashMap::new();
    for (index, basin) in basins.iter().enumerate() {
        basin_index.insert(basin.id.clone(), index);
    }
    let id_to_index: HashMap<Uuid, usize> = splats
        .iter()
        .enumerate()
        .map(|(index, splat)| (splat.memory_id, index))
        .collect();
    let assigned: Vec<usize> = splats
        .iter()
        .enumerate()
        .filter(|(_, splat)| !splat.muted && splat.basin_id.is_some())
        .map(|(index, _)| index)
        .collect();
    if assigned.is_empty() {
        return;
    }

    let mut grown: HashSet<usize> = HashSet::new();
    for index in 0..splats.len() {
        if splats[index].muted || splats[index].basin_locked || splats[index].basin_id.is_some() {
            continue;
        }
        if let Some(basin_id) = best_basin_for(
            splats,
            index,
            config,
            ann,
            semantic_neighbors,
            &id_to_index,
            &assigned,
        ) {
            assign_to_basin(splats, basins, &basin_index, &mut grown, index, &basin_id);
        }
    }

    for basin_i in grown {
        refresh_basin(&mut basins[basin_i], splats, config);
    }
}

/// Strongest well for crumb `index` over its candidate neighbors: the ANN graph when one
/// was passed, else every already-assigned splat. Returns the basin id of the best cosine
/// at or above the configured attach threshold, if any.
fn best_basin_for(
    splats: &[Splat],
    index: usize,
    config: &PhysicsConfig,
    ann: Option<&KeyedAnn>,
    semantic_neighbors: Option<&[Vec<usize>]>,
    id_to_index: &HashMap<Uuid, usize>,
    assigned: &[usize],
) -> Option<String> {
    let attach = config.basin_attach_cosine;
    let mut best_cosine = f32::NEG_INFINITY;
    let mut best_basin: Option<String> = None;

    let mut consider = |other: usize| {
        if other == index {
            return;
        }
        let Some(basin_id) = splats[other].basin_id.clone() else {
            return;
        };
        let cosine_value = cosine(&splats[index].semantics, &splats[other].semantics);
        if cosine_value >= attach && cosine_value > best_cosine {
            best_cosine = cosine_value;
            best_basin = Some(basin_id);
        }
    };

    let requested = config.basin_knn.max(8).saturating_mul(4);
    // The dream force table intentionally stores at most 32 neighbors. Fall back to
    // ANN for wider basin configurations rather than silently truncating topology.
    if requested <= crate::cuda_dream::MAX_SEM_NBR
        && let Some(neighbors) = semantic_neighbors.and_then(|all| all.get(index))
    {
        for &other in neighbors.iter().take(requested) {
            consider(other);
        }
    } else if let Some(ann) = ann {
        let neighbors = ann
            .neighbors_for_key(splats[index].memory_id)
            .unwrap_or_default();
        for id in neighbors.into_iter().take(requested) {
            if let Some(&other) = id_to_index.get(&id) {
                consider(other);
            }
        }
    } else {
        for &other in assigned {
            consider(other);
        }
    }

    best_basin
}

/// Attach crumb `index` to `basin_id` and grow the basin's member list when the crumb is
/// new to it. The splat keeps the assignment even if the basin id has no shell basin.
fn assign_to_basin(
    splats: &mut [Splat],
    basins: &mut [Basin],
    basin_index: &HashMap<String, usize>,
    grown: &mut HashSet<usize>,
    index: usize,
    basin_id: &str,
) {
    splats[index].basin_id = Some(basin_id.to_string());
    if let Some(&basin_i) = basin_index.get(basin_id) {
        let id = splats[index].memory_id;
        if !basins[basin_i].member_ids.contains(&id) {
            basins[basin_i].member_ids.push(id);
            grown.insert(basin_i);
        }
    }
}

fn refresh_basin(basin: &mut Basin, splats: &[Splat], config: &PhysicsConfig) {
    let id_to_index: HashMap<Uuid, usize> = splats
        .iter()
        .enumerate()
        .map(|(index, splat)| (splat.memory_id, index))
        .collect();
    let members: Vec<usize> = basin
        .member_ids
        .iter()
        .filter_map(|id| id_to_index.get(id).copied())
        .collect();
    if members.is_empty() {
        return;
    }
    basin.centroid = centroid(&members, splats);
    let spread = members
        .iter()
        .map(|index| distance(splats[*index].position, basin.centroid))
        .sum::<f32>()
        / members.len() as f32;
    basin.stability = (1.0 - spread / (config.basin_radius * 3.0)).clamp(0.0, 1.0);
    let mut representatives = members;
    representatives.sort_by(|left, right| {
        splats[*right]
            .radiance
            .total_cmp(&splats[*left].radiance)
            .then_with(|| splats[*left].memory_id.cmp(&splats[*right].memory_id))
    });
    basin.representative_ids = representatives
        .into_iter()
        .take(12)
        .map(|index| splats[index].memory_id)
        .collect();
    basin.updated_at = Utc::now();
}

#[derive(Debug, Clone)]
struct Edge {
    left: usize,
    right: usize,
    distance: f32,
}

struct EdgeCollector {
    threshold: f32,
    seen: HashSet<(usize, usize)>,
    edges: Vec<Edge>,
}

impl EdgeCollector {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            seen: HashSet::new(),
            edges: Vec::new(),
        }
    }

    fn add(&mut self, left: usize, right: usize, cosine_value: f32) {
        if left == right || cosine_value < self.threshold {
            return;
        }
        let (left, right) = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        if self.seen.insert((left, right)) {
            self.edges.push(Edge {
                left,
                right,
                distance: (1.0 - cosine_value).clamp(0.0, 2.0),
            });
        }
    }
}

/// 64d k-NN edges. `distance` is `1 - cosine` so Kruskal packs tight wells first.
fn hidden_knn_edges(
    splats: &[Splat],
    config: &PhysicsConfig,
    ann: Option<&KeyedAnn>,
    semantic_neighbors: Option<&[Vec<usize>]>,
) -> Vec<Edge> {
    // 0.55 is the dream force-neutral, not a basin edge. On Qwen3 SciFact, random
    // paper–paper cosine p90 is ~0.55, so that gate percolates the whole science
    // manifold. Pack wells at `basin_core_cosine` (the tight 64d neighborhood).
    let threshold = config.basin_core_cosine.max(config.semantic_threshold);
    let live: Vec<usize> = splats
        .iter()
        .enumerate()
        .filter(|(_, splat)| !splat.muted)
        .map(|(index, _)| index)
        .collect();
    if live.len() < 2 {
        return Vec::new();
    }
    let k = if config.basin_knn == 0 {
        live.len().saturating_sub(1)
    } else {
        config.basin_knn.max(1).min(live.len() - 1)
    };
    let mut collector = EdgeCollector::new(threshold);

    if k <= crate::cuda_dream::MAX_SEM_NBR
        && let Some(neighbors_by_index) = semantic_neighbors
    {
        collect_precomputed_edges(splats, &live, neighbors_by_index, k, &mut collector);
    } else if let Some(ann) = ann {
        collect_ann_edges(splats, &live, ann, k, &mut collector);
    } else {
        collect_brute_force_edges(splats, &live, k, &mut collector);
    }

    let mut edges = collector.edges;
    edges.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| splats[a.left].memory_id.cmp(&splats[b.left].memory_id))
            .then_with(|| splats[a.right].memory_id.cmp(&splats[b.right].memory_id))
    });
    edges
}

fn collect_precomputed_edges(
    splats: &[Splat],
    live: &[usize],
    neighbors_by_index: &[Vec<usize>],
    k: usize,
    collector: &mut EdgeCollector,
) {
    for &index in live {
        let Some(neighbors) = neighbors_by_index.get(index) else {
            continue;
        };
        for &other in neighbors.iter().take(k) {
            if !splats[other].muted {
                collector.add(
                    index,
                    other,
                    cosine(&splats[index].semantics, &splats[other].semantics),
                );
            }
        }
    }
}

fn collect_ann_edges(
    splats: &[Splat],
    live: &[usize],
    ann: &KeyedAnn,
    k: usize,
    collector: &mut EdgeCollector,
) {
    let id_to_index: HashMap<Uuid, usize> = splats
        .iter()
        .enumerate()
        .map(|(index, splat)| (splat.memory_id, index))
        .collect();
    for &index in live {
        let neighbors = ann
            .neighbors_for_key(splats[index].memory_id)
            .unwrap_or_default();
        for id in neighbors.into_iter().take(k) {
            let Some(&other) = id_to_index.get(&id) else {
                continue;
            };
            if !splats[other].muted {
                collector.add(
                    index,
                    other,
                    cosine(&splats[index].semantics, &splats[other].semantics),
                );
            }
        }
    }
}

fn collect_brute_force_edges(
    splats: &[Splat],
    live: &[usize],
    k: usize,
    collector: &mut EdgeCollector,
) {
    for (rank, &index) in live.iter().enumerate() {
        let mut scored: Vec<(f32, usize)> = live
            .iter()
            .enumerate()
            .filter(|(other_rank, _)| *other_rank != rank)
            .map(|(_, &other)| {
                (
                    cosine(&splats[index].semantics, &splats[other].semantics),
                    other,
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        for (cosine_value, other) in scored.into_iter().take(k) {
            collector.add(index, other, cosine_value);
        }
    }
}

fn split_overloaded(members: &[usize], splats: &[Splat], max_size: usize) -> Vec<Vec<usize>> {
    if members.len() <= max_size {
        return vec![members.to_vec()];
    }
    let mut ordered = members.to_vec();
    let axis = widest_axis(members, splats);
    ordered.sort_by(|left, right| {
        splats[*left].position[axis]
            .total_cmp(&splats[*right].position[axis])
            .then_with(|| splats[*left].memory_id.cmp(&splats[*right].memory_id))
    });
    ordered
        .chunks(max_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn widest_axis(members: &[usize], splats: &[Splat]) -> usize {
    let mut ranges = [0.0_f32; 3];
    for (axis, range) in ranges.iter_mut().enumerate() {
        let min = members
            .iter()
            .map(|index| splats[*index].position[axis])
            .fold(f32::INFINITY, f32::min);
        let max = members
            .iter()
            .map(|index| splats[*index].position[axis])
            .fold(f32::NEG_INFINITY, f32::max);
        *range = max - min;
    }
    (0..3)
        .max_by(|left, right| ranges[*left].total_cmp(&ranges[*right]))
        .unwrap_or(0)
}

fn stable_basin_id(ids: &[Uuid]) -> String {
    let identity = ids
        .iter()
        .map(Uuid::to_string)
        .collect::<Vec<_>>()
        .join("\0");
    Uuid::new_v5(&BASIN_NAMESPACE, identity.as_bytes()).to_string()
}

fn centroid(indices: &[usize], splats: &[Splat]) -> [f32; 3] {
    let total_mass = indices
        .iter()
        .map(|index| splats[*index].mass.max(0.001))
        .sum::<f32>();
    let mut center = [0.0_f32; 3];
    for index in indices {
        let mass = splats[*index].mass.max(0.001);
        for (axis, value) in center.iter_mut().enumerate() {
            *value += splats[*index].position[axis] * mass / total_mass;
        }
    }
    center
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            size: vec![1; size],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            self.parent[index] = self.find(self.parent[index]);
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) -> (usize, usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return (left, right);
        }
        if self.size[left] < self.size[right]
            || (self.size[left] == self.size[right] && left > right)
        {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        self.size[left] += self.size[right];
        (left, right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn splat(id: u128, position: [f32; 3], axis: usize) -> Splat {
        let mut semantics = vec![0.0; 64];
        semantics[axis] = 1.0;
        Splat {
            memory_id: Uuid::from_u128(id),
            semantics,
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
            muted: false,
            duplicate_count: 1,
            domain: "chat".into(),
            basin_id: None,
            lineage: vec![],
        }
    }

    fn config() -> PhysicsConfig {
        PhysicsConfig {
            basin_knn: 8,
            basin_core_cosine: 0.75,
            min_basin_size: 3,
            semantic_threshold: 0.55,
            ..PhysicsConfig::default()
        }
    }

    #[test]
    fn hd_clusters_split_even_when_positions_mash() {
        // Two 64d modes sitting in the same 3D cell — the picture is soup, the hidden vectors are not.
        let mut splats = vec![
            splat(1, [0.0, 0.0, 0.0], 0),
            splat(2, [0.05, 0.0, 0.0], 0),
            splat(3, [0.1, 0.0, 0.0], 0),
            splat(4, [0.0, 0.0, 0.0], 1),
            splat(5, [0.05, 0.0, 0.0], 1),
            splat(6, [0.1, 0.0, 0.0], 1),
        ];
        let (basins, _) = discover_basins(&mut splats, &[], &config(), None, None);
        assert_eq!(
            basins.len(),
            2,
            "got {:?}",
            basins
                .iter()
                .map(|b| b.member_ids.len())
                .collect::<Vec<_>>()
        );
        let axes: Vec<_> = basins
            .iter()
            .map(|basin| {
                basin
                    .member_ids
                    .iter()
                    .filter_map(|id| splats.iter().find(|s| s.memory_id == *id))
                    .map(|s| s.semantics.iter().position(|v| *v > 0.5).unwrap())
                    .collect::<HashSet<_>>()
            })
            .collect();
        assert!(axes.iter().all(|set| set.len() == 1));
    }

    #[test]
    fn same_hidden_vector_packs_across_split_positions() {
        let mut splats = vec![
            splat(1, [0.0, 0.0, 0.0], 0),
            splat(2, [0.1, 0.0, 0.0], 0),
            splat(3, [0.2, 0.0, 0.0], 0),
            splat(4, [50.0, 0.0, 0.0], 0),
            splat(5, [50.1, 0.0, 0.0], 0),
            splat(6, [50.2, 0.0, 0.0], 0),
        ];
        let (basins, _) = discover_basins(&mut splats, &[], &config(), None, None);
        assert_eq!(basins.len(), 1);
        assert_eq!(basins[0].member_ids.len(), 6);
    }

    #[test]
    fn precomputed_neighbors_drive_basin_discovery_without_another_ann_walk() {
        let mut splats = vec![
            splat(1, [0.0; 3], 0),
            splat(2, [0.0; 3], 0),
            splat(3, [0.0; 3], 0),
            splat(4, [0.0; 3], 1),
            splat(5, [0.0; 3], 1),
            splat(6, [0.0; 3], 1),
        ];
        let neighbors = vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![4, 5],
            vec![3, 5],
            vec![3, 4],
        ];
        let (basins, _) = discover_basins(&mut splats, &[], &config(), None, Some(&neighbors));
        let mut sizes = basins
            .iter()
            .map(|basin| basin.member_ids.len())
            .collect::<Vec<_>>();
        sizes.sort();
        assert_eq!(sizes, vec![3, 3]);
    }

    #[test]
    fn weak_bridge_does_not_glue_two_cores() {
        let mut bridge = splat(7, [0.0, 0.0, 0.0], 0);
        bridge.semantics[0] = 0.7;
        bridge.semantics[1] = 0.7;
        let mut splats = vec![
            splat(1, [0.0; 3], 0),
            splat(2, [0.0; 3], 0),
            splat(3, [0.0; 3], 0),
            splat(4, [0.0; 3], 1),
            splat(5, [0.0; 3], 1),
            splat(6, [0.0; 3], 1),
            bridge,
        ];
        let (basins, _) = discover_basins(&mut splats, &[], &config(), None, None);
        assert_eq!(basins.len(), 2);
    }

    #[test]
    fn crumb_attaches_to_nearest_core_without_merging() {
        let mut crumb = splat(7, [0.0; 3], 0);
        crumb.semantics[0] = 0.73;
        crumb.semantics[1] = (1.0 - 0.73 * 0.73_f32).sqrt();
        let mut splats = vec![
            splat(1, [0.0; 3], 0),
            splat(2, [0.0; 3], 0),
            splat(3, [0.0; 3], 0),
            splat(4, [0.0; 3], 1),
            splat(5, [0.0; 3], 1),
            splat(6, [0.0; 3], 1),
            crumb,
        ];
        let (basins, _) = discover_basins(&mut splats, &[], &config(), None, None);
        assert_eq!(basins.len(), 2);
        let sizes: Vec<usize> = {
            let mut s: Vec<usize> = basins.iter().map(|b| b.member_ids.len()).collect();
            s.sort();
            s
        };
        assert_eq!(
            sizes,
            vec![3, 4],
            "crumb should join one well, not glue both: {sizes:?}"
        );
        assert_eq!(splats[6].basin_id, splats[0].basin_id);
    }
}
