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
) -> (Vec<Basin>, Vec<PersistenceInterval>) {
    let edges = spatial_semantic_edges(splats, config.basin_radius, config.semantic_threshold);
    let mut union = UnionFind::new(splats.len());
    let mut intervals = Vec::new();
    for edge in &edges {
        let left_root = union.find(edge.left);
        let right_root = union.find(edge.right);
        if left_root == right_root {
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
        for chunk in split_overloaded(&members, splats, 5_000) {
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

    basins.sort_by(|a, b| a.id.cmp(&b.id));
    (basins, intervals)
}

#[derive(Debug, Clone)]
struct Edge {
    left: usize,
    right: usize,
    distance: f32,
}

fn spatial_semantic_edges(splats: &[Splat], radius: f32, semantic_threshold: f32) -> Vec<Edge> {
    let cell_size = radius.max(0.001);
    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (index, splat) in splats.iter().enumerate() {
        cells
            .entry(cell(splat.position, cell_size))
            .or_default()
            .push(index);
    }
    let mut edges = Vec::new();
    for (index, splat) in splats.iter().enumerate() {
        let origin = cell(splat.position, cell_size);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(neighbors) =
                        cells.get(&(origin.0 + dx, origin.1 + dy, origin.2 + dz))
                    {
                        for &other in neighbors {
                            if other <= index {
                                continue;
                            }
                            let distance = distance(splat.position, splats[other].position);
                            if distance <= radius
                                && cosine(&splat.semantics, &splats[other].semantics)
                                    >= semantic_threshold
                            {
                                edges.push(Edge {
                                    left: index,
                                    right: other,
                                    distance,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    edges.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| splats[a.left].memory_id.cmp(&splats[b.left].memory_id))
            .then_with(|| splats[a.right].memory_id.cmp(&splats[b.right].memory_id))
    });
    edges
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

fn cell(position: [f32; 3], size: f32) -> (i32, i32, i32) {
    (
        (position[0] / size).floor() as i32,
        (position[1] / size).floor() as i32,
        (position[2] / size).floor() as i32,
    )
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

    fn splat(id: u128, x: f32) -> Splat {
        Splat {
            memory_id: Uuid::from_u128(id),
            semantics: vec![0.125; 64],
            position: [x, 0.0, 0.0],
            velocity: [0.0; 3],
            scale: [1.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [255; 4],
            mass: 1.0,
            radiance: 1.0,
            gain: 0.0,
            basin_locked: false,
            domain: "chat".into(),
            basin_id: None,
            lineage: vec![],
        }
    }

    #[test]
    fn h0_components_separate_distant_groups() {
        let mut splats = vec![
            splat(1, 0.0),
            splat(2, 0.1),
            splat(3, 0.2),
            splat(4, 5.0),
            splat(5, 5.1),
            splat(6, 5.2),
        ];
        let config = PhysicsConfig {
            basin_radius: 0.5,
            ..PhysicsConfig::default()
        };
        let (basins, intervals) = discover_basins(&mut splats, &[], &config);
        assert_eq!(basins.len(), 2);
        assert_eq!(intervals.len(), 4);
    }
}
