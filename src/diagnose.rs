//! Diagnostic controls. They do not change ranking or clustering.
//!
//! John (2026-08-27): separate basin geometry (64-d vs 3-d connectivity) from
//! 64-d candidate generation vs 4096-d discrimination, before any architecture move.

use crate::geometry::{HotState, cosine};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

const MIN_BASIN: usize = 3;
const COS_THRESHOLDS: &[f32] = &[
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
];
const DIST_THRESHOLDS: &[f32] = &[
    0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00,
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentRow {
    pub threshold: f32,
    pub n_components: usize,
    pub lcc: usize,
    pub lcc_frac: f64,
    pub n_basins_ge_min: usize,
    pub singletons: usize,
    pub top10: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeMerge {
    pub cosine: f32,
    pub left_n: usize,
    pub right_n: usize,
    pub new_n: usize,
    pub left_domain: String,
    pub right_domain: String,
    pub cross_domain: bool,
    pub left_id: String,
    pub right_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasinCurveReport {
    pub n_live: usize,
    pub n_muted: usize,
    pub dream_cycle: u64,
    pub current_n_basins: usize,
    pub current_lcc: usize,
    pub current_lcc_frac: f64,
    pub curve_64d_cosine: Vec<ComponentRow>,
    pub curve_3d_cosine: Vec<ComponentRow>,
    pub curve_3d_euclidean: Vec<ComponentRow>,
    pub bridges_64d: Vec<BridgeMerge>,
    pub giant_25pct_birth: Option<BridgeMerge>,
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    /// Returns (left_size, right_size, new_root) before the merge. None if already joined.
    fn union(&mut self, a: usize, b: usize) -> Option<(usize, usize, usize)> {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return None;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        let sa = self.size[ra];
        let sb = self.size[rb];
        self.parent[rb] = ra;
        self.size[ra] = sa + sb;
        Some((sa, sb, ra))
    }

    fn summarize(&self) -> ComponentRow {
        let mut root_size = vec![0usize; self.parent.len()];
        for i in 0..self.parent.len() {
            let mut x = i;
            while self.parent[x] != x {
                x = self.parent[x];
            }
            root_size[x] += 1;
        }
        let mut sizes: Vec<usize> = root_size.into_iter().filter(|s| *s > 0).collect();
        sizes.sort_by(|a, b| b.cmp(a));
        let n = self.parent.len().max(1);
        let lcc = sizes.first().copied().unwrap_or(0);
        ComponentRow {
            threshold: 0.0,
            n_components: sizes.len(),
            lcc,
            lcc_frac: lcc as f64 / n as f64,
            n_basins_ge_min: sizes.iter().filter(|s| **s >= MIN_BASIN).count(),
            singletons: sizes.iter().filter(|s| **s == 1).count(),
            top10: sizes.into_iter().take(10).collect(),
        }
    }
}

fn live_indices(hot: &HotState) -> Vec<usize> {
    hot.splats
        .iter()
        .enumerate()
        .filter(|(_, splat)| !splat.muted)
        .map(|(i, _)| i)
        .collect()
}

fn components_for_predicate<F>(n: usize, pred: F) -> ComponentRow
where
    F: Fn(usize, usize) -> bool,
{
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if pred(i, j) {
                uf.union(i, j);
            }
        }
    }
    uf.summarize()
}

/// Threshold → largest connected component in 64-d cosine and in the 3-d picture.
/// Also walks 64-d Kruskal to list well-to-well bridge merges.
pub fn basin_component_curve(hot: &HotState) -> BasinCurveReport {
    let live = live_indices(hot);
    let n = live.len();
    let mut current_sizes: Vec<usize> = hot.basins.iter().map(|b| b.member_ids.len()).collect();
    current_sizes.sort_by(|a, b| b.cmp(a));
    let current_lcc = current_sizes.first().copied().unwrap_or(0);

    let mut cos64 = vec![0.0f32; n.saturating_mul(n)];
    let mut cos3 = vec![0.0f32; n.saturating_mul(n)];
    let mut dist = vec![0.0f32; n.saturating_mul(n)];
    for i in 0..n {
        let si = &hot.splats[live[i]];
        for j in i..n {
            let sj = &hot.splats[live[j]];
            let c64 = cosine(&si.semantics, &sj.semantics);
            let c3 = cosine(&si.position, &sj.position);
            let d = dist3(si.position, sj.position);
            cos64[i * n + j] = c64;
            cos64[j * n + i] = c64;
            cos3[i * n + j] = c3;
            cos3[j * n + i] = c3;
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }

    let mut curve_64d_cosine = Vec::new();
    for &t in COS_THRESHOLDS {
        let mut row = components_for_predicate(n, |i, j| cos64[i * n + j] >= t);
        row.threshold = t;
        curve_64d_cosine.push(row);
    }

    let mut curve_3d_cosine = Vec::new();
    for &t in COS_THRESHOLDS {
        let mut row = components_for_predicate(n, |i, j| cos3[i * n + j] >= t);
        row.threshold = t;
        curve_3d_cosine.push(row);
    }

    let mut curve_3d_euclidean = Vec::new();
    for &t in DIST_THRESHOLDS {
        let mut row = components_for_predicate(n, |i, j| dist[i * n + j] <= t);
        row.threshold = t;
        curve_3d_euclidean.push(row);
    }

    // 0.70 is the percolation window (LCC jumps 0.35→0.80 between 0.75 and 0.70).
    // 0.30 would materialise ~every pair and is not a well-merge diagnostic.
    let (bridges_64d, giant_25pct_birth) = kruskal_bridges_from_matrix(hot, &live, &cos64, 0.70);

    BasinCurveReport {
        n_live: n,
        n_muted: hot.splats.len().saturating_sub(n),
        dream_cycle: hot.dream_cycle,
        current_n_basins: hot.basins.len(),
        current_lcc,
        current_lcc_frac: if n == 0 {
            0.0
        } else {
            current_lcc as f64 / n as f64
        },
        curve_64d_cosine,
        curve_3d_cosine,
        curve_3d_euclidean,
        bridges_64d,
        giant_25pct_birth,
    }
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn kruskal_bridges_from_matrix(
    hot: &HotState,
    live: &[usize],
    cos64: &[f32],
    min_cos: f32,
) -> (Vec<BridgeMerge>, Option<BridgeMerge>) {
    let n = live.len();
    let mut edges: Vec<(f32, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let c = cos64[i * n + j];
            if c >= min_cos {
                edges.push((c, i, j));
            }
        }
    }
    edges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    let mut uf = UnionFind::new(n);
    let mut bridges = Vec::new();
    let mut giant = None;
    let giant_cut = (n as f64 * 0.25) as usize;
    for (c, i, j) in edges {
        let Some((sa, sb, _)) = uf.union(i, j) else {
            continue;
        };
        if sa >= MIN_BASIN && sb >= MIN_BASIN {
            let li = live[i];
            let lj = live[j];
            let rec = BridgeMerge {
                cosine: c,
                left_n: sa,
                right_n: sb,
                new_n: sa + sb,
                left_domain: hot.splats[li].domain.clone(),
                right_domain: hot.splats[lj].domain.clone(),
                cross_domain: hot.splats[li].domain != hot.splats[lj].domain,
                left_id: hot.splats[li].memory_id.to_string(),
                right_id: hot.splats[lj].memory_id.to_string(),
            };
            if giant.is_none() && rec.new_n >= giant_cut {
                giant = Some(rec.clone());
            }
            if bridges.len() < 40 {
                bridges.push(rec);
            }
        }
    }
    (bridges, giant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Splat;
    use uuid::Uuid;

    fn splat(id: u128, sem: [f32; 4], pos: [f32; 3]) -> Splat {
        Splat {
            memory_id: Uuid::from_u128(id),
            semantics: sem.to_vec(),
            position: pos,
            velocity: [0.0; 3],
            scale: [1.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [0; 4],
            mass: 1.0,
            radiance: 1.0,
            usage: 0.0,
            conductivity: 1.0,
            gain: 0.0,
            basin_locked: false,
            muted: false,
            duplicate_count: 1,
            domain: "scifact".into(),
            basin_id: None,
            lineage: vec![],
        }
    }

    #[test]
    fn two_wells_merge_only_below_their_cosine() {
        // Two tight pairs far apart in 64-d; 3-d puts them all in one blob.
        let a = splat(1, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let b = splat(2, [0.99, 0.14, 0.0, 0.0], [0.01, 0.0, 0.0]);
        let c = splat(3, [0.0, 1.0, 0.0, 0.0], [0.02, 0.0, 0.0]);
        let d = splat(4, [0.0, 0.99, 0.14, 0.0], [0.03, 0.0, 0.0]);
        let hot = HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: vec![a, b, c, d],
            basins: vec![],
            projection: None,
        };
        let report = basin_component_curve(&hot);
        let at_80 = report
            .curve_64d_cosine
            .iter()
            .find(|r| (r.threshold - 0.80).abs() < f32::EPSILON)
            .expect("0.80 row");
        assert!(
            at_80.lcc <= 2,
            "tight 64-d wells should not be one component at 0.80, lcc={}",
            at_80.lcc
        );
        let at_close = report
            .curve_3d_euclidean
            .iter()
            .find(|r| (r.threshold - 0.20).abs() < f32::EPSILON)
            .expect("0.20 row");
        assert_eq!(
            at_close.lcc, 4,
            "3-d proximity should glue all four at dist<=0.20"
        );
    }
}
