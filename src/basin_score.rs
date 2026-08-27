//! Basin-aware retrieval signals from embedding geometry (not domain stamps).
//!
//! Pure functions so unit tests drive the real scoring path without embedder/CLI.

use crate::geometry::cosine;
use std::collections::HashMap;
use uuid::Uuid;

/// For each basin id, the strongest dense cosine among members that have a dense score.
/// Geometry-only: uses basin membership of dense hits, not domain labels.
pub fn basin_peer_dense_support(
    dense_scores: &HashMap<Uuid, f32>,
    id_to_basin: &HashMap<Uuid, Option<String>>,
) -> HashMap<String, f32> {
    let mut out: HashMap<String, f32> = HashMap::new();
    for (id, &score) in dense_scores {
        if score <= 0.0 {
            continue;
        }
        if let Some(Some(basin)) = id_to_basin.get(id) {
            out.entry(basin.clone())
                .and_modify(|s| *s = s.max(score))
                .or_insert(score);
        }
    }
    out
}

/// Soft boost for a candidate: peer support in its basin (0 if no basin / no dense peers).
pub fn peer_support_for_candidate(
    basin_id: Option<&str>,
    peer_support: &HashMap<String, f32>,
) -> f32 {
    basin_id
        .and_then(|b| peer_support.get(b).copied())
        .unwrap_or(0.0)
}

/// Cosine of query semantics to mean of member semantics (centroid).
/// Empty members or empty vectors → 0.
pub fn basin_centroid_affinity(
    query_semantics: &[f32],
    member_ids: &[Uuid],
    id_to_semantics: &HashMap<Uuid, Vec<f32>>,
) -> f32 {
    if member_ids.is_empty() || query_semantics.is_empty() {
        return 0.0;
    }
    let dim = query_semantics.len();
    let mut acc = vec![0.0_f32; dim];
    let mut n = 0usize;
    for id in member_ids {
        let Some(sem) = id_to_semantics.get(id) else {
            continue;
        };
        if sem.len() != dim {
            continue;
        }
        for (a, b) in acc.iter_mut().zip(sem.iter()) {
            *a += *b;
        }
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    let inv = 1.0 / n as f32;
    for a in &mut acc {
        *a *= inv;
    }
    cosine(query_semantics, &acc).max(0.0)
}

/// Whether the local-neighborhood boost should run for a basin of this size.
///
/// `max_members == 0` means no cap. Mega-basins (tens of thousands of members)
/// otherwise add a near-constant sum that drowns BM25 under RRF.
pub fn local_boost_applies(member_count: usize, max_members: usize) -> bool {
    max_members == 0 || member_count <= max_members
}

/// Combine hybrid base score with basin peer support and optional centroid affinity.
///
/// `weight_peer` multiplies max dense score in the candidate's basin.
/// `weight_centroid` multiplies query–centroid cosine for that basin.
pub fn basin_aware_final(
    base_score: f32,
    peer_support: f32,
    centroid_affinity: f32,
    weight_peer: f32,
    weight_centroid: f32,
) -> f32 {
    base_score + weight_peer * peer_support + weight_centroid * centroid_affinity
}

/// Keep the `limit` nearest neighbor ids given (id, distance) pairs.
/// Distance-ascending order is the real graph; UUID sort is not geometry.
pub fn nearest_neighbor_ids(mut hits: Vec<(Uuid, f32)>, limit: usize) -> Vec<Uuid> {
    if limit == 0 {
        return Vec::new();
    }
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    hits.dedup_by(|a, b| a.0 == b.0);
    hits.into_iter().take(limit).map(|(id, _)| id).collect()
}

/// Local geometry cluster boost: max dense score among dense hits whose semantics are
/// close enough to the candidate (soft basin without PH mega-components).
pub fn local_semantic_neighborhood_boost(
    candidate_semantics: &[f32],
    dense_hits: &[(f32, Vec<f32>)],
    threshold: f32,
) -> f32 {
    let mut best = 0.0_f32;
    for (score, sem) in dense_hits {
        if *score <= 0.0 || sem.len() != candidate_semantics.len() {
            continue;
        }
        if cosine(candidate_semantics, sem) >= threshold {
            best = best.max(*score);
        }
    }
    best
}

/// Sum of dense scores in the candidate's soft semantic neighborhood.
/// Prefers candidates at the center of a dense hit clump (many peers) over isolates
/// that only share max with one hit — helps rank_miss golds sitting mid-pool.
pub fn local_semantic_neighborhood_sum(
    candidate_semantics: &[f32],
    dense_hits: &[(f32, Vec<f32>)],
    threshold: f32,
) -> f32 {
    let mut acc = 0.0_f32;
    for (score, sem) in dense_hits {
        if *score <= 0.0 || sem.len() != candidate_semantics.len() {
            continue;
        }
        if cosine(candidate_semantics, sem) >= threshold {
            acc += *score;
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_support_takes_max_dense_in_basin() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let c = Uuid::from_u128(3);
        let dense = HashMap::from([(a, 0.4), (b, 0.9), (c, 0.2)]);
        let map = HashMap::from([
            (a, Some("basin-x".into())),
            (b, Some("basin-x".into())),
            (c, Some("basin-y".into())),
        ]);
        let support = basin_peer_dense_support(&dense, &map);
        assert!((support["basin-x"] - 0.9).abs() < 1e-5);
        assert!((support["basin-y"] - 0.2).abs() < 1e-5);
        assert_eq!(peer_support_for_candidate(Some("basin-x"), &support), 0.9);
        assert_eq!(peer_support_for_candidate(None, &support), 0.0);
    }

    #[test]
    fn centroid_affinity_prefers_aligned_basin() {
        let q = vec![1.0, 0.0, 0.0];
        let m1 = Uuid::from_u128(10);
        let m2 = Uuid::from_u128(11);
        let sem = HashMap::from([(m1, vec![1.0, 0.0, 0.0]), (m2, vec![0.9, 0.1, 0.0])]);
        let high = basin_centroid_affinity(&q, &[m1, m2], &sem);
        let ortho = basin_centroid_affinity(&q, &[m1], &HashMap::from([(m1, vec![0.0, 1.0, 0.0])]));
        assert!(high > 0.9);
        assert!(ortho < 0.1);
    }

    #[test]
    fn basin_aware_final_adds_weighted_terms() {
        let s = basin_aware_final(1.0, 0.5, 0.8, 2.0, 1.0);
        assert!((s - (1.0 + 1.0 + 0.8)).abs() < 1e-5);
    }

    #[test]
    fn huge_basin_skips_local_boost_small_basin_keeps_it() {
        assert!(local_boost_applies(12, 200), "small basin is boosted");
        assert!(
            !local_boost_applies(20_983, 200),
            "20k unlabeled bag must not get the local sum/max"
        );
        assert!(local_boost_applies(20_983, 0), "0 disables the cap");
    }

    #[test]
    fn empty_dense_yields_empty_support() {
        let support = basin_peer_dense_support(&HashMap::new(), &HashMap::new());
        assert!(support.is_empty());
    }

    #[test]
    fn neighborhood_boost_uses_near_dense_hits() {
        let cand = vec![1.0, 0.0, 0.0];
        let hits = vec![(0.7, vec![0.99, 0.01, 0.0]), (0.9, vec![0.0, 1.0, 0.0])];
        let b = local_semantic_neighborhood_boost(&cand, &hits, 0.8);
        assert!((b - 0.7).abs() < 1e-4, "got {b}");
        let none = local_semantic_neighborhood_boost(&cand, &hits, 0.999);
        assert_eq!(none, 0.0);
    }

    #[test]
    fn nearest_neighbor_ids_orders_by_distance_not_uuid() {
        // UUIDs ordered reverse of distance so UUID-sort would pick farthest.
        let far = Uuid::from_u128(1);
        let near = Uuid::from_u128(99);
        let mid = Uuid::from_u128(50);
        let got = nearest_neighbor_ids(vec![(far, 0.9), (near, 0.1), (mid, 0.4)], 2);
        assert_eq!(got, vec![near, mid]);
    }

    #[test]
    fn neighborhood_sum_rewards_multi_peer_clump() {
        let cand = vec![1.0, 0.0, 0.0];
        let hits = vec![
            (0.5, vec![0.99, 0.01, 0.0]),
            (0.6, vec![0.98, 0.02, 0.0]),
            (0.9, vec![0.0, 1.0, 0.0]),
        ];
        let s = local_semantic_neighborhood_sum(&cand, &hits, 0.8);
        assert!((s - 1.1).abs() < 1e-4, "got {s}");
        let m = local_semantic_neighborhood_boost(&cand, &hits, 0.8);
        assert!((m - 0.6).abs() < 1e-4);
        assert!(s > m);
    }
}
