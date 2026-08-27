//! Duplicate collapse — the garbage pass of the dream cycle.
//!
//! Ingest is lossless by contract: every chat, every export, byte-for-byte into cold.
//! That contract necessarily puts boilerplate in the field. One Google Takeout export
//! contributed 10,051 copies of the same 198-character activity notice — 15.9% of the
//! personal store — and the projection packed them into a ~1.0-unit cube of a field
//! spanning 10.0 units. A clump that dense is a gravity well: with `attraction = 2.0`
//! every copy attracts every other at cosine ~1.0, their ANN neighbor lists fill with
//! each other, and the geometric component drags the rest of the store in with it. The
//! measured result was one basin holding 62,660 of 63,062 memories at stability 0.26.
//!
//! So the fix belongs *after* ingest, not inside it. Dream keeps one representative per
//! group of byte-identical texts and mutes the rest.
//!
//! # Why `content_sha256` and not the embedding
//!
//! Identical text does not produce an identical vector. llama.cpp embeds in batches
//! (`batch_size = 32`), and batch composition perturbs the result: of those 10,051
//! byte-identical records, only **65** had bitwise-equal vectors, the rest landing at
//! cosine 0.9997–0.9998. A vector-hash dedupe finds 0.6% of the problem and reports
//! success. The content hash is computed once at ingest, is exact, and is already on
//! every cold record.
//!
//! # Why mute and not delete
//!
//! [`crate::ann::KeyedAnn`] has no removal path, so a deleted splat would leave a live
//! key pointing at nothing. Muting keeps the ANN index consistent, keeps cold
//! append-only (`verify_append_only` still holds across a dream), and is reversible by
//! clearing one flag. Nothing is lost — which is the same contract ingest is under.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::geometry::HotState;
use crate::record::MemoryRecord;

/// What one collapse pass did.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DedupeReport {
    /// Groups of two or more byte-identical texts.
    pub groups: usize,
    /// Splats muted by this pass (were active, now are not).
    pub muted: usize,
    /// Splats already muted by an earlier pass.
    pub already_muted: usize,
    /// Active splats remaining in the field.
    pub active: usize,
    /// Size of the largest duplicate group seen.
    pub largest_group: usize,
}

/// Mute every splat whose cold text is byte-identical to another's, keeping one
/// representative per group.
///
/// The representative is chosen deterministically: a `basin_locked` splat wins outright
/// (locking is a human decision and must survive), then highest radiance, then lowest
/// `memory_id` as a tiebreak so repeated runs agree.
///
/// Splats with no matching cold record are left alone — this pass only ever acts on
/// evidence it can read.
pub fn collapse_duplicates(
    state: &mut HotState,
    records: &HashMap<Uuid, MemoryRecord>,
) -> DedupeReport {
    let mut by_hash: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, splat) in state.splats.iter().enumerate() {
        if let Some(record) = records.get(&splat.memory_id) {
            by_hash
                .entry(record.content_sha256.as_str())
                .or_default()
                .push(index);
        }
    }

    let mut report = DedupeReport::default();
    let mut plan: Vec<(usize, u32, Vec<usize>)> = Vec::new();

    for (_, group) in by_hash.iter() {
        if group.len() < 2 {
            continue;
        }
        report.groups += 1;
        report.largest_group = report.largest_group.max(group.len());

        let keep = *group
            .iter()
            .min_by(|&&a, &&b| {
                let sa = &state.splats[a];
                let sb = &state.splats[b];
                sb.basin_locked
                    .cmp(&sa.basin_locked)
                    .then_with(|| {
                        sb.radiance
                            .partial_cmp(&sa.radiance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| sa.memory_id.cmp(&sb.memory_id))
            })
            .expect("group is non-empty");

        let others: Vec<usize> = group.iter().copied().filter(|&i| i != keep).collect();
        plan.push((keep, group.len() as u32, others));
    }

    for (keep, count, others) in plan {
        state.splats[keep].muted = false;
        state.splats[keep].duplicate_count = count;
        for index in others {
            if state.splats[index].muted {
                report.already_muted += 1;
            } else {
                report.muted += 1;
            }
            let splat = &mut state.splats[index];
            splat.muted = true;
            splat.duplicate_count = 1;
            // A muted splat must not carry stale membership: it is out of the field, and
            // leaving a basin_id behind would keep it in that basin's member list.
            if !splat.basin_locked {
                splat.basin_id = None;
            }
            splat.velocity = [0.0, 0.0, 0.0];
        }
    }

    report.active = state.splats.iter().filter(|s| !s.muted).count();
    report
}
