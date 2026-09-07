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
//! group of byte-identical texts and mutes the rest. A second pass mutes *near*
//! duplicates (same first ~200 normalized characters, group ≥ 8) so judge-prompt
//! clones with different shard payloads stop painting as 13k dots.
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
    /// Prefix/boilerplate groups (not byte-identical) muted this pass.
    #[serde(default)]
    pub near_groups: usize,
    #[serde(default)]
    pub near_muted: usize,
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

    let near = collapse_near_duplicates(state, records);
    report.near_groups = near.near_groups;
    report.near_muted = near.near_muted;
    report.muted += near.near_muted;
    report.already_muted += near.already_muted;
    report.groups += near.near_groups;
    report.largest_group = report.largest_group.max(near.largest_group);
    report.active = state.splats.iter().filter(|s| !s.muted).count();
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{HotState, Splat};
    use crate::record::MemoryRecord;
    use uuid::Uuid;

    fn splat(id: u128) -> Splat {
        Splat {
            memory_id: Uuid::from_u128(id),
            semantics: vec![0.1; 64],
            position: [0.0; 3],
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

    fn rec(id: u128, text: &str) -> MemoryRecord {
        let mut r = MemoryRecord::new("t", format!("{id}"), text);
        r.id = Uuid::from_u128(id);
        r
    }

    fn state(n: u128) -> (HotState, HashMap<Uuid, MemoryRecord>) {
        let mut hot = HotState {
            version: 1,
            dream_cycle: 0,
            last_dream_at: None,
            kinetic_energy: 0.0,
            splats: Vec::new(),
            basins: Vec::new(),
            projection: None,
        };
        let records = HashMap::new();
        for i in 1..=n {
            hot.splats.push(splat(i));
        }
        (hot, records)
    }

    #[test]
    fn byte_identical_mutes_all_but_one() {
        let (mut hot, mut records) = state(3);
        let text = "same bytes exactly";
        records.insert(Uuid::from_u128(1), rec(1, text));
        records.insert(Uuid::from_u128(2), rec(2, text));
        records.insert(Uuid::from_u128(3), rec(3, text));
        let report = collapse_duplicates(&mut hot, &records);
        assert_eq!(report.muted, 2);
        assert_eq!(hot.splats.iter().filter(|s| !s.muted).count(), 1);
    }

    #[test]
    fn near_boilerplate_mutes_payload_variants() {
        let (mut hot, mut records) = state(8);
        for i in 1..=8 {
            let prefix = "You are a strict dataset judge for a Qwen3.8-27B QLoRA run. \
                 MANDATORY OPERATOR OVERRIDES never violate. Collaborator turns are real SFT targets. \
                 Keep Rust as Rust. Preserve flags and numbers. ";
            let text = format!("{prefix}shard-{i} unique tail payload");
            records.insert(Uuid::from_u128(i), rec(i, &text));
        }
        let report = collapse_duplicates(&mut hot, &records);
        assert!(report.near_muted >= 7, "{report:?}");
        assert_eq!(hot.splats.iter().filter(|s| !s.muted).count(), 1);
    }

    #[test]
    fn short_chats_are_not_prefix_collapsed() {
        let (mut hot, mut records) = state(8);
        for i in 1..=8 {
            records.insert(Uuid::from_u128(i), rec(i, &format!("yeah ok {i}")));
        }
        let report = collapse_duplicates(&mut hot, &records);
        assert_eq!(report.near_muted, 0, "{report:?}");
        assert_eq!(hot.splats.iter().filter(|s| !s.muted).count(), 8);
    }
}

/// Fingerprint for boilerplate that is *not* byte-identical (judge prompts with
/// different shard ids, HARD OUTPUT RULES with different payloads). Short chats
/// are left alone so two real "yeah"s do not merge.
fn boilerplate_key(text: &str) -> Option<String> {
    let compact: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() < 96 {
        return None;
    }
    Some(compact.chars().take(160).collect())
}

const NEAR_MIN_GROUP: usize = 8;

fn collapse_near_duplicates(
    state: &mut HotState,
    records: &HashMap<Uuid, MemoryRecord>,
) -> DedupeReport {
    let mut by_key: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, splat) in state.splats.iter().enumerate() {
        if splat.muted {
            continue;
        }
        let Some(record) = records.get(&splat.memory_id) else {
            continue;
        };
        let Some(key) = boilerplate_key(&record.text) else {
            continue;
        };
        by_key.entry(key).or_default().push(index);
    }

    let mut report = DedupeReport::default();
    let mut plan: Vec<(usize, u32, Vec<usize>)> = Vec::new();
    for group in by_key.values() {
        if group.len() < NEAR_MIN_GROUP {
            continue;
        }
        report.near_groups += 1;
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
        state.splats[keep].duplicate_count = state.splats[keep].duplicate_count.max(count);
        for index in others {
            if state.splats[index].muted {
                report.already_muted += 1;
            } else {
                report.near_muted += 1;
            }
            let splat = &mut state.splats[index];
            splat.muted = true;
            splat.duplicate_count = 1;
            if !splat.basin_locked {
                splat.basin_id = None;
            }
            splat.velocity = [0.0, 0.0, 0.0];
        }
    }
    report
}
