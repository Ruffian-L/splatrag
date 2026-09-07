# HANDOFF — resume here

Written 2026-08-29 at Jason's request, usage limit incoming.
**To resume: say "finish what you were doing."**

Nothing is landed. No live state was mutated. Tree is exactly as found.

---

## 1. Decisions Jason still owes (this is the resume point)

**A. The crumb patch — land it, shelve it, or measure blast radius first?**
`basin_crumb_cosine = 0.60`. Recovers ~74% of orphans (4,811 → 1,271), basin
count pinned at 1,748, largest basin +3.2%. Evidence is **structural only** —
no proof it improves ranking. Options given to him, undecided:
  1. land on structural grounds
  2. leave as patch file
  3. blast-radius measurement first: same query set at 0.72 vs 0.60 on the
     personal store, measure how much top-k moves. Risk-sizing, not quality.

**B. The compaction — I recommended just doing it; he has not said go.**
~62 MB of muted duplicate splats (42.9 MB of `hot/state.json` + ~19 MB of
HNSW). Safe: discovery already skips muted splats, so purging cannot move a
basin. Cold keeps every byte. **Do not touch cold — that is the actual rewrite.**

**C. The CHANGELOG.md entry is still owed and deliberately held.**
His `CHANGELOG.md` has uncommitted in-progress work; I did not want to tangle
my line into it. Needs his go-ahead, or land it together with the patch as one
commit.

---

## 2. Artifacts on disk

| what | where |
|---|---|
| main research log (172 lines, incl. addendum + scope correction) | `research_logs/2026-08-29_basin-orphans-crumb-threshold.md` |
| the patch (config.rs + topology.rs, ~20 lines) | `scratchpad/basin/decouple-crumb-threshold.patch` |
| sweep configs + raw topology JSON | `scratchpad/basin/` |
| patched binary (isolated target, live bin untouched) | `scratchpad/target/release/splatrag` |

Scratchpad root:
`/tmp/claude-1000/-home-ruffianl-Videos-Screencasts/f689ac34-4f1c-4ce2-9af7-c3adc4e96fe0/scratchpad`
**Volatile — copy the patch out if it matters.**

---

## 3. Findings, so they are not re-derived

- **Orphans are real.** 6,379 live (5.2%); 4,812 on a fresh discovery run. A raw
  `grep '"basin_id":null'` says 40,016 and is **misleading** — two thirds is
  muted splats, skipped by design.
- **Orphans are penalized.** `basin_id = None` zeroes three score terms in
  `retrieval.rs:291` (peer, centroid, basin_local). Too unique to cluster =
  permanently harder to retrieve.
- **`basin_attach_cosine` does two unrelated jobs** — centroid merge gate
  (`topology.rs:42`) and crumb attach threshold (`:218`/`:289`). That is the
  actual defect. Decoupling gives identical orphan rescue with no structural
  cost.
- **`min_basin_size = 2` FAILED.** Was my recommendation. Rescues 31 splats,
  doubles basin count. Do not retry.
- **SciFact is dropped** at Jason's direction — it was his store-durability
  test, not a quality guard. It cannot validate basin changes (all
  `weight_basin_*` and `weight_radiance` are 0.0, `basin_expand_max_size = 0`),
  but that is not a defect in his plan.

---

## 4. Corrections to the other Claude's report (commit `4ee51f1`, NOT in this repo)

That commit is in another tree. Nothing of Jason's or mine was swept into it.

- **Its dedup premise is wrong.** The 33,686 "redundant copies" are already
  collapsed — they are the muted set, `basin_id` cleared (`dedupe.rs:108-119`),
  skipped before discovery. They are **not** inflating density, spread,
  stability, or opacity.
- **Its motivating example is misread.** The five "identical" hits are not
  duplicates. Pulled that exact query: 8 hits, **8 distinct sha256**, lengths
  7761/6117/6839/5198/3412/5352. Six different records sharing a long
  boilerplate system-prompt prefix. Byte-level dedup structurally cannot fix
  that.
- Its "Next" ordering therefore points a 27% store mutation at a problem that
  is already solved, and would not fix the thing that produced the symptom.
- **Not checked by me:** the pairing half (`parent_id` 4.0%, speaker nulls,
  content in the speaker column). That argument stands on its own. Offered to
  run those counts; he has not said yes.
- A correction belongs appended in whichever tree holds `4ee51f1`. House rule:
  append, never rewrite.

---

## 5. Unrelated, already finished

`~/Videos/Screencasts/Screencast_20260829_155901_x.mp4` — 4.61 MB, trimmed
10.5s–27.2s, 1280x632 @30fps H.264. Posted to X. Original `.webm` untouched.
Closed, needs nothing.
