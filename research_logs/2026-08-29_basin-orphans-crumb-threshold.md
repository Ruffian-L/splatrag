# Basin orphans: the attach knob does two jobs

2026-08-29. Read-only investigation on the live personal store
(`/home/ruffianl/Hub/Projects/splatrag/memory`, dream cycle 27, 121,393 splats).
Nothing live was mutated. All sweeps used `diagnose-topology`, which previews
discovery without saving.

## Why we looked

A `recall` on the personal store returned hits with `basin_label: null`. Jason
asked whether "too unique to fall in a basin" memories exist, and whether they
need an unsorted basin so they stop reading as unsorted memory in the store.

## What is actually there

`topology.rs:84` — a component smaller than `min_basin_size` is dissolved and
its members get `basin_id = None`. `attach_crumbs` then gives each orphan one
second chance to join an existing basin at `basin_attach_cosine`. Fail both and
`basin_id` stays `None` permanently. There is no catch-all.

Live store, counted off `hot/state.json`:

| bucket | count | share |
|---|---:|---:|
| in a basin | 82,650 | 67.4% |
| muted (skipped by discovery, by design) | 33,637 | 27.4% |
| live orphans | 6,379 | 5.2% |

A raw `grep '"basin_id":null'` says 40,016 and is misleading — two thirds of it
is muted splats that `discover_basins` skips on purpose. The real orphan
population is 6,379, almost entirely `domain: chat`. A fresh discovery run
brings it to 4,812, so the stored state is mildly stale.

The cost of `basin_id = None` is not cosmetic. In `retrieval.rs:291` three score
terms silently go to zero: `peer` (peer support), `cent` (centroid boost,
`unwrap_or(0.0)`), and the `basin_local` neighborhood boost (`member_count` 0).
An orphan can only compete on raw bm25 + cosine + radiance. Being too unique to
cluster makes a memory permanently harder to retrieve.

## Sweep 1 — existing knobs (stock binary)

| variant | assigned | unassigned | basins | largest | p50 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| base (min 3 / attach 0.72) | 84,249 | 4,812 | 1,746 | 4,464 | 7 | 26 |
| min_basin_size = 2 | 84,280 | 4,781 | 3,521 | 4,106 | 4 | 15 |
| attach 0.66 | 86,850 | 2,211 | 1,508 | 4,882 | 7 | 25 |
| attach 0.60 | 87,790 | 1,271 | 1,322 | 4,889 | 7 | 26 |
| attach 0.50 | 88,264 | 797 | 1,233 | 4,876 | 7 | 28 |
| core 0.76 | 84,258 | 4,803 | 1,800 | 4,659 | 6 | 24 |

**`min_basin_size = 2` failed.** It was the pre-sweep recommendation and it is
wrong. It rescues 31 splats and doubles the basin count, because it lets pairs
form their own basin *instead of* being crumb-attached to a real neighbour. It
fragments rather than rescues. Recorded because it was predicted to work.

`basin_core_cosine` is near-inert here. `basin_attach_cosine` is the live lever.

## The finding

`basin_attach_cosine` is doing two unrelated jobs. `topology.rs:42` uses it as
the **centroid coherence gate** while components merge:

```rust
let centroid_gate = config.basin_attach_cosine
    .max(config.semantic_threshold).clamp(0.0, core);
```

and `topology.rs:218/289` uses it as the **crumb attach threshold**. So lowering
it to rescue orphans also loosens merging as a side effect — in sweep 1, basins
fall 1,746 → 1,322 and largest grows toward the 5,000 cap. The doc comment on
the field claims attachment "never merges two wells", which is true of the hop
and false of the knob.

## Sweep 2 — decoupled (patched binary, isolated target dir)

Added `basin_crumb_cosine` (0 = inherit, preserving old behavior), used only for
attachment. Merge gate pinned at 0.72.

| crumb | unassigned | basins | largest |
|---|---:|---:|---:|
| 0.0 (inherit = baseline) | 4,811 | 1,748 | 4,475 |
| 0.66 | 2,210 | 1,748 | 4,591 |
| 0.60 | 1,271 | 1,748 | 4,620 |
| 0.50 | 797 | 1,748 | 4,627 |

Decoupled 0.60 rescues **exactly the same 1,271** as coupled 0.60, with basin
count pinned at 1,748 and largest +3.2% instead of +9.5%. Same benefit, side
effect gone. `basins` never moving confirms crumb attachment genuinely does not
merge wells once the knob is separated from the gate.

Marginal rescue per band shows the knee at 0.60:

| band | rescued in band | remaining |
|---|---:|---:|
| ≥ 0.66 | 2,601 | 2,210 |
| ≥ 0.60 | 939 | 1,271 |
| ≥ 0.55 | 292 | 979 |
| ≥ 0.50 | 182 | 797 |

## We think

`basin_crumb_cosine = 0.60`, decoupled, is the right change: it recovers ~74% of
orphans with no structural cost, and the rescue curve saturates immediately
below it, so going lower buys few memories at steadily weaker similarity.

The residual ~1,271 are genuinely isolated. Those are where Jason's unsorted
idea earns its place — but as a **label/state for visibility**, never as a
geometric basin. A single catch-all basin would be actively harmful, because
`cent` scores candidates against a basin centroid and the centroid of ~1,271
unrelated fragments is a meaningless mid-space point that would fire on any
query.

## What this does NOT show

No retrieval-quality ground truth. Every number above is structural. We did not
establish that attaching a 0.60-similar crumb *improves ranking* — only that it
does not damage basin structure. SciFact was not run: `datasets/scifact` is
absent and `ultima` would need its eval store rebuilt from
`receipts/scifact-20260820/gold/data/`. Did not mutate the live store. Did not
land the patch.

## Next

Rebuild the SciFact rig and run `ultima` against crumb 0.72 vs 0.60 to check the
0.8804 freeze holds. That is the mutation that tests the "no damage" claim.

Patch held at `scratchpad/basin/decouple-crumb-threshold.patch` (config.rs +
topology.rs, ~20 lines). Tree left clean; the pre-existing uncommitted
CHANGELOG.md / service.rs work is untouched and unrelated.

---

## Addendum — the SciFact guard rail is blind to basins

Went to run `ultima` at crumb 0.72 vs 0.60 as the quality check the main log
called for. It cannot work. Every SciFact eval config — `ultima-20260822/
control.toml`, `mix.toml`, and all nine `scifact-20260821/eval_rrf_*.toml` —
sets:

```toml
weight_basin_peer = 0.0
weight_basin_centroid = 0.0
weight_basin_local = 0.0
weight_radiance = 0.0
basin_expand_max_size = 0
```

All four basin paths into the score are off. The first three are zero-weighted.
The fourth is disabled by `basin_expand_max_size = 0`, since `retrieval.rs:112`
reads `if basin.member_ids.len() > max_size { continue; }` and every real basin
has at least one member.

Live `configs/personal.toml` runs `weight_radiance = 1.0` and
`weight_basin_local = 0.3`. So the benchmark and production disagree about
whether basins participate in ranking at all.

**Consequence.** The 0.8804 freeze that `ultima --help` advertises as the guard
rail is not a guard rail for basin work. Any basin change — the crumb threshold,
`min_basin_size`, discovery, attachment, eviction — is unguarded by it. A
regression could not move the number, and an improvement could not either. An
A/B at 0.72 vs 0.60 would have returned identical results, and "the freeze held"
would have been a false negative reported as safety.

This outranks the crumb patch. A basin change cannot be validated until an eval
exists whose config turns the basin weights on.

## Did not

Did not run `ultima`. Did not build a basin-aware eval. Did not land the crumb
patch — it remains unvalidated for retrieval quality, and the plan to validate
it was based on an eval that cannot see it. Structural claims in the main log
stand; the quality question is still open and now has no cheap answer.

### Scope correction (same session)

Jason: SciFact was built to test **store durability**, not to guard retrieval
quality, and was never intended as a basin guard rail. The addendum above framed
its basin-blindness as a missing guard rail; that framing assigned an intent the
benchmark never had. The narrow fact is unchanged — SciFact cannot validate a
basin change — but it is not evidence of a gap in the codebase's own plan.
SciFact work stopped here at his direction.
