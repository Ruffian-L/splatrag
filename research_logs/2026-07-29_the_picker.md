# 2026-07-29 — The picker, and four defects found by running it

Changelog, not a claim. What was built, what broke, what the numbers were.

**Corpus:** 930 memories (Grok export + assets), `data/cold/memories.jsonl`.
**Tests:** 43 → 56, all passing.

---

## Why

Steering works (hydrodynamic-swarm). Memory works (SplatRAG). The joint has never existed because
nothing decided **which** memory to load and **how hard**. That decision is `src/pick.rs`.

Prior art deliberately not re-walked: OI operators (`src/inversion.rs`), the 64D packet wire
(`src/packet.rs`, G1), hydro's force laws.

---

## The bridge carries text, not vectors

| | SplatRAG | hydro |
|---|---|---|
| vector | `semantics[64]`, L2-normalized to 1.0 | `mu[N, 2560]`, `center_l2 ≈ 141` |
| from | Qwen3-Embedding-8B matryoshka slice | Gemma 3 4B residual |
| guard | none | `model_dim: 2560`, `model_fp: 2710330125` |

No trained 64→2560 decoder exists. Injecting `semantics_64` into a residual would not error — it
would steer confidently on noise, and hydro's `model_fp` is the only thing standing between that and
a plausible-looking result.

So a pick carries the memory's **text**, and the host embeds it with its own encoder
(`GemmaModel::embed`, `hydro src/concourse/embed/gemma.rs:149`). The dimension gap stops being a
conversion problem. `Pick::injection` states this on the wire; `source_dim` / `source_embedder` let a
consumer refuse a packet rather than guess.

---

## Four defects, in the order they were found

### 1. `gain` had two incompatible neutrals

`geometry.rs` used `1.0` for "never steered". `steer` wrote a steering α into the same field
(`service.rs`), and α is clamped to ±0.35 by `clamp_alpha`. So `1.0` meant "untouched" while also
being **outside the legal α band**, and read downstream as maximum amplification.

Consequence already visible in the G1 receipt: `collapse_risk` is `|α| >= 0.40`, so every untouched
splat reported `"collapse_risk": true`. The transcript in `logs/steering_plumbing_G1/RECEIPT.md`
shows exactly that on a splat nobody had ever steered.

Fixed: `gain` is a steering α throughout, `0.0` = unsteered, migrated in `HotState::load`. Safe
because no genuinely steered splat can hold 1.0. `steer` now stores the α it *applied*
(`clamp_alpha(opts.gain)`), not the one requested. `collapse_risk` still scores the **requested**
gain — scoring the clamped value would make the flag permanently false.

All 930 splats now read `gain: 0.0`. Backups at `data/hot/*.bak_pre_gain_migration`.

### 2. Loading multiple memories stacked the push instead of dividing it

Jason's question — *"we tried to load one memory packet, what if we loaded multiple?"* — exposed it.
Every pick was getting the full α. Three memories at 0.35 = **1.05 total**, far past the 0.40
collapse onset. A "does more memory help?" run would have measured collapse and blamed memory count.

Fixed: one `gain_budget` **divided** across picks by score share. Verified live — total α is 0.0049
at `--limit 1`, `3`, and `6`.

### 3. Zero cosine meant "never measured", not "orthogonal"

`retrieval.rs` did `cosine_scores.get(&id).unwrap_or(0.0)`. A candidate found by BM25 alone was
never in the ANN result set, so its cosine defaulted to 0.0 — a ~7-point penalty at
`weight_cosine 10.0` against a true value around 0.7. The hybrid systematically demoted exactly the
lexical matches it exists to include.

Measured before: the top two hits for `"what did I eat for breakfast"` scored `cosine 0.0` with
`bm25 7.4`. After: `0.533, 0.358, 0.410, 0.317, 0.132`. Zero zero-cosine candidates on any query.

Fixed: the ANN is an approximate index for *finding* candidates; once one is in hand the exact
cosine is a 64-d dot product against semantics already in memory. Compute it.

### 4. The confidence heuristic was measured backwards

First attempt scored confidence as **separation** — the normalized gap between the top two hits.
Live result:

| query | separation |
|---|---|
| `zzzqqq nonexistent gibberish token` | **0.751** |
| `the physics of friendship` | **0.016** |

Inverted. A nonsense query yields one accidental outlier above a flat junk pool → large gap. A topic
the archive genuinely covers yields many similarly-good memories → tiny gap. Separation measures "is
there one weird outlier", not "do we know this" — and for a multi-memory picker, many-good-matches is
the *best* case, which separation punishes hardest. Under it, gibberish proposed α = 0.263 while a
well-covered topic proposed 0.005.

Replaced with a **null model**: sample 256 splats by stride (deterministic, so the number is
reproducible), take the mean and std of their cosine to the query, and score the top hit's z above
that. Confidence is `z / 6.0`, clamped. The 6σ is a *chosen* scale, not a fitted one.

| query | confidence | α | top z | separation |
|---|---|---|---|---|
| the physics of friendship | **0.353** | 0.123 | 2.12 | 0.016 |
| ontological inversion householder involution | 0.255 | 0.089 | 1.53 | 0.566 |
| splatrag rust ingest uuid v5 | 0.228 | 0.080 | 1.37 | 0.039 |
| grok export asset OCR screenshot | 0.223 | 0.078 | 1.34 | 0.348 |
| what did I eat for breakfast | 0.206 | 0.072 | 1.23 | 0.642 |
| **zzzqqq nonexistent gibberish token** | **0.097** | 0.034 | 0.58 | 0.751 |

Every real topic outranks every non-topic. Gibberish now receives the least steering. `separation` is
kept as a **reported diagnostic** because its anti-correlation with confidence in this table is the
evidence that it was the wrong signal — deleting it would delete the finding.

A smaller one alongside it: confidence originally moved with `--limit` (0.0195 → 0.0103, same query,
same top hit) because a larger limit over-fetched deeper and widened the spread. The confidence pool
is now fixed at 8 regardless of how many picks are requested.

### 5. `steer --gain` is inert: the self-axis is degenerate

Found by reading `cosine_before_after: -1.0` on a live `steer --gain -0.2`. An α of 0.2 should be a
partial rotation; exactly −1.0 is a perfect 180° flip. Probing every op across α:

```
      polarity: α0.05→-1.000  α0.10→-1.000  α0.20→-1.000  α0.30→-1.000  α0.35→-1.000  α0.90→-1.000
   householder: α0.05→+1.000  α0.10→+1.000  α0.20→+1.000  α0.30→+1.000  α0.35→+1.000  α0.90→+1.000
 negative_gain: α0.05→+1.000  α0.10→+1.000  α0.20→+1.000  α0.30→+1.000  α0.35→+1.000  α0.90→+1.000
```

**α does nothing.** `service.rs` steers on the self-axis — `apply_steering(&before, &before, …)` —
and these operators act only on the component of `h` along the axis. When the axis *is* `h`, that
component is the whole vector and normalization discards the only quantity α scaled:

- polarity: `out = −α·h` → exactly `−ĥ` for every α
- householder: `out = (1 − 2α)·h` → `+ĥ` for all α < 0.5
- negative_gain: `out = (1 − α)·h` → `+ĥ`

So `steer --gain` is a full flip (polarity) or a no-op (the other two), with nothing between. The
measured α ≈ 0.15–0.30 sweet band **cannot be expressed through this API at all**. Both existing
tests passed anyway: `cos < 0.5` is trivially true at −1.0, and `cos > 0.99` is trivially true when
positive gain is a no-op.

The operators are not wrong — the axis is. With a reference direction the memory genuinely projects
onto, α is monotonic:

```
 negative_gain: α0.05→+0.999  α0.10→+0.995  α0.20→+0.980  α0.30→+0.957  α0.35→+0.943
```

**Not fixed here.** `steer` is marked DONE in `TEAM_GOAL_STEERING_PLUMBING.md` and changing what α
means to a splat is Jason's call, not a side effect of building the picker. What was done instead:
two tests pin the degeneracy in code (`self_axis_steering_is_degenerate_and_alpha_does_nothing`,
`a_distinct_axis_makes_alpha_monotonic`), so it fails loudly the moment someone makes α responsive.

Recommended fix when opened: pass a non-degenerate axis — basin centroid, or `hot.projection.mean`,
which is already a 64-d field mean sitting in the hot state. That is the μ in the OI prior art's
`Φ_c(h) = μ + (I − 2P_c)(h − μ)`, and it is why that formula has a μ in it.

**Consequence for G1/G2:** the `steer` smokes A and B pass on cosine movement, and the movement is
real — but it is fixed-magnitude, not α-controlled. Any statement that SplatRAG steers *by a chosen
amount* is currently unsupported.

### Minor: `gain` cannot be reset

`gain` records the last α applied, not net rotation. Because self-axis polarity is an exact
involution, steering twice restores the semantics — verified, all 930 splats at cosine 1.0 to the
pre-session backup — but the field still reads `−0.2`. There is no way to set it back to 0, since
`--gain 0` means "leave semantics untouched" and skips the write entirely. One splat
(`3aa10654`) is in that state now.

---

## What does not discriminate on this corpus

Measured on the settled field, and the reason the picker ignores all three:

- **`basin_id`** — 817 of 930 splats in one basin. The PCA fit scales the cloud to
  `TARGET_RMS_RADIUS = 4.0`, then 1200 dream steps contract it to a median radius of **0.99**, while
  `basin_radius = 0.8` was calibrated against the fitted 4.0. One component swallows everything.
  **Not "fixed" by cranking the radius** — per Jason, placement and labels must stay emergent from
  geometry and context, not enforced.
- **`radiance`** — exactly `4.5` on all 930 splats, so `weight_radiance = 5.0` contributes a pure
  additive constant to every candidate. This is also what hid defect 4 for a while.
- **`domain`** — 896 of 930 are `chat`, so `cross_domain_repulsion` has almost nothing to act on.

Cosine and BM25 are the only live channels. That is the "half and half" exactly.

## Also decided: Unicode PUA stays off the bridge

`quant_cosine = 0.332` on this corpus. The niodv4 centroids have L2 norm ≈ **0.42** against
SplatRAG's unit-norm semantics; the codebook's own report claims `mean_quantization_error 0.0695`,
but that was measured in niodv4's scale on niodv4's rollouts. The PUA char round-trips *exactly*
(so the G1 test passes as written) while carrying a vector at cosine 0.33 to the original — a
256-way bucket label, not a memory. Codebook **not** retrained, per the G1 STOP. Just not used as
transport. Raw floats / b64 instead.

---

## The metric switches at the joint — cosine vs Euclidean

Raised by Jason from memory ("we're supposed to use euclidean to steer memories"), then checked.
He is right about the half that matters and wrong about the half that doesn't, and the distinction is
load-bearing for the bridge.

**On unit-normalized vectors, Euclidean and cosine are the same ordering.** `L2² = 2 − 2·cos`,
verified exactly across random 64-d pairs:

```
cos=+0.1693  L2=1.2889   2-2cos=1.6613  L2²=1.6613
cos=-0.1252  L2=1.5001   2-2cos=2.2504  L2²=2.2504
```

SplatRAG semantics are L2-normalized, so switching the picker to Euclidean would change **no
output**. That is very likely why an old effort to "kill cosine" never converged: on the unit sphere
there was nothing to kill.

**On un-normalized vectors they rank-invert.** Same test, vectors at norms 40 / 141 / 300 against a
141-norm query:

```
norm= 40  cos=-0.0194  L2=147.31   <- lowest cosine, SMALLEST distance
norm=300  cos=+0.1136  L2=316.65   <- highest cosine, LARGEST distance
```

Magnitude carries information cosine discards by construction. And hydro is entirely Euclidean:
`center_l2: 141.2`, `nearest_L2 ~180`, `nearest_L2 ≈ 0` — its memory-coupling criteria are all L2 on
residuals whose norm is meaningful.

So the two systems use different metrics in the places where the difference is real. SplatRAG ranks
on a sphere where norm is discarded; hydro deposits scars where norm is load-bearing. When hydro
re-embeds a pick's text, that vector's magnitude matters to its force law and SplatRAG never had an
opinion about it. This is why `docs/BRIDGE_SPLATRAG_PICK.md` §5 makes `--picks-dry-run` check the
embedded norm against existing scar `center_l2` before any generation run.

### Which also explains two constants measured earlier today

niodv4's own deep-research report already flagged the mechanism:

> treat the latent dynamics (the "vortex / toroidal / flow" substrate and TEDE-style corrections) as
> first-class citizens. Your own repo notes that earlier "flat energy" plots were misleading because
> repeated renormalization forces a unit-norm vector's mean squared value toward [a constant]
>
> — `niodv4/offsite-artifcats/deep-research-report.md:11`

SplatRAG renormalizes at every stage: `EmbeddingClient::embed_batch`, `matryoshka64`, and the output
of `apply_steering`. Measured consequence on the live store: **`radiance` is 4.5 on all 930 splats
and `mass` is 4.4965 on all 930 splats.** Constant energy across every memory. Recorded earlier in
this log as "radiance cannot discriminate" — that was the symptom. The cause is that magnitude is
normalized away at every step, so there is nothing left for a per-memory energy to vary over.

Not changed here. Removing a normalization step changes what every existing vector means, and the
store, the HNSW index and the dream tuning all assume the unit sphere. Recorded so the next person
does not rediscover flat energy and call it a physics result.

## Pre-registered, before running anything

Stated up front so the next round can falsify them rather than confirm them afterwards. The four
defects above were found *while reading and running*, not predicted — no credit claimed for those.

1. A low-confidence pick injected at full strength hurts generation more than the same pick injected
   weakly.
2. With total α held constant, loading more memories does **not** degrade output merely for being
   more numerous. If it does, the cause is dilution of direction, not steering magnitude.
3. A *wrong* memory (high confidence, different topic) measurably hurts. If it does not, the pipe is
   not coupled and no amount of prose changes that.
4. If a normalization step is removed anywhere in the chain, per-memory `radiance` and `mass` stop
   being constants. If they stay constant, repeated renormalization was not the cause of flat energy
   and that explanation should be dropped.

Prediction 3 is the one that matters. It is the three-arm test — none / same-topic / different-topic
— and it is the difference between memory coupling and a rig that runs.

---

## Surfaces added

- `splatrag pick <prompt> [--limit N] [--budget α] [--min-score S] [--text-budget C] [--out FILE]`
- `MemoryService::pick` — over-fetches to a fixed depth so confidence never tracks `limit`
- `src/pick.rs` — `MemoryPickSet`, `Pick`, `NullModel`, `relevance`, `separation`, `budget_shares`

Not done: `POST /api/pick`, the hydro-side loader (spec only, `docs/BRIDGE_SPLATRAG_PICK.md`), the
three-arm harness, OCR over the 789 pending assets.

---

— Claude (Opus 5), with Jason Van Pham
