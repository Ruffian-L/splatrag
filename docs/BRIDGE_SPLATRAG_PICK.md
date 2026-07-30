# Bridge: SplatRAG `pick` → hydrodynamic-swarm

Spec for the hydro side. **No hydro files were edited** — its working tree has ~20 uncommitted files
including `vendor/`. This describes what to apply, for Grok or Jason to land.

Produce the input with:

```bash
splatrag pick "Explain the Physics of Friendship in one short paragraph." \
  --limit 3 --out data/pick.json
```

---

## 1. The core rule: embed the text, never the floats

A pick carries `text`. Hydro must call its own encoder on it:

```rust
// hydro src/concourse/embed/gemma.rs:149
let mu = model.embed(&pick.text)?;   // -> [hidden_dim], hydro's own space
```

`semantics_64` is **telemetry and dedup only**. It is a 64-d unit vector from Qwen3-Embedding-8B;
hydro's residual space is 2560-d, un-normalized, scars at L2 ≈ 141. Writing those 64 floats into a
residual will not error and will not look wrong. It will steer on noise.

Refuse the file unless you can satisfy yourself of the source:

```rust
if set.source_dim != 64 || !set.source_embedder.starts_with("Qwen3-Embedding") {
    bail!("unknown pick provenance: {} @ {}", set.source_embedder, set.source_dim);
}
```

This is the same instinct as hydro's existing `model_dim` / `model_fp` header check. Keep it.

## 2. Deposit as a scar, reusing the prefill-bridge path

Do not add a force law. The prefill-bridge scar path already lands `nearest_L2 ≈ 0`
(`research_logs/2026-07-16_prefill-bridge-scar.md`), which is what a loaded memory should look like.

| `Splat` field | from the pick |
|---|---|
| `mu` | `model.embed(&pick.text)` |
| `alpha` | signed by `gain` (or `suggested_gain`); sign is the pleasure/pain axis |
| `sigma` | existing scar default — the pick has no opinion about basin width |
| `lambda` | existing default; `0.0` if you want the memory to be an anchor |
| `is_anchor` | `true` only if Jason says a pick is a core fact |

`mass < 0` on a pick means **repel**. Keep that on the physics side, and keep it separate from
`gain` — negative gain inverts *meaning*, negative mass repels in the *field*. They have been glued
together before; do not glue them again.

If `field_logit_alpha` is enabled (`hydro src/config.rs:71`), the unit direction of the deposited
scar is the `û_g` that knob expects.

## 3. Which α to apply

Two fields, deliberately not merged:

- **`gain`** — an α deliberately set on that memory by `splatrag steer`. Authoritative. `0.0` means
  nobody has steered it.
- **`suggested_gain`** — this picker's *unvalidated* proposal: the pick's share of a total budget,
  scaled by `confidence`. Ignore it freely.

`total_suggested_gain` is the sum across the set and is bounded by the budget, so **loading more
memories divides the push rather than stacking it**. Report it in any multi-memory run — otherwise a
"more memories" result is confounded by "more total push". Three memories at full α would be 1.05,
well past the 0.40 collapse onset.

## 4. Read `confidence`, not `separation`

`confidence` is a null-model z-score: how far the top memory's cosine sits above what a *random*
memory scores for the same query. That is the number to gate on.

`separation` (gap between the top two hits) is present but is a **diagnostic**. It was tried as the
confidence signal and measured inverted — gibberish scored 0.751, a well-covered topic scored 0.016.
Details in `research_logs/2026-07-29_the_picker.md`. Do not gate steering on it.

Useful combination: **low separation + high confidence** means several memories are jointly relevant
— the multi-memory case worth studying.

## 5. Suggested CLI shape

```text
--import-picks data/pick.json     # deposit each pick as a scar
--picks-max-gain 0.35             # local ceiling regardless of what the file proposes
--picks-dry-run                   # log mu norms, alphas and ids; deposit nothing
```

`--picks-dry-run` first. The one thing worth checking before any generation run is that
`model.embed(pick.text)` produces a vector whose norm is in family with existing scar `center_l2`
(≈141 on the current store). If it is off by an order of magnitude, the encoder or pooling differs
from what wrote those scars, and nothing downstream will mean anything.

## 6. What this bridge is not

- Not a 64→2560 codec. That is the parked RAVE lane; this design exists so it is not needed.
- Not a Unicode/PUA transport. `quant_cosine 0.332` on this corpus — see the changelog.
- Not a claim that memory coupling works. It is a pipe. Prediction 3 in the changelog is the test.

---

— Claude (Opus 5), with Jason Van Pham
