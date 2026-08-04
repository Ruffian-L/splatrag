# 2026-07-30 — Routes already walked

Not a postmortem. A map, so nobody spends an afternoon re-deriving these. Every one of them looked
right on paper and was settled by measuring instead of arguing.

---

## 1. Filling in "missing" cosine scores makes retrieval worse

**The reasoning that looks right:** in `retrieval.rs`, a candidate found by BM25 alone has no entry
in `cosine_scores`, so it scores `0.0`. At `weight_cosine = 10.0` that is a ~7-point penalty against
a true similarity that runs ~0.7 on this corpus. It looks like unmeasured is being confused with
orthogonal, and the exact cosine is a cheap 64-d dot product against semantics already in memory.

**Measured on SciFact, identical store, 300 queries:**

| | recall@10 | nDCG@10 |
|---|---|---|
| leave the 0.0 | **0.8518** | **0.7257** |
| compute the real cosine | 0.8354 | 0.7142 |

Filling it in costs 1.6 points of recall and 1.2 of nDCG.

**Why:** the absence is information. The ANN declining to return a candidate is evidence *about* that
candidate. Overwriting it with a plausible ~0.7 promotes weak lexical matches into the top-k.

**Settled:** leave it. The comment in `retrieval.rs` carries the numbers.

---

## 2. You cannot recover memory "mass" by not normalizing

**The reasoning that looks right:** every vector is L2-normalized, so magnitude is discarded and a
4,000-word memory weighs the same as a one-liner. `radiance` is 4.5 on all 10,051 splats and `mass`
is 4.4965 on all of them — flat energy, exactly what the niodv4 deep-research report warned about
("repeated renormalization forces a unit-norm vector's mean squared value toward a constant"). Stop
normalizing and the field gets mass.

**First measurement — where normalization actually happens:**

```
  chars  full-4096 norm
     11         1.00000
    198         1.00000
  16453         1.00000
```

llama-server returns unit vectors already. `--embd-normalize` defaults to `2`, which *is* Euclidean.
SplatRAG's own `normalize()` call is a no-op; the magnitude was gone upstream, in llama.cpp, before
this codebase saw it. (Worth noting for the archaeology: earlier versions of this line of work used a
hand-rolled Qwen embedding loader, not llama.cpp, so this default would not have applied then.)

**Second measurement — is there anything under it?** Re-ran with `--embd-normalize -1` over 150 real
memories:

```
  min 117.5   median 131.1   max 136.9   stdev 3.6
  spread (max/min): 1.16x
  corr(norm, text length) = +0.113
```

The raw norm is an artifact. Everything sits within 16% of everything else and length explains
none of it. Qwen3-Embedding is trained against cosine objectives; it does not encode weight in its
magnitude.

**Settled, with a consequence:** mass has to be **computed**, not extracted. Nothing about what a
memory *is* tells you what it *weighs* — which is the gain/mass split holding up under test rather
than being violated by it. Candidates that would actually vary per memory: recall frequency,
connectivity (`topology.rs` already computes persistence intervals), deliberate steering, decay.

---

## 3. Euclidean vs cosine is not a live choice on this embedder

`fast-hnsw` ships `SquaredEuclidean`, `Euclidean` and `Cosine`; SplatRAG uses `Cosine`.

On unit vectors the two are the same ordering — `L2² = 2 − 2·cos`, verified exactly. And per §2 the
raw vectors are within 1.16× of unit anyway, so the equivalence survives even without normalization.
Swapping the metric would change nothing measurable.

Where Euclidean genuinely differs is **hydrodynamic-swarm**, whose scars are un-normalized residuals
(`center_l2 ≈ 141`, `nearest_L2` as the coupling metric). There, magnitude is real and carries
information. Two systems, two regimes — the distinction is about what the vectors *are*, not a
preference.

---

## 4. The SciFact contract floors were never measured here

`TARGET_RECALL_AT_10 = 0.88` and `TARGET_NDCG_AT_10 = 0.75` entered in commit `37f4858`, whose own
message reads "Codex build, unverified". Current measurement is 0.8518 / 0.7257, so `handshake`
reports `passed: false` on the absolute floors.

The poison contract, which is the part this evaluation exists for, passes decisively — see below.
Open question for Jason: are those floors a real target, or numbers that rode along from
SplatRagBench?

---

## What did hold

The poison handshake. 5,000 Urban Dictionary entries injected into a 5,183-document SciFact store,
followed by a full dream cycle:

```
baseline        recall@10 0.8388   nDCG@10 0.7085
after poison    recall@10 0.8354   nDCG@10 0.7142
recall_drop     0.0033             (allowance 0.05)
cold_preserved_during_dream: true
```

A 0.33% dip from a 5,000-document contamination, and nDCG went up. The cold log came back
byte-identical — the dream moved geometry and deleted nothing.

Measured with the §1 regression still in place, so the real resistance is at least this good.

---

— Claude (Opus 5), with Jason Van Pham
