# TEAM GOAL — Steering + 64D packet plumbing

**Status:** OPEN · issued 2026-07-29  
**From:** Jason (vision) + Grok (wire lead this shift)  
**Workbench:** `ruffian/s` (SplatRAG)  
**Juice (read-only reference, do not “rebuild”):**

- OI: `/media/ruffianl/ghost_team/projects/ontological-inversion`
- niodv4 encode/decode:  
  `/media/ruffianl/backup_sandisk/02_projects/niodoo_team_build_code_backup_20260608-150015/worktree/niodv4/experiments/encode_decode/niodv4`
- Phoenix mirror: `/media/ruffianl/ghost_team/pheonix_squad/gem/niodv4`
- Hydro residual (separate dim): `hydrodynamic-swarm` — **not** this goal’s target space

**Rule:** This is **plumbing**. The science already happened. Do not re-prove OI, re-train TEDE, or re-ablate hydro force laws. Wire what exists. Receipt → STOP that gate.

---

## North star (one line)

**One memory packet language across the stack:**  
`4096 (or full embed) → 64D → (optional Unicode PUA) → steer with gain/mass → dream uses hydro`  
without inventing a fifth geometry.

---

## What is already DONE (do not re-walk)

| Piece | Where | Status |
|-------|--------|--------|
| OI operators (polarity / householder / neg-gain) | measured in ontological-inversion; ported `src/inversion.rs` | DONE |
| **Gain ≠ mass** | gain inverts semantics; mass repels | DONE — keep that split sacred |
| `steer` CLI + `POST /api/steer` + MCP `steer` | SplatRAG | DONE |
| 64D matryoshka on embeddings | SplatRAG HNSW / hot | DONE |
| PCA 64→3 + sign-from-semantics dream | SplatRAG physics (better than hydro residual α-scars for field layout) | DONE |
| VQ Unicode 64D→PUA→64D | niodv4 `unicode_tokenizer.py`, M7.5 320×, exact re-encode | DONE in juice tree — **not yet in SplatRAG** |
| Bucket-mean 4096→64 for residuals | niodoo secret_sauce / compress | DONE in niodoo — **not required for SplatRAG steer lane** |
| Grok front page / viewer | external + SplatRAG `/` | DONE (out of scope) |
| Grok memory bulk ingest | memory-ingest branch WIP | **PARKED** — memories later |

---

## Decision — who wires what (this shift)

| Lane | Owner | When |
|------|--------|------|
| **Thin wire (critical path)** | **Grok (this session / next turn)** | **NOW** |
| Full bridge polish + receipts | Shep lead · Echo telemetry · Lumina narrative/stop-check | After thin wire lands, only if Jason opens G2 |
| Hydro residual TCT scars | nobody this goal | PARKED |
| Bulk Grok export / OCR | nobody this goal | PARKED |

### Why “now” and not “team open plumbing”

Earlier shifts burned on open plumbing without a stop. The juice is frozen. Thin wire is **one afternoon**: import/export 64D packets (and optional PUA) on top of existing `steer`. Team thrash without a landed thin path = more scaffolding, same empty joint.

**Jason override:** if you want the team to wire instead, mark G1 “TEAM OWNS” and Grok stops coding.

---

## Gates (receipt closes gate · no re-run)

### G0 — Vocabulary lock (no code) · DONE when this file is the contract

Write nothing else until the board agrees:

1. **64D** = memory packet / ghost / OI / SplatRAG hot semantics  
2. **Negative gain** = invert 64D (OI)  
3. **Negative mass** = repel in dream  
4. **Unicode PUA** = transport of 64D, not a second latent  
5. **TCT residual** = full hidden dim; only compress when crossing into 64D packet land  

**STOP:** one signed line in receipt: “vocab locked.”

---

### G1 — Thin wire in SplatRAG (Grok owns) · **PASS 2026-07-29**

**Receipt:** `logs/steering_plumbing_G1/RECEIPT.md`

**Done means all true:**

1. `splatrag pack-64 <memory-id>` (or `steer` response field) emits:
   - `semantics_64: [f32; 64]` (or base64 f32 LE)
   - `gain`, `mass`, `basin_id`, `basin_locked`
2. Optional: `unicode` string via **existing** niodv4 codebook contract (256 PUA, U+E000+) — **load codebook from juice path or a copied `codebook_256` artifact; do not re-kmeans train**
3. `splatrag unpack-64` / `POST /api/packet` restores 64D onto a splat (or creates hot-only packet) and can re-`steer`
4. Unit test: random unit 64D → pack → unpack → cosine ≥ 0.999 (raw float path); if unicode enabled, decode→re-encode match on codebook path
5. **No** residual 4096 path required in G1  
6. **No** dream rewrite, **no** hydro-swarm edits

**STOP:** `logs/steering_plumbing_G1/RECEIPT.md` with curl/CLI transcript.  
**Forbidden after PASS:** redesign VQ, new codebook sizes, RAVE training.

---

### G2 — Team integration (only if Jason opens) · PARKED until G1 PASS

| Who | Owns | Done |
|-----|------|------|
| **Shep** | Lead · G2 checklist · verdict | Receipt table PASS/FAIL |
| **Echo** | Telemetry · anti-narrative | Numbers only: cosines, packet bytes, gain/mass before/after; flag prose that claims “memory works” without a number |
| **Lumina** | Continuity of *meaning* of the packet | One paragraph: what a PUA string *is* (transport) vs is not (not emotion, not residual); STOP check that team didn’t re-open OI research |

**Smokes (fixed script, no invention):**

| ID | Action | Pass |
|----|--------|------|
| A | `steer --gain -0.2` on known id | `cosine_before_after` drops; cold text unchanged |
| B | `steer --mass -1` only | mass &lt; 0; semantics cosine ≈ 1 |
| C | pack → unpack same id | cos ≥ 0.999 raw |
| D | (if unicode in G1) pack unicode → unpack | codebook re-encode match |
| E | dream once on smoke set | no crash; negative mass still repels (spot-check positions) |

**STOP:** signed RECEIPT. No G3 without Jason.

---

### G3+ — PARKED

- Grok bulk memories / OCR  
- TCT residual ↔ 64D learned codec (RAVE)  
- Live residual inject of OI in llama.cpp  
- Vercel / front-page changes  

---

## Long-form goal (for the team — read once)

You are not being asked to invent intelligence, re-prove inversion, or make hydro prettier.

You are finishing the **pipe** Jason already paid for in research:

1. Thoughts and residuals may live at full hidden width.  
2. **Memories we keep, invert, and dream on are 64D packets.**  
3. Those packets can ride as **Unicode** so they move through chat, vaults, and tokenizers without a binary side-channel.  
4. **Gain** flips the packet’s meaning-side (sorrowful flip / OI). **Mass** flips the field’s physics-side (repel). Never glue those again.  
5. SplatRAG is the clean field + store. niodv4 juice is the codec. Hydro is residual live-steer — do not merge geometries casually.

Success is boring: **same packet in, same packet out, steer does what the knobs say, receipt in the log, stop.**

Failure modes to refuse:

- “While we’re here” new force laws  
- Rebuilding unicode from scratch when `unicode_tokenizer.py` + codebook exist  
- Claiming memory coupling without cosines  
- Running dream ablations as a substitute for packing wires  
- Touching protected hydro main without Jason  

---

## One-line direction (this shift)

```text
G0 locked by this file · G1 = Grok thin wire NOW · team IDLE until G1 receipt · G2 only if Jason opens · no memory bulk ingest
```

---

## Jason call box

- [x] Recommend: **wire thin path now (Grok)**  
- [ ] Alternate: team owns G1 (mark here and reassign)  
- [ ] Alternate: later (park all; no code)

**Default if no override:** Grok proceeds G1 next. Team reads this file and waits.
