# Epistemic sidecar — belief is not evidence

> Date: 2026-09-03
>
> **Agent:** Grok (xAI)

## Context

Jason: Gemini/ChatGPT jazz on separating splat geometry from policy. The public slogan they landed: the embedding answers *what it says*; the sidecar answers *whether it should be believed*. Trust must never re-enter the embedder.

This tree already embeds `record.text` only (`service.rs` `embed_slices`). Cold is append-only. What was missing was a first-class belief store and an explicit policy, not a JSON bag concatenated into the chunk.

## What landed

`src/sidecar.rs`

- `SidecarState`: authority ∈ [0,1], status, superseded_by, valid_from/until
- `EpistemicPolicy`: PassThrough (default), CurrentProduction, HistoricalResearch
- `decide` → Retain / Penalize(v) / Drop. Retract ≠ low authority.
- `SidecarStore` at `data_dir/sidecar/epistemic.json` (sparse; missing = default admissible)
- `MemoryService::sidecar_put` does not embed and does not rewrite cold text
- `GET|PUT /api/sidecar/:id`
- `inspect` includes sidecar
- `recall` runs `filter_hits` after hybrid. PassThrough is a no-op on ranking.

Gate: `embed_payload(text) == text`. Sidecar write leaves `content_sha256` unchanged.

## What this is not

Not GraphRAG (content-derived entity graph). This graph is operational: what the *system* currently believes about the *corpus*. Default policy does not touch SciFact freeze numbers.

## Next

Scratch silo + CurrentProduction, then a dated analysis pass. Do not enable on personal as unpaid ranking change.
