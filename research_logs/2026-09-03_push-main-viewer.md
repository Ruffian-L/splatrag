# 2026-09-03 — Push main + viewer

## Frame

Jason said push two git trees. This is not a compact-lexical, not a serve restart, not a store rewrite.

## What landed (engine)

Scoped to this session, on `splatrag-clean` `main`:

- `src/lookup.rs` — cold row is the lookup document; `lookup_text` strips thread-start harness noise; speaker is a query filter
- `src/lexical.rs` — indexes `lookup_text`, skips `!is_indexable`
- `src/dedupe.rs` — byte-identical mute plus near-dupe prefix mute (first 160 normalized chars, group ≥ 8)
- `src/retrieval.rs` — recall skips `hot.muted` (kept HEAD `context_for` signature)
- `src/sidecar.rs` + `config.sidecar_path` + `GET|PUT /api/sidecar/:id` — belief, PassThrough default
- ingest test: glued `User:\nAssistant:` is illegal
- chunk-geometry A/B + n=250 series scripts and receipts

Left uncommitted on purpose: `service.rs`/`web.rs`/`config.rs` auth token + ingest-roots + hydrate WIP, `src/hydrate.rs`, `vendor/`, CLAIMS.md, assets, CI.

## GitHub

`origin` = `git@github.com:Ruffian-L/splatrag.git`. Remote only had `memory-ingest` (July 30). Push creates `origin/main`. No force. `memory-ingest` stays.

Viewer `splatrag-v3-viewer` had no remote. New repo `Ruffian-L/splatrag-v3-viewer`. Did not commit `field.json` / `logs/` / dumps.

## We think

The two trees Jason asked for are on git. Derived indexes on the 128k personal store still have the judge heads until `compact-lexical`.

## Next

New binary, then `compact-lexical` if he wants BM25 to see the skip.
