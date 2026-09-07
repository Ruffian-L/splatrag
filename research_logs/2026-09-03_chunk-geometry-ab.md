# Chunk geometry A/B — concat vs assistant-only

> Date: 2026-09-03
>
> **Agent:** Grok (xAI)
>
> This is a geometry probe of two chunkings. Not a language judge. No win-rate.

## Arms

| arm | embed | user text |
|---|---|---|
| A concat | `User: …\nAssistant: …` | inside the vector |
| B assistant-only | assistant body | metadata only, not embedded |

Topic labels (towel / niodoo / desktop) were analysis-only. They did not go into the embedder.

9 frozen pairs: `research_logs/2026-09-03_chunk-geometry-pairs.json`.
Runner: `scripts/chunk_geometry_ab.py`. Embedder `:8081` Qwen3 4096-d. Personal store not written.

## Cluster receipt (topic intra cosine − inter cosine)

| cloud | intra | inter | intra−inter |
|---|---|---|---|
| A concat | 0.529 | 0.241 | **0.289** |
| B assistant-only | 0.514 | 0.270 | **0.243** |
| user-only (control) | 0.579 | 0.322 | 0.257 |

n_intra_pairs=9, n_inter_pairs=27. Toy. Not the 1,723 live basins.

## Own-pair user→chunk (from rows)

Concat sits closer to the user than assistant-only does on every niodoo row (e.g. niodoo-1 0.632 vs 0.409; niodoo-2 0.904 vs 0.571). Towel rows are already close both ways (parallel-drying answers reuse the question). Desktop-3 (store 401) concat 0.70-ish vs assistant — see JSON.

User-query to *other* chunks, same topic vs other topic: concat 0.539 vs 0.273; assistant-only 0.546 vs 0.290.

## What this is not

No 8/10. No basin list on the 128k store. No ingest of arm B. Jason grades whether 0.289 vs 0.243 on nine pairs is a reason to change ingest.

## Next

Same script, more pairs, or a scratch silo that actually stores B and `list_basins`.
