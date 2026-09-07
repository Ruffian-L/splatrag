# Chunk geometry series n=250

> Date: 2026-09-03
>
> **Agent:** Grok (xAI)
>
> Series, not one 9-pair cell. No HIT/MISS. Queries are verbatim user/assistant text (not paraphrases). Post-hoc read of the JSON.

## Sample

250 adjacent human→assistant pairs from personal cold log. Seed `20260903`. 170 conversations; 80 contributed 2 pairs. Pool size 250. Embedder `:8081` 4096-d. Store not mutated.

## Battery

| id | what | number (mean / median / p90) |
|---|---|---|
| T1 | concat closer to user than assistant is (Δ cosine) | **+0.226** / 0.222 / 0.369 |
| T2 | dense rank, user query, **assistant** pool | **32.4 / 9 / 99** of 250 |
| T3 | dense rank, user query, **concat** pool | 11.2 / **1** / 40 |
| T4 | dense rank, user query, **user-row** pool | **1 / 1 / 1** |
| T5 | dense rank, assistant query, assistant pool | 1 / 1 / 1 (query=doc) |
| T5 | dense rank, assistant query, concat pool | 1.74 / 1 / 2 |
| T6 | lex rank, user query, assistant docs | **55.7 / 38 / 147** |
| T6 | lex rank, user query, concat docs | 10.3 / **1** / 40 |
| T6 | lex rank, user query, user docs | 1.008 / **1** / 1 |
| T7 | convo intra−inter cosine concat | 0.202 |
| T7 | convo intra−inter cosine assistant | 0.184 |

## Lookup: metadata or not

If the user turn is **not searchable** (not in the embed, not in BM25 docs) you get T2 and T6-assistant: median rank 9 dense, **38** lexical. That is a miss on “what I asked.”

Concat fixes that (T3 / T6-concat median 1) by putting the question in the assistant vector (T1 +0.23). That is the ONE-chunk tax.

User as its **own row** (current ingest) is T4 / T6-user: rank 1 without moving the assistant splat.

So: lookup needs user text in an **index**. Sidecar that is never searched is not an index. Own row or BM25-on-metadata. Not glue.

## Hole

Verbatim queries. A paraphrase cell is not in this file.

## Files

- runner `scripts/chunk_geometry_series.py`
- receipt `research_logs/2026-09-03_chunk-geometry-series.json`
