# Durable lookup index

> Date: 2026-09-03
>
> **Agent:** Grok (xAI)

## Who owns it

| layer | job |
|---|---|
| cold `memories.jsonl` | durable truth, one turn one row |
| tantivy + HNSW | derived from `record.text`, rebuild = replay cold |
| sidecar | belief, not lookup text |
| speaker | query filter, not an index skip |

`src/lookup.rs`: `is_indexable` / `lookup_text`. Human phrasing hits the human row after rebuild.

n=250 series: user-query vs assistant-only lex median rank 38/250; vs user-row rank 1.
