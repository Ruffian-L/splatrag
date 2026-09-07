# Ultima open-domain chats

> Date: 2026-08-22
> Agent: Grok (xAI)
> Scope: historical measured run; not a current release guarantee

## What was tested

The filtered SciFact freeze (`domain=scifact`) measured R@10 **0.8804** and
nDCG@10 **0.7624**. Ultima removed that retrieval filter, kept SciFact qrels,
and measured whether colocated non-SciFact memory could take top-10 slots.

The mixed store contained SciFact, 5,000 Urban Dictionary rows, and 60,000
personal / Grok / Gemini chat rows transplanted from the personal hot store
(seed 42). It contained 70,183 cold records, HNSW keys, and hot splats. The
chat rows were not re-embedded; the full-dimension reranker embedded only its
top-30 candidates at query time.

## Results

| Window | R@10 | nDCG@10 | Mean non-SciFact slots | Queries displaced |
| --- | ---: | ---: | ---: | ---: |
| filtered SciFact freeze | 0.8804 | 0.7624 | n/a | n/a |
| SciFact + Urban, filter off | 0.8821 | 0.7630 | 0.07 Urban | 16 / 300 |
| + 60k chats, filter off | 0.8636 | 0.7354 | 0.77 chat | 96 / 300 |

The 60k chat mix lost 0.0169 Recall@10 versus the filtered freeze (inside the
0.05 filtered-handshake allowance) and 0.0270 nDCG@10. That is open-domain
retrieval interference, not a claim that the 5,000-row Urban poison handshake
passed or failed.

## Reproduction boundary

Run the current open-domain evaluator with:

```bash
cargo run --release -- --config splatrag.toml ultima \
  --dataset /path/to/scifact --k 10 --out ultima.json
```

This historical run used a frozen store and configuration. A new result needs
its own receipt; do not compare a `--limit` sample to the full 300-query table.

Private source log and original receipt directory remain on the research
machine: `research/logs/2026-08-22_ultima-open-domain-chats.md` and
`receipts/ultima-20260822/`.
