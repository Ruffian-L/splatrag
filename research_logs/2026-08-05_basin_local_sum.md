# Basin local-sum soft neighborhood (2026-08-05)

Geometry-only reweight on hybrid retrieval: sum dense scores among candidates near top dense hits
(`basin_local_sum`, threshold 0.50, weight 0.3). Not domain stamps; not fusion-grid re-runs alone.

## SciFact full-300 k=10 (Qwen3-Embedding-8B)

| Stage | R@10 | nDCG@10 |
| --- | ---: | ---: |
| Prior A0 after-poison | 0.8518 | 0.7257 |
| Local-sum on poisoned field | 0.8551 | 0.7298 |
| Fresh handshake unpoisoned | 0.8634 | 0.7316 |
| Fresh handshake after Urban + dream | 0.8568 | 0.7303 |
| Recall drop | 0.0067 | (≤0.05 hold) |

Code: `src/basin_score.rs`, wired in `src/retrieval.rs`, knobs in `src/config.rs`.
Example: `examples/splatrag.basin-local-sum.example.toml`.
