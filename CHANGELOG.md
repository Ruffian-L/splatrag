# Changelog

User-facing changes to the SplatRAG engine. Lab notes stay local.

## 3.0.0 — 2026-08-28

Engine files that the 3.0.0 README already named are now in git: `claim.sh`, `run`, CUDA dream, remaining `src/`, template configs, poison corpus. Machine systemd units stay local. Still no GitHub public flip.

## 3.0.0 — 2026-08-27

First v3 engine release.

- Hybrid recall: BM25 + 64-d HNSW, reciprocal rank fusion, optional 4096-d rerank as a third RRF arm.
- Gaussian-splat field with 64-d basin discovery (`basin_core_cosine` packs wells; 3-d is PCA of the same 64-d vectors).
- Append-only cold log. Dream mutates hot geometry only.
- Picker emits addresses (`memory_id` + basin) under one gain budget.
- MCP stdio + HTTP tool surface. Optional CUDA dream backend (`physics.backend = auto|cpu|cuda`).
- Reproducible SciFact handshake: `./claim.sh`.

**Eval baseline.** Full 300 SciFact test queries. The older 0.7822 nDCG@10 is the first-50 slice of SplatRagBench (Nomic). Filtered full-300 on this engine (RRF k=10, BM25×1.3, 4096-d arm on fusion top-30): R@10 0.8804 / nDCG@10 0.7624. That is the number to beat.

Architecture was not changed for this release. Role diagnostics (`splatrag diagnose-basin`, `splatrag diagnose-roles`) score 64-d candidate generation vs 4096-d discrimination vs fusion without retuning.
