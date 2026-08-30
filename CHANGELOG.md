# Changelog

User-facing changes to the SplatRAG engine. Lab notes stay local.

## 2026-08-30 — Clear saturated embed queue and restore MCP recall

We did: restarted `splatrag-embed.service` and `splatrag-personal-serve.service`
after `ensure_ready` hung for more than a minute behind multi-thousand-token
embedding jobs. Verified embed health (`HTTP 200`), personal store status
(`124,753` cold records, `124,066` HNSW keys/splats, `1,723` basins), and a real
MCP `recall` completing in 8.8 seconds. Did not change code, indexes, memories,
dream state, or model configuration.

We think: the services were alive but their shared embed queue was saturated;
clearing that queue restored the gateway without a data repair.

Next: identify which caller submits multi-thousand-token embedding requests and
bound or chunk those requests before they can monopolize all four embed slots.

Research: `research_logs/2026-08-30_clear-saturated-embed-queue.md`

## 2026-08-29 — Controlled A/B: basin cap + mutual-kNN topology

We did: froze a snapshot of the live personal store (`/tmp/splatrag-snapshot-aug29-1150`), built `diagnose_topology` in both the baseline (`adb93b0`) and modified topology, and ran both on identical bytes (87,165 active splats, 129,976-key ANN graph). Deterministic across re-runs.

We think: the cap/bisection bounds the largest basin (6,190 → 4,657, basins 140 → 1,507) but does NOT grow assignment coverage — it loses ~1,150 assignments on the controlled comparison (82,152 → 81,002, 94.25% → 92.93%) because splitting overloaded basins orphanes border members. The earlier 80,988 "preserved assignments" figure was contamination: it compared against a live, growing store.

Next: decide whether the 1.3 pp assignment loss is acceptable in exchange for bounded basins, or re-attach bisection-orphaned border splats in a final staged pass and re-run this frozen A/B.

Research: `research/logs/2026-08-29_controlled-a-b-of-basin-cap-and-mutual-knn-topology.md`
Agent: Codex

## 2026-08-28 — Split the rust well via basin_max_members

- Changed basin_max_members from 0 to 5000 in splatrag.toml to axis-split the single massive well. Restarted the engine and triggered a dream to un-collapse the visually unreadable 40k-member rust supernova.
- Hypothesis: Restricting maximum basin size prevents the entire memory space from clustering into a single giant node, making the visualization readable again.
- Research: `research/logs/2026-08-28_split-the-rust-well-via-basin-max-members.md`
- Agent: Antigravity

## 2026-08-28 — CUDA and memory-path performance audit

We did: removed whole-store per-query clones and other transient allocation spikes, moved read-mostly state to lock-free `ArcSwap` snapshots, reused ANN neighborhoods, streamed cold/hot loads, and reduced CUDA step synchronization. Audited the legacy Niodoo GPU tree and retained only its safe context-reuse idea. Verified 153 tests, CPU-only compilation, and both CUDA force modes on the GB10. Two read-only live-store startups took 5.77–7.37 s at about 1.303 GiB peak RSS while the store was actively growing. Did not hydrate indexes, dream the real store, merge data, train, or run a model eval. Research: `research/logs/2026-08-28_cuda-and-memory-path-performance-audit.md`.

We think: lock-free immutable `ArcSwap` snapshots and bounded batch allocations remove concurrency-shaped RSS growth and read-lock contention; sparse HNSW topology remains safer than porting legacy quadratic Rips/LoPHAT GPU code.

Next: profile concurrent recall on the live corpus after its newly observed HNSW drift (latest `cold=120859`, `hnsw=807`) is repaired in a separate, explicit data operation.

## 2026-08-28 — Fixing derived index drift via out-of-cgroup hydrate-ann

- Fixed index drift where `hnsw_keys` exceeded `cold_records` by stopping the background `serve` process and hydrating the index using `systemd-run --user --scope` to bypass the agent cgroup memory limit.
- Verified `doctor` passes correctly with 120,084 records.
- Research: `research/logs/2026-08-28_fixing-derived-index-drift-via-out-of-cgroup-hydrate-ann.md`

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
