# Changelog

This is a research repo. Not production unless Jason says so.

Pairing: every action here gets a **why**. Hypothesis form:

- We made this change. We think X will happen.
- Later: X did not happen, yet we found Y. Next we mutate Z.
- We mutated Z. Results matched. LFG.

Keep this file short. Longer writeups go in `research/logs/` (canonical; `research_logs/` is a pointer).
Local agent contract: `AGENTS.md` (gitignored). Helper: `scripts/pair_log.sh`.


## 2026-08-23 — top_z, the combine knob, and the live tree was July 30

- `top_z` is `Option<f32>`. Cosine 0.0 is retrieval's ANN-miss sentinel, not a similarity, so z-scoring it returned `(0-mean)/std` — a constant of the null, read as -3.9σ on two queries the picker got right. Filtered instead of coerced. No caller existed; `confidence` already had the lexical arm and was never wrong.
- ULTIMA #8's sweep is now code: `packet::combine` + `FieldCentroid` + `service::combine_packet` + `combine_64` MCP tool, `DEFAULT_COMBINE_T = 0.2`. Read-only — returns a packet and its receipt, never writes a splat. Live on 75,406 splats: t=0 reproduces naive exactly, t=1 exactly orthogonal, 0.6956 -> 0.6124 -> 0.5023 -> 0.0000. 146/146 tests.
- Live stack was running `engine/` = the July-30 tree, because `Hub/Projects/splatrag/splatrag-clean` was a symlink to a USB drive that dropped off the bus twice today. No CUDA kernels there, and no `fusion = "rrf"` — retrieval had silently been on linear fusion on a 75k store. docs-serve was at NRestarts=1804. Tree moved to nvme, symlink replaced with the real directory, unit repointed. Same store across the cutover: 76,251 / 85,459 / 75,406 / 283 / cycle 21.
- CUDA dream on the nvme binary: **17m36s** for 1200 steps (`CUDA force loop active=48229, sm_121`), cycle 21 -> 22, basins 283 -> 269, ke 211.2672 -> 145.2545. The CPU dream it replaced ran 40 min wall / 9h19m CPU across ~14 cores and never committed. Serve stayed up at ~half a core throughout.
- Hypothesis: a hardware fault demoted the whole system to three-week-old code and nothing reported it — the dream fell back to CPU quietly, retrieval fell back to linear fusion quietly, and the only symptom was "slow". Next "it got slow/dumb" starts with `ls -l /proc/<pid>/exe`, before any measurement.
- Did not rebuild on nvme (crates cache pruned 466/2114, network timing out — copied the identical-source binary). Did not touch the ghost_team tree. Did not merge, train, or write to any silo.
- Research: `research/logs/2026-08-23_top-z-combine-knob-and-the-live-tree.md`
- Agent: Claude (Opus 5)

## 2026-08-22 — Ultima open-domain chats

- Open-domain SciFact ultima: Urban is a register proxy (0.8821/0.7630, 0.07 slots). 60k personal chats are the near-domain poison — R 0.8636 / nDCG 0.7354, floors miss, dense arm leaks. Frozen knobs, no qrels in ranker.
- Hypothesis: UD barely displaces without the domain filter. Mixing 60k chats from the personal store drops SciFact R@10/nDCG vs 0.8804/0.7624 and shows which arm gives first (dense).
- Research: `research/logs/2026-08-22_ultima-open-domain-chats.md`
- Agent: Grok (xAI)

## 2026-08-22 — CUDA dream backend on the live door

- `physics.backend = auto|cpu|cuda`. sm_121 kernels: compact unmuted SoA, precomputed 32-NN dots, CUB cell-key sort, neighbor/full force, integrate, GPU reduce. Basin discovery stays CPU. Dream clones hot state and drops the write lock so `/api/status` and splat pages keep serving the last committed field.
- Hypothesis: personal-scale neighbor dream runs on GB10 without freezing Basin Field. CPU remains the reference.
- Later: `cargo test --lib cuda_matches` — 8-splat CPU vs CUDA at 1/10/50 full-pair and 10 neighbor-mode steps, positions within tolerance. Live personal settle started on the new binary; GPU SM ~41% on `splatrag`; SSE n=48068; step 290 / KE 454 while status still answered in 5ms. Not a frozen 50k matched factorial — no speedup claim vs the CPU 1,200-step run.
- Research: `research/logs/2026-08-22_cuda-dream-backend.md`
- Agent: Grok (xAI)

## 2026-08-22 — Visible Gaussian footprint

- Basin Field preserves the major Gaussian axis, floors the visible point box at 18px, NormalBlending, and applies live SSE frames onto a partial boot list. Why: 47k–48k valid unmuted splats were masking to ~3.4px under the basin graph, and live frames no-op’d until all ids loaded.
- Hypothesis: live memories read as anisotropic splats and walk during CUDA settle; dense wells do not additively burn white; graph remains.
- Research: `research/logs/2026-08-22_visible-gaussian-footprint.md`
- Agent: Grok (xAI)

## 2026-08-22 — CUDA path for personal-scale dream

- Current 84,904-row neighbor dream is CPU-only Rayon and still running after >1 hour. Each of up to 1,200 steps rebuilds a CPU spatial hash/adjacency, evaluates 64-d pair forces, then integrates on CPU. Local GB10 is compute 12.1; CUDA 13.3 `nvcc` supports `sm_121`.
- Hypothesis: compact unmuted splats into GPU-resident SoA, precompute the fixed 32-NN semantic edges/weights, and run grid build + halo force + integrate + settle reductions on CUDA. Keep basin discovery and persistence on CPU. No speedup claim until a frozen CPU×CUDA matched run.
- Research: `research/logs/2026-08-22_cuda-dream-acceleration-plan.md`
- Agent: Sol (OpenAI)


## 2026-08-22 — MCP and live splat visibility diagnosis

- Personal MCP returned current structured status and a relevant full-record recall. Basin Field received valid live geometry. Its 12px point-box floor still masks the default `[1.35, 0.38, 0.38]` Gaussian down to a ~3.4px visible core, so dark splats disappear beneath the clearer basin graph.
- Hypothesis: retrieval and transport are healthy; the remaining missing-splat report is a viewer footprint/contrast bug. Keep the graph and size the visible Gaussian, not only its containing point box.
- Later: frontend HTML stayed healthy (200 in 0.09s), but engine status timed out at 5s while the 47,680-splat neighbor-force dream still held the live-data door. The current load failure is the in-flight settle, not a dead Vite process.
- Re-check: settle still active; no completion line. Frontend 200 in 0.06s, engine status timed out at 10s, serve process remained CPU-busy. Did not interrupt it.
- Research: `research/logs/2026-08-22_mcp-and-splat-visibility-diagnosis.md`
- Agent: Sol (OpenAI)


## 2026-08-22 — GET slash goes to Basin Field

- GET / on :8767 307s to Basin Field :8080. Live SSE frames now emit unmuted splats only so the picture can move them. HUD Lab no longer says theater.
- Hypothesis: Opening :8767 in a browser lands on the product picture, not leftover assets.
- Research: `research/logs/2026-08-22_get-slash-goes-to-basin-field.md`
- Agent: Grok (xAI)

## 2026-08-21 — Twenty implementers landed

- Second swarm implemented, did not write hive boards. Engine: MCP pick inherits silo [pick]; pick basin-only first hit of a room; steer lock/assign on batch; unpack mutation.lock; recall basin inject; HTTP drop switch_silo; CLI --lock on batch. Picture: universeFor(n), point size cbrt, query-hit net, source=live, Walk path + Settle client (not fired). Ops: docs :8769 started not enabled; 3 stuck grok files ingested; watcher no longer stamps seen_mtime on fail; setsid embed; sp refuses palace occupant; telemetry HTTP ingest behind env. cargo test --lib 137 passed. Personal serve restarted onto new binary.
- Hypothesis: Code on the live door beats another catalog. Personal [pick] hybrid stays off so SciFact/personal ranking does not jump.
- Research: `research/logs/2026-08-21_twenty-implementers-landed.md`
- Agent: Grok (xAI)

## 2026-08-21 — Telemetry HTTP ingest with CLI fallback

- Telemetry watcher POST /api/ingest like cli_sessions, CLI fallback if URL unset/refused. Watcher stays CLI unless SPLATRAG_TELEMETRY_API_URL is set. Never :8767 personal. Did not start :8768.
- Hypothesis: Watch default stays sidecar CLI (no segment explosion change until TELEMETRY URL is set). HTTP path is live when that env is set; refused POST falls back to CLI.
- Research: `research/logs/2026-08-21_telemetry-http-ingest-with-cli-fallback.md`
- Agent: Grok (xAI)

## 2026-08-21 — Requeue three grok sessions onto personal HTTP ingest

- Three grok chat_history.jsonl files (01a0205d, 01a01f74, 01a01f6f) were discover()ed, failed ingest, and stamped in seen_mtime so the watcher never retried. POSTed the same watcher path (agent-jsonl domain=chat) to the running personal serve :8767 without --force and without stopping the watcher. 59 appended; three ledger rows; 28/27/4 cold ids.
- Hypothesis: HTTP ingest on the live serve will ledger those three paths and write 59 chat rows; watcher stays up.
- Research: `research/logs/2026-08-21_requeue-three-grok-sessions-onto-personal-http-ingest.md`
- Agent: Grok (xAI)

## 2026-08-21 — Queen-bee swarm

- Twenty worker bees mapped v3. Queen landed Basin Field Goal 1 HUD copy (Physics→View, drop unread Gravity) and repaired INDEX so the seven 2026-08-21 hybrid logs are catalogued. Hive receipts under research/receipts/2026-08-21/queen-bee-swarm/.
- Hypothesis: A 20-bee parallel audit plus one HUD brick beats sequential leftover-UI wandering. INDEX catalog lie was erasing paired hybrid work.
- Research: `research/logs/2026-08-21_queen-bee-swarm.md`
- Agent: Grok (xAI)

## 2026-08-21 — Friendship-size default hybrid pick

- `[pick] hybrid = true` on a 932-row scratch (930 bg + A/B). `pick Q` with no `--hybrid` → **B**. `pick Q2` → **A**. `--no-hybrid Q` still **B**.
- Hypothesis: the two-question table holds when hybrid is the silo default, not a flag.
- Later: first Q2 run failed — A was in the BM25 pool with cosine 0, hybrid buried it. Pick-path fill from splat 64-d fixed it. SciFact recall unchanged.
- Research: `research/logs/2026-08-21_friendship-size-default-hybrid-pick.md`
- Agent: Grok (xAI)

## 2026-08-21 — Easyway guillotine cap and beta retune

- Collapse write is `(τ × 0.05).min(0.5)`. Default `--hybrid` β=2. After persist: A 0.5, B 1.0. Next `--hybrid` **B**. `--hybrid-beta 0` also **B**. Q2 still **A**.
- Hypothesis: leftover habit can no longer beat a fresh pick; meaning can beat a 0.5 usage gap.
- Later: held on the whole scratch pipeline. `cargo test --lib hybrid` 15 passed.
- Research: `research/logs/2026-08-21_easyway-guillotine-cap-and-beta-retune.md`
- Agent: Grok (xAI)

## 2026-08-21 — Whole Easyway pipeline ranking vs state

- Full scratch pipeline (cage, ranking shatter, persist shatter, Q2). Ranking shatter helps this query. Persist 95% leftover still cages on β=0 (τ 1.2 vs 1.0). β=1 after persist picks B. Geometry unhurt.
- Hypothesis: persist Easyway would make next `--hybrid` pick B.
- Later: it did not. 0.05 collapse is dosage reduction, not Hamiltonian zero. Q2 flags-off still A.
- Research: `research/logs/2026-08-21_whole-easyway-pipeline-ranking-vs-state.md`
- Agent: Grok (xAI)

## 2026-08-21 — Easyway shatters the live cage

- `rerank_hits` now collapses high I_R before scoring. `--easyway` uses β=1 (flux after τ dies). Same scratch store, no re-saturate.
- Hypothesis: `--hybrid` still A; `--hybrid --easyway` B this cycle; `--hybrid` again A because we did not persist.
- Later: held. Cage 576 vs 0. Easyway B 76.76 vs A 7.08. Usage on disk still 24. `--easyway` with β=0 would have left leftover τ and kept A.
- Research: `research/logs/2026-08-21_easyway-shatters-the-live-cage.md`
- Agent: Grok (xAI)

## 2026-08-21 — Five Basin Field frontend-upgrade goals

- Wrote five checkable Basin Field UX goals from the 2026-08-20 connection map plus a re-check (search box exists now; Gravity is unread; Dream still walks a path). Why: Jason asked what to upgrade on the picture; the 10:50Z map said no search and that line is stale.
- Hypothesis: next frontend work takes one of these five (View sliders, honest live/theater HUD, projector scale with N, real dream SSE, query-hit neural net) instead of more dead Physics knobs or leftover assets/.
- Research: `research/logs/2026-08-21_five-basin-field-frontend-upgrade-goals.md`
- Agent: Grok (xAI)

## 2026-08-21 — Deneubourg scratch-store cage closed

- Scratch store only. Saturate A ×24 `--track-usage`, ingest B. Flags-off pick **B** (cos 0.90 vs 0.32). `--hybrid` (β=0) pick **A** (score 576 = 24²). Easyway switch is in the scorer, not thrown.
- Hypothesis: usage on A is enough to cage hybrid ranking while geometry/BM25 still prefer B.
- Later: first live run failed — neighborhood 0.40 zeroed τ at A cos 0.325, both hybrid scores 0. Neighborhood 0.25: τ ranks, cage closed. Holdout C at 0.10 still localized.
- Research: `research/logs/2026-08-21_deneubourg-scratch-store-cage-closed.md`
- Agent: Grok (xAI)

## 2026-08-21 — Hybrid usage channel and pick Brick 1

- `Splat.usage` (τ) and `Splat.conductivity` (D) on hot state, serde-default, not radiance. `PickConfig.hybrid` / `track_usage` default off. SciFact retrieval untouched.
- Hypothesis: old stores load usage=0 D=1. bump_usage does not move radiance. Shipped pick still prefers the short bridge when the long one has high radiance+usage. Trap rerank picks A; escape picks B.
- Later: held. `cargo test --lib` 123 passed.
- Research: `research/logs/2026-08-21_hybrid-usage-channel-and-pick-brick-1.md`
- Agent: Grok (xAI)

## 2026-08-21 — Hybrid ant-slime Brick 0 scorer

- Isolated ant-slime confidence in `src/hybrid.rs`. τ is usage, not radiance. Not wired to retrieval or SciFact. F_max saturation, neighborhood-local τ / evap / D.
- Hypothesis: β=0 A beats B (trap). β=1 B beats A (escape). Localized Q traffic leaves C untouched and C wins Q2. Short D grows, long D shrinks.
- Later: full-store trap tied A and C (same τ). γ=0.15 thinned a 0.90 holdout. 24 Q1 wins leaked τ_B onto Q2. γ=0.05; τ ranks only inside the cosine neighborhood. `cargo test --lib hybrid` 7 passed.
- Research: `research/logs/2026-08-21_hybrid-ant-slime-brick-0-scorer.md`
- Agent: Grok (xAI)

## 2026-08-21 — CombMNZ + term coverage for SciFact floors

- Freeze is RRF k=10 BM25×1.3 at 0.8584 / 0.7295 on poisoned full-300. Floors 0.88 / 0.75 miss by ~0.02. Rank-miss golds sit in both arms at 11–20 while single-arm FPs occupy top-10. Hypothesis: CombMNZ (RRF × number of arms that hit) plus a query-term coverage rank (not qrels) promotes both-hit lexical golds into k=10.
- Later: CombMNZ/coverage did not lift. Marker injection and BM25-rerank of the RRF pool both **regressed**. At cand=500 only 2 golds are both-miss; 23 rank-miss sit at 11–50. Hypothesis: max *sentence* query-term coverage (claim-like local pattern, not bag-of-words over the abstract) added to RRF, no qrels.
- Later: sentence weight 0.20 is a tiny new high (0.8596/0.7305). 64d dense already in RRF; we slice to 64d and throw away the query's 4096-d embedding. Next: on the RRF top-20, embed candidate texts at full dim and add that rank as a third RRF arm (no qrels).
- Later: full-dim arm on RRF top-20 = **0.8776 / 0.7579** (nDCG floor). Top-30 = **0.8798 / 0.7586**. Need ~one more gold. `weight_full` lets that arm go heavier without retuning 64d RRF.
- Research: `research/logs/2026-08-21_scifact-floors.md`
- Agent: Grok (xAI)

## 2026-08-20 — RRF pool then cosine rerank

- 64d dream + attach ran. RRF k=10 on that field **tied** 0.8551 / 0.7209 (RRF ignores basins). Linear got worse. Hypothesis: keep RRF for the pool (recall), rerank the top `rerank_pool * k` by 64d cosine for nDCG. No qrels in the ranker.
- Later: cosine rerank **failed** (0.7174 / 0.5680). SciFact is lexical. RRF k=10 BM25×1.3 is the new full-300 record: **R 0.8584 / nDCG 0.7295**. Also best floor-distance on train 809. Floors still miss.
- Research: `research/logs/2026-08-20_basins-from-64d.md`
- Agent: Grok (xAI)

## 2026-08-20 — Basins from 64d geometry, not 3D percolation

- Poisoned SciFact field: 4035 papers in one unlabeled basin, stability 0. Discovery was union-find on 3D radius ∩ cosine≥0.55, then kept every merge. Persistence intervals were recorded and unused. 3D PCA of science is one cloud, so the picture percolated. Hypothesis: pack basins to the 64d k-NN graph (the actual splat) and refuse to merge two min-sized wells unless cosine ≥ `basin_core_cosine` (0.75). 3D stays the picture / force arena.
- Later: k-NN at 0.55 made two islands (SciFact 4653 / Urban 4509). Random SciFact–SciFact cosine p90 is 0.55. 0.75 still left a 3607 SciFact well. **0.80 + k=8**: 232 basins, largest 476 SciFact at stability 0.50, 0 mixed-domain wells. Defaults: `basin_knn=8`, `basin_core_cosine=0.80`.
- Next cut (should have been first): a splat is a Gaussian. Cores are tight 64d modes; leftover points are the falloff, assigned to the nearest well (`basin_attach_cosine=0.72`) without ever merging two cores. Then dream so 3D follows that.
- Research: `research/logs/2026-08-20_basins-from-64d.md`
- Agent: Grok (xAI)

## 2026-08-20 — SciFact RRF k=10 (geometry off)

- Ranking-only sweep on a copy of the poisoned claim store. First-50 0.7822 already beaten. Full-300 floors 0.88/0.75 still open. Hypothesis: rank-miss golds at 11–20 plus drowned RRF (basin-local) — pure RRF k=10, radiance=0, local=0 promotes that band.
- Later: held on direction, missed the floor. Best R **0.8551**, best nDCG **0.7256**. Wide pool did nothing. Geometry is a constant on this field (one basin, same radiance on every gold).
- Research: `research/logs/2026-08-20_scifact-rrf-k10.md`
- Agent: Grok (xAI)

## 2026-08-20 — Frontend backend connection map

- Frontend Grok dropped the v3 UI↔engine connection map into `research/receipts/2026-08-20/frontend-backend-connections.v3.json`. Basin Field does not call `:8767`. Search/status/inspect/dream/steer are open and unused by the product picture.
- Hypothesis: Backend does not need new product routes for those verbs. The missing door is a live `splatrag.basin.v1` dump with `member.id` = UUID — or a clear no, so the picture treats off-page hits as a first-class state.
- Research: `research/logs/2026-08-20_frontend-backend-connection-map.md`
- Agent: Grok (xAI, frontend)

## 2026-08-20 — Basin is one steer object

- Live zero-gain smoke steered two pins that share a wallet. Next brick is one basin address as one steer object: room 64-d (member mean), gain≠0, same HTTP :8767 door. Not member 0 + member 1. Not two steers.
- Hypothesis: One POST `/api/tools/steer` with `addresses:[{basin_id}]` and `gain=-0.2` returns one target `kind=basin`, `members_touched`=room size (friendship basin 44), `cosine_before_after`<1, one `semantics_64` that is the room not member 0.
- Later: held. One target, `kind=basin`, `share=1`, `members_touched=44`, `total_applied=0.2`, cosine **0.503**, one 64-d on the report. Live field moved for that room. Pick still emits pins — that is the next hole, not this brick.
- Research: `research/logs/2026-08-20_basin-is-one-steer-object.md`
- Agent: Grok (xAI)

## 2026-08-20 — Pick/steer receipts out of /tmp

- Moved the 2026-08-20 pick/steer JSON dumps from `/tmp/splatrag-*.json` into `research/receipts/2026-08-20/`. Why: Jason — nothing in tmp. Those files were the live door smoke and would have been lost.
- Hypothesis: later smokes write here, not `/tmp`.
- Research: `research/logs/2026-08-20_steer_to_address.md`
- Agent: Grok (xAI)

## 2026-08-20 — Live door smoke (pick + zero-gain multi-steer)

- HTTP `:8767` already has `pick` and `steer` with `addresses`. Stdio MCP PIDs from 12:10 still run the noon binary, so Grok’s tool list can lag the serve. Why: the remaining eat is calling two full-α steers because the session never saw `pick`.
- Smoke: `POST /api/tools/pick` friendship `--limit 2`; `POST /api/tools/steer` those two addresses with `gain=0` so the field does not move. Hypothesis: pick returns v2 addresses and non-zero confidence; steer batch reports two targets, `total_applied=0`.
- Later: hypothesis held. Pick v2 confidence 0.406, suggested 0.071/0.071. Steer two pins, shares 0.501/0.499, `total_applied=0`, cosine ~1. Receipts now `research/receipts/2026-08-20/` (moved out of `/tmp`). README still said `steer` is one splat — corrected to match the wire.
- Research: `research/logs/2026-08-20_steer_to_address.md`
- Agent: Grok (xAI)

## 2026-08-20 — Changelog exists because it was missing

- Jason stopped the agent: edits went in before a changelog row. This tree had no `CHANGELOG.md`. Why: house law is changelog first, then code. Structure was missing; organize before more mutations.
- Hypothesis: the next mutation will have a why in this file before the rust changes land.
- Research: `research/logs/2026-08-20_research-house-changelog.md`
- Agent: Grok (xAI)

## 2026-08-20 — Steer to address wired

- CLI `steer --memory UUID --memory UUID --budget 0.35 --gain -0.35`. MCP/HTTP `addresses` + `budget`. Basin-only `--basin ID` moves the room (member mean, members ride `μ'−μ`). Why: one call, one split. Legacy one-id `steer <uuid>` kept.
- Pick confidence is `max(cosine-z, lexical BM25)`. ANN-miss floor 0.5. Why: friendship-style hits were α 0.
- Hypothesis: two-address steer total |α| stays ≤ budget. Room 64-d ≠ member 0 (unit test). Friendship pick suggested α > 0.
- Research: `research/logs/2026-08-20_steer_to_address.md`
- Agent: Grok (xAI)

## 2026-08-20 — Steer to address (one call, one budget)

- **Logged before finishing the wire.** Pick already emits addresses. Steer is still one pin. Two MCP/HTTP steers still pay 0.35 twice. Why: that is the live eat. Next brick is steer-to-address, not a second picker.
- Logic: N addresses, one signed budget, split once, one write lock. `memory_id` keys the pin. Basin-only → room 64-d = mean of members, not member 0. Share clamp must not lift tiny α to 0.05 or the split restacks. Confidence cannot be cosine-only or friendship-style BM25 hits stay at α 0.
- Started in tree (out of order, before this row): `clamp_alpha_share` / `apply_steering_share` in `src/inversion.rs`; `relevance_from_hits` + lexical floor in `src/pick.rs`. Not done: batch steer on CLI/MCP/HTTP, basin room write, serve restart.
- Hypothesis: one `steer` with two pick addresses spends ~0.51/0.49 of one budget, not 0.35+0.35. Friendship pick `--limit 2` gets non-zero suggested α.
- Research: `research/logs/2026-08-20_steer_to_address.md`
- Agent: Grok (xAI)

## 2026-08-20 — Pick emits addresses, not essays

- Picker wire v2: `injection: address`, `address.{memory_id,basin_id,source,domain}`, `inspect` ≤ 240, `text` off the JSON. MCP tool `pick` in source. Why: 2000-char dumps were the payload; hydro residual cannot take that pile, and 64-d is look-only.
- Live `--limit 2` OI split ~0.51/0.49 (nobody ate). Friendship: cosine 0 → α 0 (confidence was cosine-z). Serve at `:8767` was still the noon binary when this landed.
- Hypothesis: consumers look up by id; empty `text` makes hydro import refuse, which is correct.
- Research: `research/logs/2026-08-20_pick_addresses.md`
- Agent: Grok (xAI)
- Agent: Grok (xAI)
