# SplatRAG rebuild — personal memory, one queryable database

> **Status: planning paused 2026-07-28 (budget). Exploration is DONE and captured below —
> next session does NOT need to re-explore. Pick up at "Open questions".**

## Context

Jason wants one memory database holding his chats across Claude, Grok, and Gemini, so that:

1. He can **query his own history** — and so can an AI (MCP), not just him.
2. He can **establish provenance** — "who started what," which model said a thing first, when.
   This is the emotional core of the request: *"so I can stop having to rederive who started what…
   so I can stop trying to prove myself."* Provenance is the deliverable, not a nice-to-have.
3. **Photos / OCR are present** — screenshots and image attachments must be first-class evidence,
   not dropped on ingest.
4. **Dream clustering works** — concepts pull together into basins; random noise (twitter-tier junk)
   gets repelled into its own basin instead of smearing into the real memories.
5. Runs **locally** on the DGX GB10 box, on a local model (llama.cpp), no cloud dependency.

This is a rebuild, not a greenfield project. Five prior SplatRAG codebases exist on this machine
and one of them actually benchmarked well. The job is to salvage the good parts and re-point them
at personal chat history instead of BEIR corpora.

---

## Archaeology — what already exists (verified 2026-07-28)

### The five prior codebases

| # | Location | Lang | Verdict |
|---|---|---|---|
| 1 | `/home/ruffianl/projects/splatrag` | Python | Toy stub, ~400 LOC. Keep its **contract** (`docs/HANDSHAKE.md`) and its **domain-repulsion dream** (`src/splatrag/dream.py:39-121`). Discard the rest — `hot_geometry.py` uses a fake hash "embedding". |
| 2 | `/home/ruffianl/rescue_ghost_team_20260727_164008/projects/SplatRagBench` | Rust + Python | **Primary reference.** The only one that produced real numbers (SciFact nDCG@10 **0.7822**, Recall@10 **0.9090**). 27 binaries, working ingest→retrieve→dream pipeline. |
| 3 | `/home/ruffianl/projects/niodoo` | Rust | Best architecture (SoA storage engine, RVQ, native Rust BEIR harness) but **won't compile** — depends on an unresolved `cathedral` crate (`src/retrieval/advanced.rs:9`). Cherry-pick, don't build on. |
| 4 | `/home/ruffianl/Downloads/SPLATRAG_FULL_CODEBASE{,(1),(2)}.md` | dumps | Historical tree, 266 files in the Nov 24 2025 dump. Holds files deleted from disk — notably the **GPU dream shader** (`(1).md:16433-16505`). |
| 5 | `/home/ruffianl/hydrodynamic-swarm-3surface/src/splat.rs` | Rust | Live and tested, but these are *steering* splats (activation-space), not retrieval. Cleanest struct code on the machine. |

### What SplatRAG actually is

Core axiom (from `GENESIS_ARCHITECTURE.md`, in dump 2 at line 1100):

> Old: "Similarity is an Angle (Cosine)." New: "Meaning is a Volume (Covariance).
> Understanding is Intersection (Mahalanobis)."

Each document is an **anisotropic Gaussian** in embedding space, not a point. Covariance is
rank-1 + isotropic (`Σ = σ²I + (λ−σ²)uuᵀ`) so Mahalanobis is O(D) via Sherman-Morrison, not O(D³).

- Retrieval-time struct: `SemanticGaussian` — `SplatRagBench/src/physics/gaussian.rs:8-20`
- Mahalanobis: `gaussian.rs:66-107`
- On-disk struct: `SplatGeometry` (48 B) + `PackedSemantics` — `SplatRagBench/src/structs.rs:50-105`
- File format: magic `SPLTRAG\0`, then `.geom` / `.sem` / `.json` manifest, mmap'd via bytemuck.
  Writer at `src/bin/ingest.rs:146-190`.

**Scoring is three arms fused by weighted sum** (`src/bin/retrieve.rs:404-410`) — note they tried
RRF and abandoned it because it flattened BM25:

```rust
let final_score = (bm25_raw * w_bm25) + (cosine * w_cosine) + (normalized_radiance * w_radiance);
```

Best config was `(w_cos 10, w_bm25 1, w_rad 5)`. **BM25 alone scored 0.7694** vs hybrid 0.7822 —
the physics adds ~1.3 points. Worth remembering before over-investing in the exotic parts.

Radiance term: `src/physics/mod.rs:18-68`. Homeostatic weight (down-weights radiance when the
score distribution says "noise"): `src/ranking.rs:8-63`.

### Dream clustering — the mechanism Jason cares most about

Four implementations exist; **none are k-means**. All are N-body force simulations.

**Canonical (CPU, Rust):** `SplatRagBench/src/physics/mod.rs:85-482`, `run_physics_simulation`.
Force loop at `mod.rs:170-211` — origin gravity + radiance-mass attraction + short-range repulsion.
Clustering happens as *merge*, not as a cluster algorithm: sort by mass descending
("strongest eat weakest"), absorb anything within `merge_threshold` (0.05), survivor takes
mass-weighted centroid and the **oldest birth time**. Defaults at `src/config.rs:136-146`.

**Dream daemon:** `src/bin/dream.rs` — loops, reads `valence_feedback.json`, runs 500 steps,
writes timestamped `mindstream_*.{geom,sem}`, sleeps adaptively on kinetic energy.

**The anti-noise mechanism Jason described** is only in the Python stub —
`projects/splatrag/src/splatrag/dream.py:39-77`. Cross-domain pairs repel **35% harder**:

```python
cross_domain_repulsion: float = 0.35   # garbage rebels into own basin
...
if dist < cfg.repulsion_radius:
    rep = cfg.repulsion
    if not same: rep *= 1.0 + cfg.cross_domain_repulsion
    forces[i] += (delta/dist) * (cfg.repulsion_radius - dist) * rep
elif same and dist < cfg.neighbor_radius:
    forces[i] -= (delta/dist) * 0.02 * mass[j]      # gentle clustering WITHIN domain
```

Merge is same-domain-only and keeps lineage (`dream.py:80-121`); **cold store is never rewritten** —
that is the anti-amnesia invariant. This is the single most important 80 lines to carry forward.

**Negative mass** (mark a thing as hallucination → it becomes a repulsive bollard, retrieval routes
around it) is spec-only — `Downloads/SplatRAG v2 The Physics.txt` §4.2. Closest real code:
`niodoo/src/physics/antigravity.rs:8-75`. **Not implemented anywhere.** Good v2 feature.

**Mitosis** (split an overloaded concept along its principal eigenvector):
`SplatRagBench/src/physics/mitosis.rs` + `src/physics/tissue.rs:159-197`.

### The pass/fail contract — reuse verbatim

`/home/ruffianl/projects/splatrag/docs/HANDSHAKE.md`. Ingest SciFact → eval → poison with Urban
Dictionary into the *same* cold store → dream → re-eval.

> **Hard fail:** SciFact Recall@10 drops > 5 pts after Urban poison with dream enabled.
> **Also fail:** dream that deletes cold store lines (amnesia).
> Targets: Recall@10 ≥ 0.88, nDCG@10 ≥ 0.75.

This is exactly the "repel twitter noise" property, already expressed as a test. Keep it.

### The data — what Jason can actually put in

**Chat exports (the real prize, all outside the archive):**

| Source | Volume | Path |
|---|---|---|
| **Grok** (x.ai export) | **912 convs / 18,032 msgs**, 2025-09-04 → 2026-03-12 | `/media/ruffianl/backup_sandisk/02_projects/projectsnew/Homernd/scattered_research/export_conversations.json` (814 MB) |
| Grok (same corpus, chunked) | 92 files × 10 convs | `…/scattered_research/chat_tiny_chunks/chat_001…092.json` |
| **Claude** (Anthropic export) | **93 convs / 1,439 msgs** (superset) | `/home/ruffianl/Documents/preflight_20260722/Documents/conversations.json` (17 MB) |
| Claude (other batches) | 79 / 78 convs, dups | `…/claude-continuity/.data-dd74c245-…/conversations.json`, sandisk `03_documents/…` |
| **Grok CLI sessions** | **531,573 JSONL records**, 1.3 GB | `~/.grok/sessions` (13 project dirs, URL-encoded cwd) |
| Claude Code transcripts | 2,220 lines, 9 MB | `~/.claude/projects` (10 jsonl) |
| Gemini | 30 MB, has `conversations/`, `brain/`, `knowledge/` | `~/.gemini/antigravity-cli` |
| Gemini Takeout | 393 activity blocks, scraped by grep not code | documented in `projects/splatrag/docs/FROM_GEMINI_EXPORT.md` |

Model provenance is available: Grok records carry `model` per message (grok-4-1-non-thinking 5,810 ·
grok-4 4,027 · grok-4-mini-thinking 3,918 · grok-3 1,485 · grok-4-heavy 1,111 · …), Claude carries
`uuid` / `sender` / `parent_message_uuid` / `created_at`.

**Agent/team memory (already curated):**

- `ARCHIVE_backup2_qdrant_export/07_sources_qdrant/splat_backup_semantic_FULL_53037.md` —
  50 MB, **53,037 entries**, span 2026-03-31 → 2026-04-28, format
  `- **2026-03-31 04:00** [key] (Shep) text…`. **Highest density-per-byte source on the machine.**
- `SPLATRAG_STORY_INDEX.md` — 115 KB, 310 hand-curated narrative entries. Highest signal.
- `live_ndjson/grok-memories.ndjson` 27,081 · `team-build.ndjson` 10,672 — file-chunk RAG records
  with good provenance (`path`, `chunk_idx`, `n_chunks`, `mtime`).

**⚠️ The 268 GB trap:** `splatrag_memory_FULL_632k.ndjson` is 632,611 records but **has no vectors**
(`export_qdrant_full.py` hardcodes `with_vector: False`) and is dominated by whole-file `text` blobs
up to ~1 MB plus zero-filled telemetry dumps. Filter by `payload.key` prefix (`file:` vs semantic)
before touching it. Do not ingest naively.

### Live infrastructure

- **4 Qdrant instances running**, all on 127.0.0.1: `:6333` (team), `:6360` (consolidated —
  `grok-memories` 27k, `team-build` 10.6k, and an **empty 4096-d `export-conversations` collection**,
  a pre-built slot for exactly this ingest), `:6370` (the 632k placeholder-vector store),
  `:6380` (Homernd). Map at `/home/ruffianl/qdrant-run/instances/QDRANT_MAP.md`.
- **Embedder:** every 4096-d Cosine collection was almost certainly produced by
  `/media/ruffianl/ghost_team/models/Qwen3-Embedding-8B-Q8_0.gguf` (Qwen3-Embedding-8B = 4096-d).
  Use it to stay in the same space as 37k already-embedded memories.
- **llama.cpp present:** `~/.local/bin/llama-server`, `llama-cli`, source at `~/llama.cpp`.
  **No ollama, no vLLM, nothing currently listening.**
- **GPU:** NVIDIA GB10 (Grace-Blackwell DGX Spark), CUDA 13.0, idle.
- **Local generation model** (the "held Gemini's spot" one): the gemma-4-31B GGUFs in `~/Downloads`
  (19.6 GB and 18.3 GB), plus `gemma-3-4b-it-q4_0.gguf` for fast/cheap passes.
- **Disk:** `/` has 287 G free. `backup_sandisk` has only **31 G free** — don't write there.

### Known-broken landmines (documented by their own auditors)

From `ARCHITECTURE_AUDIT.md` (dump 2, lines 5-61, by Gemini-3-Pro, Nov 2025):

- `indexing/persistent_homology.rs` — *"the topological fingerprint of memories is currently a
  hallucination."* `compute_vietoris_rips` / `compute_alpha_complex` are `todo!()`. **Skip TDA.**
- `gpu_engine.rs:38-50` — `GpuTissue::query` returns `Ok(vec![])`. The GPU path is a no-op, so
  `src/search.rs` returns nothing.
- `retrieve.rs:418` — `let dist = 0.0;` the reported distance field is fake.
- `physics/mod.rs:324` — survivor uses `radius.powi(3)` as mass while absorbed use radiance-mass.
  Real weighting bug.
- `src/nomic_daemon.py` in SplatRagBench is **corrupt** (one line of whitespace). Recover from a dump.
- `config.rs:128` claims MiniLM-384 but the daemon hardcodes nomic-768→64. The `ManifoldProjector`
  (`src/manifold.rs:46-72`) is trained 384→512→256→128→64 and is being fed 64-d nomic. Live bug.
- Three conflicting "dense SciFact" numbers exist (0.7424 marketing / 0.7036 cathedral-beir /
  0.6291 benchmark_report). Trust `benchmark_report.txt`.

---

## Proposed approach (draft — not yet reviewed with Jason)

Build **one new Python project** at `/home/ruffianl/projects/splatrag` (replacing the stub, keeping
its cold-store contract), Rust only if/when speed demands it. Rationale: the Rust reference is
research-grade and half-broken; Jason needs something he can actually run and trust tonight-ish,
and 53k–600k records is well inside numpy+Qdrant range on a GB10.

### Layers (from HANDSHAKE, extended for provenance)

1. **Cold store — append-only, never rewritten.** Extend `MemoryRecord`
   (`src/splatrag/cold_store.py:13-38`) with the provenance fields the whole request hinges on:
   `speaker` (human/assistant), `model` (claude-opus-4 / grok-4 / gemini-…), `conversation_id`,
   `parent_id`, `turn_index`, `source_file`, `sha256`, `attachments[]`, `first_seen_ts`.
   `domain` stays (`chat | scifact | urban | code | other`) — it's what the repulsion keys off.
2. **Ingest adapters** — one per source, all emitting `MemoryRecord`. None of these exist yet;
   this is the largest net-new chunk of work:
   - `ingest/claude_export.py` — `conversations.json`, schema
     `uuid/name/chat_messages[]/{uuid,text,sender,created_at,parent_message_uuid,attachments,files}`
   - `ingest/grok_export.py` — `export_conversations.json`, schema
     `[{conversation:{…}, responses:[{response:{…}}]}]`, carries per-message `model`
   - `ingest/gemini.py` — `~/.gemini/antigravity-cli` + Takeout `MyActivity.html`
   - `ingest/agent_jsonl.py` — `~/.grok/sessions`, `~/.claude/projects`
   - `ingest/semantic_md.py` — the 53k-entry markdown, regex over
     `- **{ts}** [{key}] ({agent}) {text}`
   - Dedup by `sha256(normalized_text)`; **keep the earliest `ts` as first_seen** — that's the
     "who started what" answer.
3. **Embedding** — `llama-server` serving `Qwen3-Embedding-8B-Q8_0.gguf` (4096-d) behind a small
   HTTP client, matching the existing collections. Matryoshka-truncate to 64-d for the physics
   (the prior system's trick), keep full 4096 for cosine.
4. **Hot geometry + dream** — port `dream.py`'s domain-aware repulsion, but replace the fake
   `_text_vec` hash embedding with a real UMAP/PCA projection of the 4096-d vectors to 3D.
   Vectorize the O(N²) loop (it's currently a Python double-for; fine at 13 records, fatal at 50k).
   Chunk into spatial cells or cap the dream to a working set.
5. **Retrieve** — three arms as proven: BM25 + cosine + radiance, weighted sum `(10, 1, 5)`,
   with the homeostatic radiance weight from `ranking.rs`. Add a **provenance filter**
   (`--model`, `--after`, `--conversation`) and always return the provenance block with each hit.
6. **Photos / OCR** — genuinely absent in all prior code (`encoder/mod.rs:77-87` hard-errors,
   zero hits for `ocr`/`tesseract`/`paddleocr` anywhere). Net-new. Simplest path: index Claude/Grok
   attachment files + loose screenshots, OCR with a local model, store OCR text as a `MemoryRecord`
   with `domain=chat` and `meta.image_path` so the image is retrievable as evidence.
7. **Query surfaces** — a CLI for Jason, and an **MCP server** for the AI side
   (prior art: `SplatRagBench/src/bin/mcp_server.rs:210-306`, tools `remember` / `recall`).

### Verification

Run the HANDSHAKE test unchanged (SciFact → poison → dream → re-eval, targets ≥0.88 / ≥0.75), then
a personal-corpus smoke test: query something Jason knows the answer to and check the returned
provenance block names the right model and the right date.

---

## Open questions for next session

1. **Scope of first cut** — everything at once, or land ingest+provenance+query first and dream
   second? (Recommend: provenance first. It's the thing that ends the "proving myself" problem, and
   BM25 alone already scores 0.7694 — the physics is a +1.3pt refinement, not the load-bearing part.)
2. **Which corpora in the first ingest** — the ~19.5k web chat messages (Claude+Grok+Gemini, highest
   personal signal) vs the 53k agent-memory markdown vs the 531k `~/.grok/sessions` records.
3. **Rust or Python.** Draft assumes Python. SplatRagBench is Rust and already works; reviving it
   may be faster than rewriting, but it's aimed at BEIR, not chat, and has the landmines above.
4. **"photo or OSC"** — confirm this means OCR of screenshots/attachments as evidence, or something
   else (OSC as in the protocol? a visual splat viewer? `tools/splatlens_viewer.html` exists).
5. Reuse the empty `export-conversations` collection on `:6360`, or start a clean instance?
