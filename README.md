# SplatRAG

**3.0.0.** A local Rust AI-memory store.

Every imported message is appended to a cold log and never deleted. Retrieval is hybrid BM25 plus keyed HNSW over 64-d Matryoshka embeddings, with an optional full-dimension rerank. A separate Gaussian-splat field is evolved into labeled basins. A picker chooses which memories steer a live model, and how hard.

Embeddings and optional basin labels come from local `llama-server`. Nothing is sent to a cloud model.

The local ANN is `fast-hnsw` with UUID labels: insertion positions stay internal; search ties and exported keys are sorted by UUID.

## Quickstart

```bash
cp .env.example .env          # optional; defaults work
./run init                    # write splatrag.toml from the template
./run up                      # embedder, then the service
./run recall "what did we decide about memory clustering?"
```

The HTTP API and the 3-D viewer are at `http://127.0.0.1:8767`. `./run --help` lists the rest; unknown subcommands are passed to the engine.

## Reproduce the claim

```bash
./claim.sh --smoke            # ~15 min plumbing check on a reduced corpus
./claim.sh                    # SciFact → poison → dream → SciFact; writes a receipt
```

`./claim.sh` checks the toolchain, verifies the embedding-model hash, builds, downloads and hash-verifies SciFact, starts the embedder, runs the handshake against a fresh store, and writes `claim-receipt.json`.

**Baseline going forward is the full 300 SciFact test queries**, not the first-50 slice that produced the older 0.7822 nDCG@10 figure. On the current engine, filtered SciFact+Urban (domain=`scifact` so poison cannot occupy top-10):

| Slice | System | R@10 | nDCG@10 |
| --- | --- | ---: | ---: |
| first 50 | SplatRagBench hybrid (Nomic, 2026-08-04) | 0.9090 | 0.7822 |
| full 300 | SplatRagBench hybrid (same binary) | 0.7876 | 0.6664 |
| full 300 | this engine, RRF + 4096-d arm (`eval_rrf_full3_w13`) | 0.8804 | 0.7624 |

Handshake floors are Recall@10 ≥ 0.88 and nDCG@10 ≥ 0.75 on that filtered full-300 run. Poison resistance is a separate check: recall drop ≤ 0.05 and the cold log is byte-identical across dream.

Use a separate `data_dir` / `scope_key` so benchmark documents never enter a personal store:

```bash
cp examples/splatrag.example.toml splatrag.handshake.toml
# set data_dir and scope_key

cargo run --release -- --config splatrag.handshake.toml handshake \
  --dataset /path/to/scifact \
  --poison corpora/poison/urban-dictionary.jsonl
```

The poison corpus ships in-tree (`corpora/poison/`). Without `--poison`, the command reports the SciFact baseline only. `--limit` is a development sample.

## Retrieval, in one paragraph

Recall fuses BM25 rank with 64-d HNSW cosine rank (reciprocal rank fusion). The 64-d vectors are the first 64 dimensions of Qwen3-Embedding-8B, L2-renormalized. When `rerank = "full"`, the fusion top-`k * rerank_pool` documents are re-embedded at 4096-d and that rank is added as a third RRF arm. Radiance (dream connectivity) is a small add-on, not a third list. The 3-D picture is PCA of the 64-d vectors — a view, not the index.

## Storage invariants

- `data/cold/memories.jsonl` is authoritative and append-only.
- Text equality does not deduplicate. Stable source identities make re-import idempotent.
- Indexes, hot geometry, and basin labels are derived.
- Dreaming changes only hot geometry and basin metadata. It never deletes cold messages.
- Attachment paths and hashes are kept. OCR is deferred.

`splatrag rebuild-index` reconstructs derived files from the cold log. Run `splatrag dream --label` afterwards if you want basins relabeled.

## MCP and HTTP

```bash
cargo run --release -- mcp
```

Tools: `remember`, `recall`, `pick`, `steer`, `pack_64`, `unpack_64`, `list_basins`, `browse_basin`, `memory_status`, `ensure_ready`, `list_silos`. `pick` emits addresses (`memory_id` + basin), not a text dump. The same tools are on HTTP: `GET /api/tools`, `POST /api/tools/<name>`. `POST /api/recall` takes the full filter set.

## Local models

```bash
llama-server --model /path/to/Qwen3-Embedding-8B-Q8_0.gguf \
  --embedding --pooling last --host 127.0.0.1 --port 8081 \
  --n-gpu-layers all --flash-attn on
```

Optional basin labeler on `:8082`. `cargo run -- doctor` checks dimensions and cold/derived count parity.

Copy `examples/splatrag.example.toml` to `splatrag.toml` and point `data_dir` **outside** the git checkout. See `DATA_LAYOUT.md`.

## Where this came from

*— Jason Van Pham*

Memories as Gaussians started in research sessions with Gemini, before this was framed as hybrid retrieval. [SplatRagBench](https://github.com/Ruffian-L/SplatRagBench) shipped first, while the engine was still rough. This repository is the engine.

Full attribution: [`CREDITS.md`](CREDITS.md) · [`NOTICE`](NOTICE) · [`attributions.md`](attributions.md) for llama.cpp.

Model weights are not distributed here and are not covered by the MIT license.
