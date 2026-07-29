# SplatRAG

SplatRAG is a local Rust AI-memory store. It keeps every imported message in an append-only cold
log, indexes it with BM25 plus keyed HNSW and scoped Qdrant cosine search, and evolves a separate
Gaussian-splat field into persistent, locally labeled memory basins.

No cloud model is used. Embeddings and basin labels come from local `llama-server` instances.
The local ANN is `fast-hnsw` with UUID strings as labels: insertion positions remain internal
implementation details, while search ties and exported keys are sorted deterministically by UUID.

## Services

- Qdrant: `http://127.0.0.1:6360`, collection `export-conversations`, filtered by `scope_key`.
- Embedder: `http://127.0.0.1:8081`, Qwen3-Embedding-8B, 4096 dimensions.
- Optional basin labeler: `http://127.0.0.1:8082`, local instruction model.

Copy/edit the generated `splatrag.toml` if your ports or model aliases differ.

```bash
cargo run -- init
cargo run -- doctor
cargo run -- ingest --source claude /path/to/conversations.json
cargo run -- ingest --source grok /path/to/export_conversations.json
cargo run -- ingest --source gemini /path/to/MyActivity.html
cargo run -- ingest --source agent-jsonl /path/to/sessions
cargo run -- ingest --source semantic-md /path/to/splat_backup_semantic_FULL_53037.md
cargo run -- dream --label
cargo run -- recall "what did we decide about memory clustering?"
cargo run -- serve
```

The viewer is served at `http://127.0.0.1:8765`. Start the MCP stdio server with:

```bash
cargo run --release -- mcp
```

Available MCP tools are `remember`, `recall`, `list_basins`, `browse_basin`, and `memory_status`.
The viewer supports basin/memory level-of-detail, orbit and zoom, local hybrid search, basin cards,
and click inspection of individual splats.

## Local model processes

The defaults in `splatrag.toml` expect these two endpoints:

```bash
llama-server --model /path/to/Qwen3-Embedding-8B-Q8_0.gguf \
  --embedding --pooling last --host 127.0.0.1 --port 8081 \
  --n-gpu-layers all --flash-attn on

llama-server --model /path/to/gemma-3-4b-it-q4_0.gguf \
  --host 127.0.0.1 --port 8082 --ctx-size 4096 \
  --n-gpu-layers all --flash-attn on
```

`cargo run -- doctor` verifies model dimensions, Qdrant compatibility, and cold/derived count
parity before a large import.

## Executable HANDSHAKE

The preserved acceptance test is implemented by the `handshake` command. It imports the SciFact
corpus, measures Recall@10 and nDCG@10, optionally imports an Urban Dictionary poison corpus into
the same cold store, dreams, measures again, and exits nonzero on failure. The fixed targets are
Recall@10 ≥ 0.88, nDCG@10 ≥ 0.75, no more than 0.05 Recall loss, and byte-for-byte preservation of
the cold log during dream.

Use a separate config/scope so benchmark documents do not enter the personal memory scope:

```bash
cp splatrag.toml splatrag.handshake.toml
# Set a separate data_dir and scope_key in splatrag.handshake.toml.

cargo run --release -- --config splatrag.handshake.toml handshake \
  --dataset /path/to/scifact \
  --poison /path/to/urban-dictionary.jsonl
```

Without `--poison`, the command runs and reports the SciFact baseline only. `--limit` is available
for a quick development sample; omit it for an acceptance run. SciFact `_id` values are retained
as source record keys so the evaluator scores the same identifiers present in the qrels.

## Storage invariants

- `data/cold/memories.jsonl` is authoritative and append-only.
- Text equality does not deduplicate messages. Stable source identities make re-import idempotent.
- `data/indexes/`, `data/hot/`, Tantivy, HNSW, Qdrant points, and basin labels are derived.
- Dreaming changes only hot geometry and basin metadata; it never deletes cold messages.
- Attachment paths and hashes are retained. OCR is intentionally deferred.

`splatrag rebuild-index` moves current derived files to a timestamped backup and reconstructs all
indices from the cold log. It does not alter the cold log. Because basins and local labels are
derived state, run `splatrag dream --label` after a rebuild.
