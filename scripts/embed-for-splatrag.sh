#!/usr/bin/env bash
# One CUDA embedding process shared by every SplatRAG silo and MCP client.
set -euo pipefail

SR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SR_ROOT"

# shellcheck source=lib.sh
. "$SR_ROOT/scripts/lib.sh"
sr_load_env

MODEL_PATH="${SPLATRAG_EMBED_MODEL:-${EMBED_MODEL:-}}"
LLAMA_BIN="${LLAMA:-}"
PORT="${SPLATRAG_EMBED_PORT:-8081}"
if [[ -z "$LLAMA_BIN" ]]; then
  LLAMA_BIN="$(command -v llama-server || true)"
fi
if [[ -z "$LLAMA_BIN" && -x "${HOME}/.local/bin/llama-server" ]]; then
  LLAMA_BIN="${HOME}/.local/bin/llama-server"
fi
if [[ -z "$MODEL_PATH" ]]; then
  echo "splatrag-embed: set SPLATRAG_EMBED_MODEL or EMBED_MODEL to your Qwen3-Embedding GGUF" >&2
  exit 1
fi

[[ -x "$LLAMA_BIN" ]] || { echo "splatrag-embed: llama-server missing: $LLAMA_BIN" >&2; exit 1; }
[[ -f "$MODEL_PATH" ]] || { echo "splatrag-embed: model missing: $MODEL_PATH" >&2; exit 1; }

echo "splatrag-embed: one shared CUDA server model=$MODEL_PATH port=$PORT" >&2
exec "$LLAMA_BIN" \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --embedding \
  --pooling last \
  --ubatch-size 2048 \
  --batch-size 2048 \
  --n-gpu-layers 999 \
  --flash-attn on
