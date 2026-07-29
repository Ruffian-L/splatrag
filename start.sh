#!/usr/bin/env bash
# Bring SplatRAG back up after a reboot. Everything binds to localhost only.
#
#   ./start.sh          models + both viewers
#   ./start.sh models   models only
set -u
cd "$(dirname "$0")"

LLAMA="${LLAMA:-$HOME/.local/bin/llama-server}"
EMBED_MODEL="${EMBED_MODEL:-/media/ruffianl/ghost_team/models/Qwen3-Embedding-8B-Q8_0.gguf}"
LABEL_MODEL="${LABEL_MODEL:-$HOME/Downloads/gemma-3-4b-it-q4_0.gguf}"
LOGS="${LOGS:-/tmp/splatrag-logs}"
mkdir -p "$LOGS"

up() { curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$1/health" 2>/dev/null; }
listening() { ss -ltn 2>/dev/null | grep -q "127.0.0.1:$1 "; }

start_model() {          # port, model, extra args
  if up "$1"; then echo "  :$1 already up"; return; fi
  echo "  starting :$1 $(basename "$2")"
  # shellcheck disable=SC2086
  nohup "$LLAMA" --model "$2" --host 127.0.0.1 --port "$1" \
    --n-gpu-layers 999 --flash-attn on $3 > "$LOGS/$1.log" 2>&1 &
}

echo "models:"
start_model 8081 "$EMBED_MODEL" "--embedding --pooling last"
start_model 8082 "$LABEL_MODEL" "--ctx-size 4096"

printf "  waiting for embedder"
for _ in $(seq 60); do up 8081 && break; printf .; sleep 2; done; echo
up 8081 && echo "  embedder :8081 ok" || echo "  embedder :8081 FAILED - see $LOGS/8081.log"
up 8082 && echo "  labeler  :8082 ok" || echo "  labeler  :8082 FAILED - see $LOGS/8082.log"

[ "${1:-all}" = "models" ] && exit 0

# 8765 is the running niodoo service. Do not take it.
echo "viewers:"
start_viewer() {         # config, port, label
  if listening "$2"; then echo "  :$2 already in use ($3)"; return; fi
  nohup ./target/release/splatrag --config "$1" serve > "$LOGS/viewer-$2.log" 2>&1 &
  sleep 2
  listening "$2" && echo "  $3  http://127.0.0.1:$2" \
                 || echo "  $3 FAILED - see $LOGS/viewer-$2.log"
}
start_viewer splatrag.toml         8767 "memories"
start_viewer splatrag.cluster.toml 8766 "cluster demo"

echo
echo "no WebGL2 in the browser? the static views need no GPU:"
echo "  file://$PWD/viz/clusters.svg"
echo "  file://$PWD/viz/memories.svg"
