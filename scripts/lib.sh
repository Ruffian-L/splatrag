#!/usr/bin/env bash
# Shared plumbing for ./run and ./claim.sh. Source it, do not execute it.
#
# Everything configurable is an env var with a working default, overridable in .env.
# No absolute paths to anyone's home directory live in this file.

# Resolve the repo root from this file's location, following symlinks.
_lib_self="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  _lib_self="$(readlink -f "$_lib_self" 2>/dev/null || echo "$_lib_self")"
fi
SR_ROOT="$(cd "$(dirname "$_lib_self")/.." && pwd)"
unset _lib_self

# --- output ------------------------------------------------------------------

if [[ -t 1 ]]; then
  c_ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
  c_bad()  { printf '\033[31m%s\033[0m\n' "$*"; }
  c_warn() { printf '\033[33m%s\033[0m\n' "$*"; }
  c_dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
  c_head() { printf '\033[1m%s\033[0m\n' "$*"; }
else
  c_ok()   { printf '%s\n' "$*"; }
  c_bad()  { printf '%s\n' "$*"; }
  c_warn() { printf '%s\n' "$*"; }
  c_dim()  { printf '%s\n' "$*"; }
  c_head() { printf '%s\n' "$*"; }
fi

step() { c_head ""; c_head "── $* ${_sr_rule:-}"; }
ok()   { c_ok   "  ✓ $*"; }
bad()  { c_bad  "  ✗ $*"; }
warn() { c_warn "  ! $*"; }
note() { c_dim  "    $*"; }

die() { bad "$*"; exit 1; }

# --- env ---------------------------------------------------------------------

sr_load_env() {
  if [[ -f "$SR_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$SR_ROOT/.env"
    set +a
  fi

  SPLATRAG_RUNTIME="${SPLATRAG_RUNTIME:-$SR_ROOT/splatrag-runtime}"
  # Expand a leading ./ so the value is usable from any cwd.
  case "$SPLATRAG_RUNTIME" in
    ./*) SPLATRAG_RUNTIME="$SR_ROOT/${SPLATRAG_RUNTIME#./}" ;;
  esac

  MODELS="${MODELS:-$HOME/models}"
  EMBED_MODEL="${EMBED_MODEL:-$MODELS/Qwen3-Embedding-8B-Q8_0.gguf}"
  LABEL_MODEL="${LABEL_MODEL:-$MODELS/gemma-3-4b-it-q4_0.gguf}"

  SPLATRAG_EMBED_PORT="${SPLATRAG_EMBED_PORT:-8081}"
  SPLATRAG_LABEL_PORT="${SPLATRAG_LABEL_PORT:-8082}"
  SPLATRAG_LENS_PORT="${SPLATRAG_LENS_PORT:-8767}"
  SPLATRAG_CLAIM_PORT="${SPLATRAG_CLAIM_PORT:-8766}"
  SPLATRAG_PALACE_PORT="${SPLATRAG_PALACE_PORT:-8080}"

  SPLATRAG_LOGS="${SPLATRAG_LOGS:-$SPLATRAG_RUNTIME/logs}"
  SPLATRAG_CONFIG="${SPLATRAG_CONFIG:-$SR_ROOT/splatrag.toml}"
  SPLATRAG_BIN="${SPLATRAG_BIN:-$SR_ROOT/target/release/splatrag}"
  SPLATRAG_PALACE_DIR="${SPLATRAG_PALACE_DIR:-$HOME/projects/splatrag-palace}"

  LLAMA="$(sr_find_llama || true)"

  mkdir -p "$SPLATRAG_LOGS"
}

# llama-server: explicit $LLAMA, then the conventional user-local install, then PATH.
sr_find_llama() {
  if [[ -n "${LLAMA:-}" && -x "${LLAMA}" ]]; then
    printf '%s\n' "$LLAMA"
    return 0
  fi
  if [[ -x "$HOME/.local/bin/llama-server" ]]; then
    printf '%s\n' "$HOME/.local/bin/llama-server"
    return 0
  fi
  local found
  if found="$(command -v llama-server 2>/dev/null)"; then
    printf '%s\n' "$found"
    return 0
  fi
  return 1
}

# --- small helpers -----------------------------------------------------------

have() { command -v "$1" >/dev/null 2>&1; }

sha256_of() {
  [[ -e "$1" ]] || return 1
  sha256sum "$1" | cut -d' ' -f1
}

# JSON string escape for hand-built receipt fields.
json_str() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

# --- service probes ----------------------------------------------------------

port_listening() {
  ss -ltn 2>/dev/null | grep -qE "127\.0\.0\.1:$1 "
}

health_up() {
  curl -sf --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

api_up() {
  curl -sf --max-time 2 "http://127.0.0.1:$1/api/status" >/dev/null 2>&1
}

http_ok() {
  local code
  code=$(curl -s --max-time 2 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/" 2>/dev/null || echo 000)
  [[ "$code" == "200" || "$code" == "302" || "$code" == "304" ]]
}

# --- build -------------------------------------------------------------------

ensure_bin() {
  if [[ -x "$SPLATRAG_BIN" ]]; then
    return 0
  fi
  note "building release binary (first run takes a few minutes)…"
  (cd "$SR_ROOT" && cargo build --release) || return 1
  [[ -x "$SPLATRAG_BIN" ]] || { bad "no binary at $SPLATRAG_BIN"; return 1; }
}

# --- llama-server lifecycle --------------------------------------------------
# SR_STARTED_EMBED / SR_STARTED_LABEL record whether *we* started a process, so
# callers only stop what they own and never kill a server the user had running.

# Exported because the reader is in another file: claim.sh's cleanup trap consults these
# to decide whether it owns the process it is about to stop.
export SR_STARTED_EMBED=0
export SR_STARTED_LABEL=0

start_embed() {
  if health_up "$SPLATRAG_EMBED_PORT"; then
    ok "embedder :$SPLATRAG_EMBED_PORT already up (leaving it alone)"
    return 0
  fi
  [[ -n "$LLAMA" && -x "$LLAMA" ]] || {
    bad "llama-server not found"
    note "set LLAMA=/path/to/llama-server in .env, or put it on PATH"
    note "build: https://github.com/ggml-org/llama.cpp"
    return 1
  }
  [[ -e "$EMBED_MODEL" ]] || {
    bad "embedding model missing: $EMBED_MODEL"
    note "set EMBED_MODEL in .env, or run ./claim.sh --fetch-models"
    return 1
  }
  note "starting embedder :$SPLATRAG_EMBED_PORT $(basename "$EMBED_MODEL")"
  # --ubatch-size is load-bearing: the embedder is charged one request for the sum of
  # a batch's tokens, and SplatRAG batches 32 memories at a time.
  # setsid: new session so mcp-for-silo.sh's later `exec mcp` does not leave the
  # 8B in the grok MCP process group (house `embedder up` is `nohup setsid`).
  nohup setsid "$LLAMA" --model "$EMBED_MODEL" --host 127.0.0.1 --port "$SPLATRAG_EMBED_PORT" \
    --embedding --pooling last --ubatch-size 2048 --batch-size 2048 \
    --n-gpu-layers 999 --flash-attn on \
    >"$SPLATRAG_LOGS/embed-${SPLATRAG_EMBED_PORT}.log" 2>&1 < /dev/null &
  echo $! >"$SPLATRAG_LOGS/embed.pid"
  export SR_STARTED_EMBED=1
  printf '    waiting'
  for _ in $(seq 90); do
    if health_up "$SPLATRAG_EMBED_PORT"; then
      echo
      ok "embedder :$SPLATRAG_EMBED_PORT up"
      return 0
    fi
    printf '.'
    sleep 2
  done
  echo
  bad "embedder :$SPLATRAG_EMBED_PORT failed to come up"
  note "log: $SPLATRAG_LOGS/embed-${SPLATRAG_EMBED_PORT}.log"
  tail -20 "$SPLATRAG_LOGS/embed-${SPLATRAG_EMBED_PORT}.log" 2>/dev/null || true
  return 1
}

start_label() {
  if health_up "$SPLATRAG_LABEL_PORT"; then
    ok "labeler :$SPLATRAG_LABEL_PORT already up"
    return 0
  fi
  [[ -n "$LLAMA" && -x "$LLAMA" ]] || { bad "llama-server not found"; return 1; }
  [[ -e "$LABEL_MODEL" ]] || {
    bad "label model missing: $LABEL_MODEL"
    note "the labeler is optional — the claim does not need it"
    return 1
  }
  note "starting labeler :$SPLATRAG_LABEL_PORT $(basename "$LABEL_MODEL")"
  nohup "$LLAMA" --model "$LABEL_MODEL" --host 127.0.0.1 --port "$SPLATRAG_LABEL_PORT" \
    --ctx-size 4096 --n-gpu-layers 999 --flash-attn on \
    >"$SPLATRAG_LOGS/label-${SPLATRAG_LABEL_PORT}.log" 2>&1 &
  echo $! >"$SPLATRAG_LOGS/label.pid"
  export SR_STARTED_LABEL=1
  printf '    waiting'
  for _ in $(seq 90); do
    if health_up "$SPLATRAG_LABEL_PORT"; then
      echo
      ok "labeler :$SPLATRAG_LABEL_PORT up"
      return 0
    fi
    printf '.'
    sleep 2
  done
  echo
  bad "labeler :$SPLATRAG_LABEL_PORT failed — see $SPLATRAG_LOGS/label-${SPLATRAG_LABEL_PORT}.log"
  return 1
}

stop_port_listeners() {
  if have fuser; then
    fuser -k "${1}/tcp" >/dev/null 2>&1 || true
  fi
}
