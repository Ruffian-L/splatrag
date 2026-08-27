#!/usr/bin/env bash
# Check everything ./claim.sh needs before it starts spending time.
#
#   scripts/preflight.sh            report and exit nonzero on a blocker
#   scripts/preflight.sh --quiet    only print problems
#
# Exit 0 = ready. Exit 2 = something is missing; every miss prints its own fix.
set -uo pipefail

SR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
# shellcheck source=scripts/lib.sh
. "$SR_LIB"
sr_load_env

QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1
problems=0

say_ok() { [[ $QUIET -eq 1 ]] || ok "$*"; }

# --- toolchain ---------------------------------------------------------------

if have cargo && have rustc; then
  say_ok "rust $(rustc --version | awk '{print $2}') · cargo $(cargo --version | awk '{print $2}')"
else
  bad "cargo/rustc not found"
  note "install: https://rustup.rs"
  problems=$((problems + 1))
fi

for tool in curl sha256sum python3; do
  if have "$tool"; then
    say_ok "$tool"
  else
    bad "$tool not found — required by the harness"
    problems=$((problems + 1))
  fi
done

if have unzip; then
  say_ok "unzip"
else
  warn "unzip not found — needed only to extract the SciFact download"
fi

# --- llama-server ------------------------------------------------------------

if [[ -n "$LLAMA" && -x "$LLAMA" ]]; then
  say_ok "llama-server  $LLAMA"
else
  bad "llama-server not found"
  note "checked: \$LLAMA, ~/.local/bin/llama-server, PATH"
  note "build it: https://github.com/ggml-org/llama.cpp  then set LLAMA= in .env"
  problems=$((problems + 1))
fi

# --- embedding model ---------------------------------------------------------
# The pin below is this project's reference copy. A different quantisation of the
# same model will run, but the published numbers are only comparable against this one.

EMBED_SHA_EXPECT="d20ddc71e8a5c4344f2343481e242233a997dc5eaff442427a945836c97b4deb"
EMBED_BYTES_EXPECT=8047105824

if [[ -e "$EMBED_MODEL" ]]; then
  bytes="$(stat -Lc %s "$EMBED_MODEL" 2>/dev/null || echo 0)"
  if [[ "$bytes" == "$EMBED_BYTES_EXPECT" ]]; then
    say_ok "embedding model  $(basename "$EMBED_MODEL")  ($bytes bytes, size matches pin)"
  else
    warn "embedding model  $(basename "$EMBED_MODEL")  ($bytes bytes, pin expects $EMBED_BYTES_EXPECT)"
    note "it will still run; the receipt records the hash actually used"
  fi
else
  bad "embedding model missing: $EMBED_MODEL"
  note "Qwen3-Embedding-8B-Q8_0.gguf  sha256 $EMBED_SHA_EXPECT"
  note "download: ./claim.sh --fetch-models      (or set EMBED_MODEL= in .env)"
  problems=$((problems + 1))
fi

# --- disk --------------------------------------------------------------------
# A full run writes ~90 GB-free-worth of nothing, but the store plus indexes for
# 10,183 documents at 4096 dimensions lands near 90 MB; the model download is the
# real cost. 12 GB gives room for both.

mkdir -p "$SPLATRAG_RUNTIME" 2>/dev/null || true
if avail_kb="$(df -Pk "$SPLATRAG_RUNTIME" 2>/dev/null | awk 'NR==2{print $4}')" && [[ -n "$avail_kb" ]]; then
  avail_gb=$((avail_kb / 1024 / 1024))
  if [[ "$avail_gb" -ge 12 ]]; then
    say_ok "disk  ${avail_gb} GB free at $SPLATRAG_RUNTIME"
  else
    bad "disk  only ${avail_gb} GB free at $SPLATRAG_RUNTIME (want 12+)"
    note "point SPLATRAG_RUNTIME= at a bigger volume in .env"
    problems=$((problems + 1))
  fi
else
  warn "could not read free space for $SPLATRAG_RUNTIME"
fi

# --- corpora -----------------------------------------------------------------

POISON="$SR_ROOT/corpora/poison/urban-dictionary.jsonl"
if [[ -e "$POISON" ]]; then
  say_ok "poison corpus  $(wc -l <"$POISON" | tr -d ' ') rows (shipped in repo)"
else
  bad "poison corpus missing: $POISON"
  note "it ships with the repo — a fresh clone should have it"
  problems=$((problems + 1))
fi

if [[ -d "$SPLATRAG_RUNTIME/datasets/scifact" ]]; then
  say_ok "SciFact present at $SPLATRAG_RUNTIME/datasets/scifact"
else
  [[ $QUIET -eq 1 ]] || note "SciFact not fetched yet — ./claim.sh will download it"
fi

# --- verdict -----------------------------------------------------------------

if [[ $problems -gt 0 ]]; then
  echo
  bad "$problems blocker(s) — fix the lines above, then re-run"
  exit 2
fi
[[ $QUIET -eq 1 ]] || { echo; ok "preflight clear"; }
exit 0
