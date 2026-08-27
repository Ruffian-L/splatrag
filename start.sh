#!/usr/bin/env bash
# Kept for muscle memory. The real surface is ./run — see `./run --help`.
#
#   ./start.sh          →  ./run up      (embedder + service)
#   ./start.sh models   →  embedder + labeler
#   ./start.sh <cmd>    →  ./run <cmd>
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/run"
[[ -x "$RUN" ]] || { echo "missing $RUN" >&2; exit 1; }
case "${1:-up}" in
  models) "$RUN" embed && "$RUN" label ;;
  all|up|"") "$RUN" up ;;
  *) "$RUN" "$@" ;;
esac
