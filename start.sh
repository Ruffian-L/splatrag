#!/usr/bin/env bash
# Thin wrapper → peer-style cmd surface at bin/splatrag
#
#   ./start.sh          same as: splatrag up   (embed + lens + palace)
#   ./start.sh models   same as: splatrag embed && splatrag label
#   ./start.sh embed    same as: splatrag embed
#   ./start.sh palace   same as: splatrag palace
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SPLAT="$ROOT/bin/splatrag"
if [[ ! -x "$SPLAT" ]]; then
  echo "missing $SPLAT" >&2
  exit 1
fi
case "${1:-up}" in
  models) "$SPLAT" embed; "$SPLAT" label ;;
  all|up|"") "$SPLAT" up ;;
  *) "$SPLAT" "$@" ;;
esac
