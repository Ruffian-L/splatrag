#!/usr/bin/env bash
# Pair a changelog why with a subject-sliced research log.
# Canonical log folder here is research/logs/ (see research/INDEX.md).
# Usage:
#   scripts/pair_log.sh --why "short why" --title "Subject title" [--hyp "we think X"]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHY=""
TITLE=""
HYP=""
AGENT="${RESEARCH_AGENT:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --why) WHY="${2:-}"; shift 2 ;;
    --title) TITLE="${2:-}"; shift 2 ;;
    --hyp|--hypothesis) HYP="${2:-}"; shift 2 ;;
    --agent) AGENT="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,6p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$WHY" || -z "$TITLE" ]]; then
  echo "need --why and --title" >&2
  exit 2
fi

LOGDIR="$ROOT/research/logs"
LOGREL="research/logs"
mkdir -p "$LOGDIR"

DATE="$(date -u +%Y-%m-%d)"
slug="$(printf '%s' "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//; s/-{2,}/-/g' | cut -c1-60)"
[[ -n "$slug" ]] || slug="entry"
LOGFILE="$LOGDIR/${DATE}_${slug}.md"
n=2
while [[ -e "$LOGFILE" ]]; do
  LOGFILE="$LOGDIR/${DATE}_${slug}-$n.md"
  n=$((n + 1))
done

CHANGELOG="$ROOT/CHANGELOG.md"
if [[ ! -f "$CHANGELOG" ]]; then
  cat > "$CHANGELOG" <<'EOF'
# Changelog

Research repo. Not production unless Jason says so.
Short whys here. Longer subject logs in research/logs/.
Local agent contract: `AGENTS.md` (gitignored).

EOF
fi

# Newest-first: splice before the first ^##  (>> would bury the latest swarm row).
entry="$(mktemp "${CHANGELOG}.entry.XXXXXX")"
out="$(mktemp "${CHANGELOG}.XXXXXX")"
trap 'rm -f "$entry" "$out"' EXIT
{
  echo "## ${DATE} — ${TITLE}"
  echo
  echo "- ${WHY}"
  if [[ -n "$HYP" ]]; then
    echo "- Hypothesis: ${HYP}"
  fi
  echo "- Research: \`${LOGREL}/$(basename "$LOGFILE")\`"
  if [[ -n "$AGENT" ]]; then
    echo "- Agent: ${AGENT}"
  fi
  echo
} > "$entry"
awk -v e="$entry" '
  /^## / && !ins {
    while ((getline l < e) > 0) print l
    close(e)
    ins = 1
  }
  { print }
  END {
    if (!ins) {
      while ((getline l < e) > 0) print l
    }
  }
' "$CHANGELOG" > "$out"
mv "$out" "$CHANGELOG"
trap 'rm -f "$entry"' EXIT

{
  echo "# ${TITLE}"
  echo
  echo "> Date: ${DATE}"
  if [[ -n "$AGENT" ]]; then
    echo "> Agent: ${AGENT}"
  fi
  echo "> Repo: splatrag-clean"
  echo
  echo "## Context"
  echo
  echo "$WHY"
  echo
  echo "## Hypothesis"
  echo
  echo "${HYP:-}"
  echo
  echo "## Next"
  echo
  echo "See CHANGELOG.md."
} > "$LOGFILE"

echo "changelog: $CHANGELOG"
echo "research:  $LOGFILE"
