#!/usr/bin/env bash
# Bundle the product picture (Basin Field) plus the wire contract it talks to.
#
#   scripts/pack_frontend.sh              → dist/splatrag-frontend-<stamp>.zip
#   scripts/pack_frontend.sh --out FILE   write somewhere specific
#
# Product is Downloads/jDsRnHGLMAZpLBS8-grok-workspace via `sp start`.
# palace/ is leftover — never packed as the app. Palace trees on disk are left alone.
# API.md stays in the zip.
#
# Refuses to produce a zip if anything that looks like a secret or a personal absolute path
# survives into the staged tree.
set -euo pipefail

SR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
# shellcheck source=scripts/lib.sh
. "$SR_LIB"
sr_load_env

OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$SR_ROOT/dist/splatrag-frontend-$STAMP.zip}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
BUNDLE="$STAGE/splatrag-frontend"
mkdir -p "$BUNDLE"

have zip || die "zip not found — install it (apt install zip)"

# Product picture: Basin Field. Never palace. Do not npm run palace.
if [[ -z "${SPLATRAG_VIEWER:-}" ]]; then
  if [[ -f "$HOME/Downloads/jDsRnHGLMAZpLBS8-grok-workspace/package.json" ]]; then
    SPLATRAG_VIEWER="$HOME/Downloads/jDsRnHGLMAZpLBS8-grok-workspace"
  elif [[ -f "$SR_ROOT/../splatrag-v3-viewer/package.json" ]]; then
    SPLATRAG_VIEWER="$(cd "$SR_ROOT/../splatrag-v3-viewer" && pwd)"
  fi
fi

# --- 1. Basin Field (the app) ------------------------------------------------

step "basin field"
if [[ -n "${SPLATRAG_VIEWER:-}" && -d "$SPLATRAG_VIEWER" && -f "$SPLATRAG_VIEWER/package.json" ]]; then
  mkdir -p "$BUNDLE/basin-field"
  for item in src scripts server migrations public package.json package-lock.json \
              vite.config.ts tsconfig.json eslint.config.mjs; do
    if [[ -e "$SPLATRAG_VIEWER/$item" ]]; then
      cp -r "$SPLATRAG_VIEWER/$item" "$BUNDLE/basin-field/"
    fi
  done
  rm -rf "$BUNDLE/basin-field/node_modules" "$BUNDLE/basin-field/dist" "$BUNDLE/basin-field/.git"
  # Postcard dump + research notes are leftover; they leak machine paths into the zip.
  rm -f "$BUNDLE/basin-field/public/field.json"
  rm -rf "$BUNDLE/basin-field/public/research"
  find "$BUNDLE/basin-field" \( -name '.env' -o -name '.env.local' -o -name '*.log' \) -print0 \
    | while IFS= read -r -d '' f; do
      rm -f "$f"
    done
  ok "basin-field/  ($(find "$BUNDLE/basin-field" -type f | wc -l | tr -d ' ') files)"
else
  warn "no Basin Field at ${SPLATRAG_VIEWER:-unset} — skipping the app"
  note "set SPLATRAG_VIEWER to Downloads/jDsRnHGLMAZpLBS8-grok-workspace"
  note "do not pack palace/; leftover, not the product"
fi

# palace/ is leftover. Do not copy it. Do not delete palace trees on disk.

# --- 2. leftover builtin viewer + the Rust side of the wire ------------------

step "rust api reference"
mkdir -p "$BUNDLE/rust-api" "$BUNDLE/builtin-viewer"
for f in src/web.rs src/wire.rs src/mcp.rs; do
  cp "$SR_ROOT/$f" "$BUNDLE/rust-api/$(basename "$f")"
done
# Stage-only: engine source may mention a machine path. Do not ship it. Disk source stays.
sed -i -E 's|/home/[A-Za-z0-9._-]+|/home/<user>|g; s|/Users/[A-Za-z0-9._-]+|/Users/<user>|g' \
  "$BUNDLE/rust-api"/*.rs
cp "$SR_ROOT"/assets/index.html "$SR_ROOT"/assets/app.js "$SR_ROOT"/assets/style.css \
  "$BUNDLE/builtin-viewer/"
ok "rust-api/  web.rs wire.rs mcp.rs"
ok "builtin-viewer/  leftover assets/ (not the product)"

# --- 3. generated contract + brief ------------------------------------------

step "docs"
cp "$SR_ROOT/scripts/frontend-brief/API.md" "$BUNDLE/API.md"
cp "$SR_ROOT/scripts/frontend-brief/BRIEF.md" "$BUNDLE/BRIEF.md"
ok "API.md  BRIEF.md"

# --- 4. leftover + secret gate ----------------------------------------------
# Do not ship a zip that leaks a key, hardcodes a home path, or treats palace as the app.

step "secret scan"
violations=0
scan() {
  local label="$1" pattern="$2"
  local hits
  # Exclude package-lock.json: npm integrity hashes trip broad patterns.
  hits="$(grep -rIlE "$pattern" "$BUNDLE" 2>/dev/null | grep -v 'package-lock.json' || true)"
  if [[ -n "$hits" ]]; then
    bad "$label"
    printf '%s\n' "$hits" | sed 's|^|      |'
    violations=$((violations + 1))
  fi
}
scan "api key literal"        'sk-[A-Za-z0-9]{16,}'
scan "private key block"      'BEGIN [A-Z ]*PRIVATE KEY'
scan "aws key"                'AKIA[0-9A-Z]{16}'
scan "personal absolute path" '(/home/[a-z]|/Users/[a-z]|/media/[a-z])'

if find "$BUNDLE" -name '.env' -o -name '.env.local' | grep -q .; then
  bad "a live .env survived staging"
  violations=$((violations + 1))
fi
if [[ -d "$BUNDLE/basin-field/node_modules" ]]; then
  bad "node_modules survived staging"
  violations=$((violations + 1))
fi
if [[ -d "$BUNDLE/palace" ]]; then
  bad "palace/ staged as the app — leftover, not the product"
  violations=$((violations + 1))
fi

if [[ $violations -gt 0 ]]; then
  die "$violations problem(s) — refusing to write a zip"
fi
ok "clean: no keys, no live .env, no node_modules, no personal paths, no palace/"

# --- 5. zip ------------------------------------------------------------------

step "writing bundle"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" splatrag-frontend)
ok "$OUT"
note "$(du -h "$OUT" | cut -f1) · $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files"
echo
c_dim "  unzip -l \"$OUT\"    # inspect before sending"
