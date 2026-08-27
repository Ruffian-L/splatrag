#!/usr/bin/env bash
# SplatRAG installer — repo copy. Makes `sp` work from any terminal.
set -euo pipefail

ENGINE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP="$ENGINE/scripts/sp"
BIN_DIR="${HOME}/.local/bin"
CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/splatrag"

say() { printf '%s\n' "$*"; }

[[ -x "$SP" ]] || chmod +x "$SP"

mkdir -p "$BIN_DIR" "$CONF_DIR" "$CONF_DIR/logs"

CONFIG="$ENGINE/splatrag.toml"
if [[ ! -f "$CONFIG" && -f "$ENGINE/examples/splatrag.example.toml" ]]; then
  say "No splatrag.toml yet. Copy examples/splatrag.example.toml and set data_dir, then re-run."
  CONFIG="$ENGINE/examples/splatrag.example.toml"
fi

cat >"$CONF_DIR/v3.env" <<EOF
SPLATRAG_ENGINE=$ENGINE
SPLATRAG_CONFIG=$CONFIG
SPLATRAG_BIN=$ENGINE/target/release/splatrag
SPLATRAG_LENS_PORT=8767
SPLATRAG_EMBED_PORT=8081
SPLATRAG_UNIT_SERVE=splatrag-serve.service
EOF

ln -sfn "$SP" "$BIN_DIR/sp"
chmod +x "$BIN_DIR/sp"

if [[ ! -x "$ENGINE/target/release/splatrag" ]]; then
  say "Building SplatRAG (first time takes a few minutes)…"
  (cd "$ENGINE" && cargo build --release)
fi

say ""
say "SplatRAG is installed."
say ""
say "  ./run init      write splatrag.toml"
say "  ./run up        embedder + service"
say "  sp              status"
say ""
if ! printf '%s' ":$PATH:" | grep -q ":$BIN_DIR:"; then
  say "If 'sp' is not found, add this to ~/.bashrc then open a new terminal:"
  say "  export PATH=\"\$BIN_DIR:\$PATH\""
fi
