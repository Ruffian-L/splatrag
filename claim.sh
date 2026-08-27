#!/usr/bin/env bash
#
#   ./claim.sh              reproduce the claim, end to end
#   ./claim.sh --smoke      15-minute plumbing check on a reduced corpus (NOT the claim)
#   ./claim.sh --help       every flag
#
# THE CLAIM
#
#   Ingest SciFact. Measure Recall@10 and nDCG@10. Ingest 5,000 Urban Dictionary entries into
#   the same cold store. Dream. Measure again.
#
#   A — poison resistance: recall loses no more than 0.05, and the cold log is byte-for-byte
#       identical before and after the dream. Nothing is deleted to achieve this.
#
#       SCOPE: the evaluator filters to domain="scifact" (src/eval.rs), so poison documents
#       cannot rank as false positives. This measures that noise in the same store does not
#       degrade retrieval of the good documents. It is NOT a claim that the geometry found
#       the junk on its own: domain is stamped at ingest and the dream's cross_domain_repulsion
#       acts on that stamp. An unsupervised-structure claim needs a single-domain control run.
#   B — absolute floors: Recall@10 >= 0.88 and nDCG@10 >= 0.75.
#
#   A is the claim and this script's exit code follows it. B is an aspirational contract the
#   measured runs miss, and it is reported as such — see research_logs/.
#
# The `handshake` subcommand folds B into its own exit status, so it exits nonzero on a run
# that reproduces the claim perfectly. That is why this script reads the JSON instead of the
# exit code.
#
set -uo pipefail

SR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/lib.sh"
# shellcheck source=scripts/lib.sh
. "$SR_LIB"
sr_load_env

# The published reference numbers live in scripts/verdict.py and scripts/write_receipt.py,
# so the comparison table and the receipt can never quote different figures.

EMBED_SHA_EXPECT="d20ddc71e8a5c4344f2343481e242233a997dc5eaff442427a945836c97b4deb"
HF_REPO="Qwen/Qwen3-Embedding-8B-GGUF"
HF_FILE="Qwen3-Embedding-8B-Q8_0.gguf"

# --- flags -------------------------------------------------------------------

SMOKE=0
DO_POISON=1
FETCH_MODELS=0
KEEP_EMBEDDER=0
ASSUME_YES=0
REUSE=0
OUT=""
POISON_PATH="$SR_ROOT/corpora/poison/urban-dictionary.jsonl"
EVAL_LIMIT=""

usage() {
  # The header comment block, up to the first line that is not a comment.
  awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "$0"
  cat <<EOF

Flags
  --smoke            reduced corpus + 25 queries + 100 physics steps. Plumbing only.
  --no-poison        SciFact baseline only; skips the poison half of the claim.
  --poison PATH      use a different noise corpus (default: corpora/poison/urban-dictionary.jsonl)
  --limit N          evaluate only the first N queries (development sampling)
  --fetch-models     download the embedding model if it is missing
  --keep-embedder    leave llama-server running when the script exits
  --out DIR          receipt directory (default: receipts/claim-<stamp>)
  --reuse            reuse an existing store instead of demanding a fresh one
  -y, --yes          do not pause for confirmation
  -h, --help         this

Cost
  A full run embeds 10,183 documents through an 8B model and dreams 1,200 steps.
  Budget roughly an hour. --smoke does the same shape in ~15 minutes and proves the
  plumbing, but its numbers are not the claim and are labelled so everywhere.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)         SMOKE=1; shift ;;
    --no-poison)     DO_POISON=0; shift ;;
    --poison)        POISON_PATH="$2"; shift 2 ;;
    --limit)         EVAL_LIMIT="$2"; shift 2 ;;
    --fetch-models)  FETCH_MODELS=1; shift ;;
    --keep-embedder) KEEP_EMBEDDER=1; shift ;;
    --out)           OUT="$2"; shift 2 ;;
    --reuse)         REUSE=1; shift ;;
    -y|--yes)        ASSUME_YES=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) bad "unknown flag: $1"; echo; usage; exit 1 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT:-$SR_ROOT/receipts/claim-$STAMP}"
mkdir -p "$OUT"
PREFIX=""
[[ $SMOKE -eq 1 ]] && PREFIX="SMOKE — "

t_start=$(date +%s)
declare -A TIMING

banner() {
  c_head ""
  if [[ $SMOKE -eq 1 ]]; then
    c_warn "  SPLATRAG — SMOKE RUN.  Plumbing check on a reduced corpus."
    c_warn "  The numbers below are NOT the claim. Run without --smoke for that."
  else
    c_head "  SPLATRAG — reproducing the claim"
    c_dim  "  SciFact → measure → poison → dream → measure"
  fi
  c_dim "  receipt: $OUT"
}
banner

# Invoked by trap, not called directly.
# shellcheck disable=SC2317
cleanup() {
  if [[ $SR_STARTED_EMBED -eq 1 && $KEEP_EMBEDDER -eq 0 ]]; then
    note "stopping the embedder this script started"
    stop_port_listeners "$SPLATRAG_EMBED_PORT"
  fi
}
trap cleanup EXIT

# ============================================================================
step "1/9  preflight"
# ============================================================================
t0=$(date +%s)
if ! "$SR_ROOT/scripts/preflight.sh" --quiet; then
  # preflight already printed each problem and its fix
  exit 2
fi
ok "toolchain, llama-server, disk, corpora"
TIMING[preflight]=$(( $(date +%s) - t0 ))

# ============================================================================
step "2/9  embedding model"
# ============================================================================
t0=$(date +%s)
if [[ ! -e "$EMBED_MODEL" ]]; then
  if [[ $FETCH_MODELS -eq 1 ]]; then
    have huggingface-cli || die "huggingface-cli not found — pip install huggingface_hub[cli]"
    mkdir -p "$MODELS"
    note "downloading $HF_FILE (~8 GB)"
    huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$MODELS" \
      || die "download failed"
  else
    bad "embedding model missing: $EMBED_MODEL"
    echo
    c_dim "  Download it:"
    c_dim "    huggingface-cli download $HF_REPO $HF_FILE --local-dir \"$MODELS\""
    c_dim "  or re-run with --fetch-models."
    echo
    c_dim "  Reference copy: 8,047,105,824 bytes"
    c_dim "  sha256 $EMBED_SHA_EXPECT"
    exit 2
  fi
fi

note "hashing the model (8 GB, ~20s)…"
EMBED_SHA="$(sha256_of "$EMBED_MODEL")"
EMBED_BYTES="$(stat -Lc %s "$EMBED_MODEL")"
if [[ "$EMBED_SHA" == "$EMBED_SHA_EXPECT" ]]; then
  ok "$(basename "$EMBED_MODEL")  sha256 matches the reference copy"
else
  warn "$(basename "$EMBED_MODEL")  sha256 differs from the reference copy"
  note "got      $EMBED_SHA"
  note "expected $EMBED_SHA_EXPECT"
  note "it will run; the receipt records the hash actually used, so the difference is not silent"
fi
TIMING[model]=$(( $(date +%s) - t0 ))

# ============================================================================
step "3/9  build"
# ============================================================================
t0=$(date +%s)
ensure_bin || exit 2
GIT_SHA="$(git -C "$SR_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="clean"
if ! git -C "$SR_ROOT" diff --quiet 2>/dev/null || ! git -C "$SR_ROOT" diff --cached --quiet 2>/dev/null; then
  GIT_DIRTY="dirty"
fi
RUSTC_V="$(rustc --version)"
LOCK_SHA="$(sha256_of "$SR_ROOT/Cargo.lock")"
ok "$(basename "$SPLATRAG_BIN")  git ${GIT_SHA:0:12} ($GIT_DIRTY)"
note "$RUSTC_V"
[[ "$GIT_DIRTY" == "dirty" ]] && warn "working tree has uncommitted changes — recorded in the receipt"
TIMING[build]=$(( $(date +%s) - t0 ))

# ============================================================================
step "4/9  corpora"
# ============================================================================
t0=$(date +%s)
"$SR_ROOT/scripts/fetch_datasets.sh" >"$OUT/datasets.log" 2>&1 || {
  cat "$OUT/datasets.log"
  die "corpora not ready"
}
grep -E '^\s+(✓|!)' "$OUT/datasets.log" | sed 's/^/  /' || true

DATASET="$SPLATRAG_RUNTIME/datasets/scifact"
SCIFACT_SHA="$(sha256_of "$DATASET/corpus.jsonl")"
QUERIES_SHA="$(sha256_of "$DATASET/queries.jsonl")"
QRELS_SHA="$(sha256_of "$DATASET/qrels/test.tsv")"
POISON_SHA_ACTUAL="$(sha256_of "$POISON_PATH" 2>/dev/null || echo missing)"

if [[ $SMOKE -eq 1 ]]; then
  SMOKE_DIR="$OUT/smoke-dataset"
  note "building the reduced smoke corpus"
  python3 - "$DATASET" "$SMOKE_DIR" <<'PY'
import json, os, sys, itertools
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(dst, "qrels"), exist_ok=True)

# First 25 test queries and everything they mark relevant.
rows, qids, gold = [], [], set()
with open(os.path.join(src, "qrels", "test.tsv")) as fh:
    header = fh.readline()
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        if parts[0] not in qids:
            if len(qids) >= 25:
                continue
            qids.append(parts[0])
        rows.append(line)
        gold.add(parts[1])
qids = set(qids)
rows = [r for r in rows if r.split("\t")[0] in qids]

with open(os.path.join(dst, "qrels", "test.tsv"), "w") as fh:
    fh.write(header)
    fh.writelines(rows)

with open(os.path.join(src, "queries.jsonl")) as fh, \
     open(os.path.join(dst, "queries.jsonl"), "w") as out:
    for line in fh:
        if line.strip() and json.loads(line)["_id"] in qids:
            out.write(line)

# Every gold document, plus 500 distractors so ranking still has to work.
kept = distractors = 0
with open(os.path.join(src, "corpus.jsonl")) as fh, \
     open(os.path.join(dst, "corpus.jsonl"), "w") as out:
    for line in fh:
        if not line.strip():
            continue
        doc = json.loads(line)["_id"]
        if doc in gold:
            out.write(line); kept += 1
        elif distractors < 500:
            out.write(line); distractors += 1
print(f"    {len(qids)} queries · {kept} gold + {distractors} distractor documents")
PY
  DATASET="$SMOKE_DIR"
  SMOKE_POISON="$OUT/smoke-poison.jsonl"
  head -500 "$POISON_PATH" >"$SMOKE_POISON"
  POISON_PATH="$SMOKE_POISON"
  EVAL_LIMIT="${EVAL_LIMIT:-25}"
  ok "smoke corpus built (this is not the claim)"
fi
TIMING[corpora]=$(( $(date +%s) - t0 ))

# ============================================================================
step "5/9  embedder"
# ============================================================================
t0=$(date +%s)
start_embed || exit 2
TIMING[embedder]=$(( $(date +%s) - t0 ))

# ============================================================================
step "6/9  config"
# ============================================================================
t0=$(date +%s)
DATA_DIR="$SPLATRAG_RUNTIME/stores/claim-$STAMP"
[[ $SMOKE -eq 1 ]] && DATA_DIR="$SPLATRAG_RUNTIME/stores/claim-smoke-$STAMP"
CONFIG_OUT="$OUT/splatrag.claim.toml"
PHYSICS_STEPS=1200
[[ $SMOKE -eq 1 ]] && PHYSICS_STEPS=100

# Freeze guard. This script writes a new store every run and must never be able to
# reach an existing measured one. Both halves matter: the name has to be ours, and
# the directory has to be empty of a cold log.
case "$(basename "$DATA_DIR")" in
  claim-*) ;;
  *) die "refusing to run: store name '$(basename "$DATA_DIR")' is not a claim-* store" ;;
esac
if [[ -e "$DATA_DIR/cold/memories.jsonl" ]]; then
  if [[ $REUSE -eq 1 ]]; then
    warn "reusing the existing store at $DATA_DIR"
  else
    die "store already holds a cold log: $DATA_DIR (pass --reuse to measure against it)"
  fi
fi

sed -e "s|__DATA_DIR__|$DATA_DIR|g" \
    -e "s|__SCOPE_KEY__|splatrag-claim|g" \
    -e "s|__EMBED_URL__|http://127.0.0.1:$SPLATRAG_EMBED_PORT|g" \
    -e "s|__PHYSICS_STEPS__|$PHYSICS_STEPS|g" \
    -e "s|__BIND__|127.0.0.1:$SPLATRAG_CLAIM_PORT|g" \
    "$SR_ROOT/configs/claim.toml.in" >"$CONFIG_OUT"

ok "fresh store  $DATA_DIR"
note "config      $CONFIG_OUT"
note "physics     $PHYSICS_STEPS steps · basin local-sum on (weight 0.3, threshold 0.50)"
TIMING[config]=$(( $(date +%s) - t0 ))

# ============================================================================
step "7/9  handshake"
# ============================================================================
if [[ $ASSUME_YES -eq 0 && $SMOKE -eq 0 ]]; then
  echo
  c_dim "  About to embed $(wc -l <"$DATASET/corpus.jsonl" | tr -d ' ') SciFact documents"
  [[ $DO_POISON -eq 1 ]] && c_dim "  plus $(wc -l <"$POISON_PATH" | tr -d ' ') poison entries, then dream $PHYSICS_STEPS steps."
  c_dim "  This takes about an hour. Ctrl-C is safe — the store is disposable."
  printf '  continue? [Y/n] '
  read -r reply
  [[ "$reply" =~ ^[Nn] ]] && { note "stopped"; exit 0; }
fi

t0=$(date +%s)
hs_args=(--config "$CONFIG_OUT" handshake --dataset "$DATASET")
[[ $DO_POISON -eq 1 ]] && hs_args+=(--poison "$POISON_PATH")
[[ -n "$EVAL_LIMIT" ]] && hs_args+=(--limit "$EVAL_LIMIT")

note "${PREFIX}running — progress on stderr, tailing $OUT/handshake.log"
echo
( cd "$SR_ROOT" && "$SPLATRAG_BIN" "${hs_args[@]}" ) \
  >"$OUT/handshake.json" 2>"$OUT/handshake.log"
HS_EXIT=$?
TIMING[handshake]=$(( $(date +%s) - t0 ))

if [[ ! -s "$OUT/handshake.json" ]]; then
  bad "the handshake produced no JSON (exit $HS_EXIT)"
  note "last lines of $OUT/handshake.log:"
  tail -20 "$OUT/handshake.log" | sed 's/^/      /'
  exit 2
fi
ok "handshake finished in $(( TIMING[handshake] / 60 ))m $(( TIMING[handshake] % 60 ))s (binary exit $HS_EXIT)"
[[ $HS_EXIT -ne 0 ]] && note "nonzero exit is expected — the binary folds the absolute floors into its status"

# ============================================================================
step "8/9  verdict"
# ============================================================================
VERDICT_ARGS=("$OUT/handshake.json")
[[ $SMOKE -eq 1 ]] && VERDICT_ARGS+=(--smoke)
"$SR_ROOT/scripts/verdict.py" "${VERDICT_ARGS[@]}"
VERDICT_EXIT=$?

# ============================================================================
step "9/9  receipt"
# ============================================================================
TIMING[total]=$(( $(date +%s) - t_start ))

# Export the run's facts; the writer reads them from the environment so nothing
# has to survive shell quoting.
export SR_STAMP="$STAMP" SR_SMOKE="$SMOKE" SR_VERDICT_EXIT="$VERDICT_EXIT"
export SR_GIT_SHA="$GIT_SHA" SR_GIT_DIRTY="$GIT_DIRTY" SR_RUSTC="$RUSTC_V" SR_LOCK_SHA="$LOCK_SHA"
export SR_EMBED_MODEL="$EMBED_MODEL" SR_EMBED_SHA="$EMBED_SHA" SR_EMBED_BYTES="$EMBED_BYTES"
export SR_EMBED_SHA_EXPECT="$EMBED_SHA_EXPECT"
export SR_DATASET="$DATASET" SR_SCIFACT_SHA="$SCIFACT_SHA" SR_QUERIES_SHA="$QUERIES_SHA"
export SR_QRELS_SHA="$QRELS_SHA" SR_POISON_PATH="$POISON_PATH" SR_POISON_SHA="$POISON_SHA_ACTUAL"
for k in preflight model build corpora embedder config handshake total; do
  export "SR_T_${k^^}=${TIMING[$k]:-0}"
done

"$SR_ROOT/scripts/write_receipt.py" "$OUT" || die "failed to write the receipt"

ok "claim-receipt.json"
ok "RECEIPT.md"
note "$OUT"
echo
c_dim "  store kept at $DATA_DIR — delete it when you are done"
[[ $KEEP_EMBEDDER -eq 1 ]] && c_dim "  embedder left running on :$SPLATRAG_EMBED_PORT"

exit $VERDICT_EXIT
