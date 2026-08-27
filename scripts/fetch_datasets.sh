#!/usr/bin/env bash
# Fetch and verify the corpora the HANDSHAKE measures against.
#
#   scripts/fetch_datasets.sh           fetch what is missing, verify what is present
#   scripts/fetch_datasets.sh --verify  verify only, never download
#
# SciFact is downloaded from BEIR. The poison corpus ships in the repo. Both are pinned by
# sha256 below — a mismatch is reported loudly, because the published numbers are only
# comparable against these exact bytes.
set -uo pipefail

SR_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
# shellcheck source=scripts/lib.sh
. "$SR_LIB"
sr_load_env

VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

DATASETS="$SPLATRAG_RUNTIME/datasets"
SCIFACT="$DATASETS/scifact"
POISON="$SR_ROOT/corpora/poison/urban-dictionary.jsonl"

# BEIR SciFact. This is the canonical public mirror the BEIR authors publish.
SCIFACT_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

# path-relative-to-scifact  sha256  expected-lines
SCIFACT_PINS=(
  "corpus.jsonl      dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6 5183"
  "queries.jsonl     8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313 1109"
  "qrels/test.tsv    0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c 340"
)
POISON_SHA="43d4d451a2288351b430928636b9b2d4fb17541b8baae7f44f8704022d657a7e"

mismatch=0
missing=0

verify_one() {
  local path="$1" want="$2" lines="$3" label="$4"
  if [[ ! -e "$path" ]]; then
    bad "$label missing: $path"
    missing=$((missing + 1))
    return 1
  fi
  local got
  got="$(sha256_of "$path")"
  if [[ "$got" == "$want" ]]; then
    ok "$label  $(wc -l <"$path" | tr -d ' ') lines  sha256 ok"
  else
    warn "$label  sha256 differs from the pin"
    note "expected $want"
    note "got      $got"
    note "expected $lines lines, found $(wc -l <"$path" | tr -d ' ')"
    note "the harness will still run; results are not comparable to the published table"
    mismatch=$((mismatch + 1))
  fi
  return 0
}

# --- poison (shipped) --------------------------------------------------------

step "poison corpus"
verify_one "$POISON" "$POISON_SHA" 5000 "urban-dictionary.jsonl" || {
  note "this file ships with the repo — a fresh clone has it at corpora/poison/"
}

# --- scifact (fetched) -------------------------------------------------------

step "SciFact"
need_fetch=0
for pin in "${SCIFACT_PINS[@]}"; do
  read -r rel _sha _lines <<<"$pin"
  [[ -e "$SCIFACT/$rel" ]] || need_fetch=1
done

if [[ $need_fetch -eq 1 ]]; then
  if [[ $VERIFY_ONLY -eq 1 ]]; then
    bad "SciFact not present at $SCIFACT"
    note "fetch it: scripts/fetch_datasets.sh"
    exit 2
  fi
  have curl  || die "curl not found"
  have unzip || die "unzip not found (apt install unzip)"

  mkdir -p "$DATASETS"
  zip_path="$DATASETS/scifact.zip"
  note "downloading BEIR SciFact (~3 MB)"
  note "$SCIFACT_URL"
  if ! curl -fL --progress-bar -o "$zip_path" "$SCIFACT_URL"; then
    bad "download failed"
    note "if the BEIR mirror is down, fetch scifact yourself so that these exist:"
    note "  $SCIFACT/corpus.jsonl"
    note "  $SCIFACT/queries.jsonl"
    note "  $SCIFACT/qrels/test.tsv"
    note "the sha256 pins are listed at the top of this script"
    exit 2
  fi
  note "extracting"
  unzip -qo "$zip_path" -d "$DATASETS"
  rm -f "$zip_path"
  ok "extracted to $SCIFACT"
fi

for pin in "${SCIFACT_PINS[@]}"; do
  read -r rel sha lines <<<"$pin"
  verify_one "$SCIFACT/$rel" "$sha" "$lines" "$rel"
done

# --- verdict -----------------------------------------------------------------

echo
if [[ $missing -gt 0 ]]; then
  bad "$missing file(s) missing"
  exit 2
fi
if [[ $mismatch -gt 0 ]]; then
  warn "$mismatch file(s) differ from the pinned bytes — see above"
  # Not fatal: a reproducer may deliberately swap a corpus. The receipt records
  # the hashes actually used, so the difference is never silent.
fi
ok "corpora ready"
echo "SPLATRAG_DATASETS=$DATASETS"
exit 0
