#!/usr/bin/env python3
"""Write claim-receipt.json and RECEIPT.md from a finished handshake.

    scripts/write_receipt.py OUT_DIR

Reads the run's facts from the environment (claim.sh exports them) so nothing has
to be quoted through a shell heredoc. Every value that could make a number
different — model hash, corpus hashes, git sha, config — lands in the receipt, so
a disagreement between two runs can be traced instead of argued about.
"""

import json
import os
import sys

REFERENCE = {
    "source": "research_logs/2026-08-05_basin_local_sum.md",
    "baseline": {"recall_at_10": 0.8634, "ndcg_at_10": 0.7316},
    "after_poison": {"recall_at_10": 0.8568, "ndcg_at_10": 0.7303},
    "recall_drop": 0.0067,
}


def env(name, default=""):
    return os.environ.get(name, default)


def env_int(name):
    try:
        return int(os.environ.get(name, "0"))
    except ValueError:
        return 0


def main() -> int:
    out = sys.argv[1]
    smoke = env("SR_SMOKE") == "1"
    claim_a = env("SR_VERDICT_EXIT") == "0"

    with open(os.path.join(out, "handshake.json")) as fh:
        hs = json.load(fh)

    base = hs.get("baseline") or {}
    after = hs.get("after_poison")
    contract = hs.get("contract", {})
    min_r = contract.get("recall_at_10_min", 0.88)
    min_n = contract.get("ndcg_at_10_min", 0.75)

    def meets(m):
        return bool(m and m.get("recall_at_k", 0) >= min_r and m.get("ndcg_at_k", 0) >= min_n)

    claim_b = meets(base) and (meets(after) if after else True)

    config_path = os.path.join(out, "splatrag.claim.toml")
    config_text = ""
    if os.path.exists(config_path):
        with open(config_path) as fh:
            config_text = fh.read()

    model_sha = env("SR_EMBED_SHA")
    receipt = {
        "claim": (
            "SciFact -> Urban Dictionary poison -> dream -> SciFact; recall loses no more "
            "than 0.05 and the cold log is byte-identical across the dream"
        ),
        "smoke": smoke,
        "generated_utc": env("SR_STAMP"),
        "verdict": {
            "claim_a_poison_resistance": claim_a,
            "claim_b_absolute_floors": claim_b,
            "note": (
                "exit code follows claim A only; the handshake binary folds claim B into "
                "its own exit status, so it exits nonzero on a run that reproduces the claim"
            ),
        },
        "reference": REFERENCE,
        "build": {
            "git_sha": env("SR_GIT_SHA"),
            "git_state": env("SR_GIT_DIRTY"),
            "rustc": env("SR_RUSTC"),
            "cargo_lock_sha256": env("SR_LOCK_SHA"),
        },
        "model": {
            "path": env("SR_EMBED_MODEL"),
            "sha256": model_sha,
            "bytes": env_int("SR_EMBED_BYTES"),
            "matches_reference": model_sha == env("SR_EMBED_SHA_EXPECT"),
            "server": "llama-server --embedding --pooling last --ubatch-size 2048",
        },
        "corpora": {
            "dataset_dir": env("SR_DATASET"),
            "scifact_corpus_sha256": env("SR_SCIFACT_SHA"),
            "scifact_queries_sha256": env("SR_QUERIES_SHA"),
            "scifact_qrels_sha256": env("SR_QRELS_SHA"),
            "poison_path": env("SR_POISON_PATH"),
            "poison_sha256": env("SR_POISON_SHA"),
        },
        "config": config_text,
        "timing_seconds": {
            key: env_int(f"SR_T_{key.upper()}")
            for key in (
                "preflight", "model", "build", "corpora",
                "embedder", "config", "handshake", "total",
            )
        },
        "handshake": hs,
    }

    with open(os.path.join(out, "claim-receipt.json"), "w") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")

    total = env_int("SR_T_TOTAL")
    lines = [
        "# SplatRAG claim receipt" + ("  (SMOKE — not the claim)" if smoke else ""),
        "",
        "SciFact → Urban Dictionary poison → dream → SciFact, k=10.",
        "",
        "| | R@10 | nDCG@10 |",
        "| --- | ---: | ---: |",
    ]
    if base:
        lines.append(f"| baseline | {base['recall_at_k']:.4f} | {base['ndcg_at_k']:.4f} |")
    if after:
        lines.append(f"| after poison + dream | {after['recall_at_k']:.4f} | {after['ndcg_at_k']:.4f} |")
    if hs.get("recall_drop") is not None:
        lines.append(f"| recall drop | {hs['recall_drop']:.4f} | |")

    lines += [
        "",
        f"- **CLAIM A** (poison resistance, ≤0.05 loss + cold log preserved): "
        f"**{'HOLDS' if claim_a else 'FAILS'}**",
        f"- **CLAIM B** (absolute floors R@10 ≥ {min_r}, nDCG@10 ≥ {min_n}): "
        f"**{'MET' if claim_b else 'MISSED'}**",
        f"- cold log byte-identical across dream: `{hs.get('cold_preserved_during_dream')}`",
        f"- queries evaluated: {base.get('queries', 'n/a')}",
        "",
        "## Provenance",
        "",
        f"- git `{env('SR_GIT_SHA')}` ({env('SR_GIT_DIRTY')})",
        f"- {env('SR_RUSTC')}",
        f"- `Cargo.lock` sha256 `{env('SR_LOCK_SHA')}`",
        f"- embedding model sha256 `{model_sha}`",
        f"- SciFact corpus sha256 `{env('SR_SCIFACT_SHA')}`",
        f"- poison corpus sha256 `{env('SR_POISON_SHA')}`",
        f"- wall clock {total // 60}m {total % 60}s",
        "",
        "Full machine-readable detail in `claim-receipt.json`, including the exact config used.",
    ]
    if smoke:
        lines += [
            "",
            "> Reduced corpus and 100 physics steps. These numbers are a plumbing check "
            "and are not comparable to the published table.",
        ]

    with open(os.path.join(out, "RECEIPT.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
