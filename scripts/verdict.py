#!/usr/bin/env python3
"""Score a handshake.json against the two claims and print the comparison table.

    scripts/verdict.py handshake.json [--smoke]

Exit 0 when CLAIM A (poison resistance) holds or was not run, 1 when it fails.
CLAIM B (the absolute floors) is reported but never decides the exit code — the
measured runs miss it, and that gap is published, not hidden.
"""

import argparse
import json
import sys

# The published basin-local-sum run: research_logs/2026-08-05_basin_local_sum.md,
# SciFact full-300 at k=10 with Qwen3-Embedding-8B.
REFERENCE = {
    "baseline": (0.8634, 0.7316),
    "after_poison": (0.8568, 0.7303),
    "recall_drop": 0.0067,
}

BOLD, DIM, GRN, RED, YEL, OFF = (
    ("\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m")
    if sys.stdout.isatty()
    else ("", "", "", "", "", "")
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("handshake")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.handshake) as fh:
        hs = json.load(fh)

    base = hs.get("baseline")
    after = hs.get("after_poison")
    drop = hs.get("recall_drop")
    preserved = hs.get("cold_preserved_during_dream")
    contract = hs.get("contract", {})
    min_r = contract.get("recall_at_10_min", 0.88)
    min_n = contract.get("ndcg_at_10_min", 0.75)
    max_drop = contract.get("max_recall_drop", 0.05)
    tag = "SMOKE — " if args.smoke else ""

    def row(label, metrics, ref_key):
        if not metrics:
            return
        ref = ""
        if not args.smoke:
            r_r, r_n = REFERENCE[ref_key]
            ref = f"{DIM}reference {r_r:.4f} / {r_n:.4f}{OFF}"
        print(
            f"    {label:<24} {metrics['recall_at_k']:.4f}   "
            f"{metrics['ndcg_at_k']:.4f}   {ref}"
        )

    print()
    print(f"    {BOLD}{'':<24} {'R@10':<8} {'nDCG@10':<8}{OFF}")
    row("baseline", base, "baseline")
    row("after poison + dream", after, "after_poison")
    if drop is not None:
        ref = "" if args.smoke else f"{DIM}reference {REFERENCE['recall_drop']:.4f}{OFF}"
        print(f"    {'recall drop':<24} {drop:.4f}            {ref}")
    if base:
        print(f"    {DIM}{base.get('queries', '?')} queries evaluated at k=10{OFF}")
    print()

    # --- CLAIM A: poison resistance. This is the claim. ---------------------
    if after is None:
        print(f"    {YEL}CLAIM A  poison resistance ....... NOT RUN{OFF}")
        print(f"    {DIM}         --no-poison was passed; the claim needs the poison half{OFF}")
        claim_a = None
    else:
        drop_ok = drop is not None and drop <= max_drop
        claim_a = bool(drop_ok and preserved)
        colour = GRN if claim_a else RED
        print(
            f"    {colour}{BOLD}{tag}CLAIM A  poison resistance ....... "
            f"{'HOLDS' if claim_a else 'FAILS'}{OFF}"
        )
        print(f"      {'✓' if drop_ok else '✗'} recall drop {drop:.4f} <= {max_drop}")
        print(f"      {'✓' if preserved else '✗'} cold log byte-identical across the dream")
        print()
        print(f"      {DIM}Scope: the evaluator filters to domain=\"scifact\" (src/eval.rs), so the{OFF}")
        print(f"      {DIM}poison cannot appear as a false positive. This measures that adding noise{OFF}")
        print(f"      {DIM}to the same store does not degrade retrieval of the good documents. It is{OFF}")
        print(f"      {DIM}NOT evidence that the geometry discovered the junk unsupervised — domain is{OFF}")
        print(f"      {DIM}stamped at ingest and cross_domain_repulsion acts on it. See the bust note.{OFF}")

    # --- CLAIM B: the absolute floors. Reported, never the exit code. -------
    def meets(m):
        return bool(m and m["recall_at_k"] >= min_r and m["ndcg_at_k"] >= min_n)

    claim_b = meets(base) and (meets(after) if after else True)
    colour = GRN if claim_b else YEL
    print()
    print(f"    {colour}{tag}CLAIM B  absolute floors ......... {'MET' if claim_b else 'MISSED'}{OFF}")
    print(f"      {DIM}targets R@10 >= {min_r}, nDCG@10 >= {min_n}{OFF}")
    if not claim_b and not args.smoke:
        print(f"      {DIM}the published runs miss these too — see research_logs/{OFF}")
        print(f"      {DIM}this is a known gap, not a failed reproduction{OFF}")
    print()

    if args.smoke:
        print(f"    {YEL}Reduced corpus. These numbers are not comparable to anything.{OFF}")
        print(f"    {YEL}Re-run without --smoke to reproduce the claim.{OFF}")
        print()

    return 1 if claim_a is False else 0


if __name__ == "__main__":
    sys.exit(main())
