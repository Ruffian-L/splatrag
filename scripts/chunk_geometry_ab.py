#!/usr/bin/env python3
"""Chunk geometry A/B. Not a language judge. No verdict column.

Arm A: embed(user + assistant) as one chunk.
Arm B: embed(assistant only); user is metadata, not in the vector.

Prints cosines and topic intra/inter separation for both clouds.
Does not say which arm is better.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "research_logs" / "2026-09-03_chunk-geometry-pairs.json"
OUT_PATH = ROOT / "research_logs" / "2026-09-03_chunk-geometry-ab.json"
EMBED_URL = "http://127.0.0.1:8081/v1/embeddings"


def l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def embed(texts: list[str]) -> list[list[float]]:
    body = json.dumps(
        {"model": "qwen", "input": texts, "encoding_format": "float"}
    ).encode()
    req = urllib.request.Request(
        EMBED_URL,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    items = sorted(payload["data"], key=lambda row: row["index"])
    if len(items) != len(texts):
        raise SystemExit(f"embed count {len(items)} != {len(texts)}")
    return [l2(row["embedding"]) for row in items]


def mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def topic_separation(ids: list[str], topics: list[str], vecs: list[list[float]]) -> dict:
    intra: list[float] = []
    inter: list[float] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            c = cos(vecs[i], vecs[j])
            if topics[i] == topics[j]:
                intra.append(c)
            else:
                inter.append(c)
    mi, mj = mean(intra), mean(inter)
    return {
        "n_intra_pairs": len(intra),
        "n_inter_pairs": len(inter),
        "mean_intra_cos": mi,
        "mean_inter_cos": mj,
        "intra_minus_inter": None if mi is None or mj is None else mi - mj,
        "intra": intra,
        "inter": inter,
    }


def main() -> int:
    pairs = json.loads(PAIRS_PATH.read_text())["pairs"]
    users = [p["user"] for p in pairs]
    assts = [p["assistant"] for p in pairs]
    concats = [f"User: {u}\nAssistant: {a}" for u, a in zip(users, assts)]
    topics = [p["topic"] for p in pairs]
    ids = [p["id"] for p in pairs]

    # One batch: users, assistants, concats. Geometry only.
    blob = embed(users + assts + concats)
    n = len(pairs)
    u_vec, a_vec, c_vec = blob[:n], blob[n : 2 * n], blob[2 * n :]

    rows = []
    for i, p in enumerate(pairs):
        rows.append(
            {
                "id": p["id"],
                "topic": p["topic"],
                "source": p["source"],
                "user_chars": len(p["user"]),
                "assistant_chars": len(p["assistant"]),
                "cos_user_to_concat": cos(u_vec[i], c_vec[i]),
                "cos_user_to_assistant": cos(u_vec[i], a_vec[i]),
                "cos_assistant_to_concat": cos(a_vec[i], c_vec[i]),
            }
        )

    # Cross-talk: user of i vs concat/assistant of j (same vs other topic)
    user_hits_concat_same = []
    user_hits_concat_other = []
    user_hits_asst_same = []
    user_hits_asst_other = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cc = cos(u_vec[i], c_vec[j])
            ca = cos(u_vec[i], a_vec[j])
            if topics[i] == topics[j]:
                user_hits_concat_same.append(cc)
                user_hits_asst_same.append(ca)
            else:
                user_hits_concat_other.append(cc)
                user_hits_asst_other.append(ca)

    report = {
        "protocol": "chunk-geometry-ab-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embed_url": EMBED_URL,
        "n_pairs": n,
        "topics": sorted(set(topics)),
        "arms": {
            "A_concat": "embed(User: …\\nAssistant: …) — one chunk",
            "B_assistant_only": "embed(assistant); user held as metadata, not in the vector",
        },
        "note": "No verdict. intra-inter is whether the cloud separates by topic label. Jason grades.",
        "rows": rows,
        "cluster_concat": topic_separation(ids, topics, c_vec),
        "cluster_assistant_only": topic_separation(ids, topics, a_vec),
        "cluster_user_only": topic_separation(ids, topics, u_vec),
        "user_query_to_other_chunks": {
            "concat_same_topic_mean": mean(user_hits_concat_same),
            "concat_other_topic_mean": mean(user_hits_concat_other),
            "assistant_same_topic_mean": mean(user_hits_asst_same),
            "assistant_other_topic_mean": mean(user_hits_asst_other),
        },
        "dim": len(c_vec[0]) if c_vec else 0,
    }
    # Drop bulky intra/inter lists from the written file; keep means.
    for key in ("cluster_concat", "cluster_assistant_only", "cluster_user_only"):
        report[key] = {
            k: v
            for k, v in report[key].items()
            if k not in ("intra", "inter")
        }

    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "n_pairs",
        "topics",
        "cluster_concat",
        "cluster_assistant_only",
        "cluster_user_only",
        "user_query_to_other_chunks",
        "dim",
    )}, indent=2))
    print(f"wrote {OUT_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(f"embedder down: {exc}", file=sys.stderr)
        raise SystemExit(2)
