#!/usr/bin/env python3
"""Series: concat vs assistant-only vs user-row. Geometry + lookup ranks.

Not a language judge. No HIT/MISS. No pass rate. Writes ranks and cosines.

T1 pollution: cos(user,concat) - cos(user,assistant)
T2 dense rank, query=user, pool=assistants
T3 dense rank, query=user, pool=concats
T4 dense rank, query=user, pool=users (current: user is its own row)
T5 dense rank, query=assistant, pool=assistants / concats
T6 lexical rank, query=user, docs=assistant / concat / user
T7 same-conversation vs other-conversation cosine (no hand topics)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HUMAN = {"human", "user"}
ASST = {"collaborator", "assistant"}
SKIP = ("[external_agent_tool", "tool exec", "TRANSCRIPT START", "declare const tools")
TOK = re.compile(r"[a-z0-9]+")


def l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def clip(text: str, n: int = 1200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n]


def mean(xs: list[float]) -> float | None:
    return None if not xs else sum(xs) / len(xs)


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[i]


def summarize(xs: list[float]) -> dict:
    return {
        "n": len(xs),
        "mean": mean(xs),
        "median": pct(xs, 50),
        "p90": pct(xs, 90),
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
    }


def ranks_from_scores(own: list[float], matrix: list[list[float]]) -> list[int]:
    """matrix[i][j] = score of query i vs doc j. Rank of the true pair (i,i). 1 = best."""
    out = []
    n = len(own)
    for i in range(n):
        s = own[i]
        r = 1
        for j in range(n):
            if j == i:
                continue
            if matrix[i][j] > s:
                r += 1
        out.append(r)
    return out


def embed_batch(url: str, texts: list[str], batch: int = 16) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        body = json.dumps(
            {"model": "qwen", "input": chunk, "encoding_format": "float"}
        ).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.loads(resp.read().decode())
        items = sorted(payload["data"], key=lambda row: row["index"])
        if len(items) != len(chunk):
            raise SystemExit(f"embed {len(items)} != {len(chunk)}")
        out.extend(l2(row["embedding"]) for row in items)
        print(f"  embed {min(i + batch, len(texts))}/{len(texts)}", file=sys.stderr)
    return out


def tokenize(text: str) -> list[str]:
    return TOK.findall(text.lower())


def lexical_ranks(queries: list[str], docs: list[str]) -> list[int]:
    doc_toks = [tokenize(d) for d in docs]
    df: dict[str, int] = defaultdict(int)
    for toks in doc_toks:
        for w in set(toks):
            df[w] += 1
    n = len(docs)
    idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
    ranks = []
    for i, q in enumerate(queries):
        qset = tokenize(q)
        scores = []
        for toks in doc_toks:
            tf: dict[str, int] = defaultdict(int)
            for w in toks:
                tf[w] += 1
            s = 0.0
            for w in qset:
                if w in tf:
                    s += (1.0 + math.log(tf[w])) * idf.get(w, 1.0)
            scores.append(s)
        own = scores[i]
        r = 1 + sum(1 for j, s in enumerate(scores) if j != i and s > own)
        ranks.append(r)
    return ranks


def load_pairs(jsonl: Path, seed: int, n_single: int, n_multi_convos: int, per_convo: int) -> list[dict]:
    by: dict[str, list] = defaultdict(list)
    with jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            cid = o.get("conversation_id")
            sp = o.get("speaker")
            ti = o.get("turn_index")
            t = (o.get("text") or "").strip()
            if not cid or ti is None or not t:
                continue
            if sp not in HUMAN and sp not in ASST:
                continue
            if any(s in t for s in SKIP):
                continue
            by[cid].append((int(ti), sp, t, o.get("id"), o.get("source")))

    multi: list[list[dict]] = []
    single: list[dict] = []
    for cid, rows in by.items():
        rows.sort(key=lambda r: r[0])
        found = []
        for i in range(len(rows) - 1):
            a, b = rows[i], rows[i + 1]
            if a[1] not in HUMAN or b[1] not in ASST:
                continue
            u, v = clip(a[2]), clip(b[2])
            if not (40 <= len(u) <= 1200 and 40 <= len(v) <= 2500):
                continue
            found.append(
                {
                    "conversation_id": cid,
                    "user_id": a[3],
                    "assistant_id": b[3],
                    "source": a[4],
                    "user": u,
                    "assistant": v,
                    "user_turn": a[0],
                    "assistant_turn": b[0],
                }
            )
        if len(found) >= 2:
            multi.append(found)
        elif len(found) == 1:
            single.append(found[0])

    rng = random.Random(seed)
    rng.shuffle(multi)
    rng.shuffle(single)
    out: list[dict] = []
    for bundle in multi[:n_multi_convos]:
        out.extend(bundle[:per_convo])
    need = n_single
    out.extend(single[:need])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/home/ruffianl/Hub/Projects/splatrag/memory/cold/memories.jsonl")
    ap.add_argument("--embed", default="http://127.0.0.1:8081/v1/embeddings")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--n-single", type=int, default=90)
    ap.add_argument("--n-multi-convos", type=int, default=80)
    ap.add_argument("--per-convo", type=int, default=2)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1]
            / "research_logs"
            / "2026-09-03_chunk-geometry-series.json"
        ),
    )
    args = ap.parse_args()

    print("loading pairs from cold log…", file=sys.stderr)
    pairs = load_pairs(
        Path(args.jsonl), args.seed, args.n_single, args.n_multi_convos, args.per_convo
    )
    n = len(pairs)
    print(f"sampled {n} pairs", file=sys.stderr)
    users = [p["user"] for p in pairs]
    assts = [p["assistant"] for p in pairs]
    concats = [f"User: {u}\nAssistant: {a}" for u, a in zip(users, assts)]

    print("embedding users + assistants + concats…", file=sys.stderr)
    blob = embed_batch(args.embed, users + assts + concats)
    u_vec, a_vec, c_vec = blob[:n], blob[n : 2 * n], blob[2 * n :]

    pollution = []
    rows = []
    for i, p in enumerate(pairs):
        cu = cos(u_vec[i], c_vec[i])
        ua = cos(u_vec[i], a_vec[i])
        ac = cos(a_vec[i], c_vec[i])
        pollution.append(cu - ua)
        rows.append(
            {
                "i": i,
                "conversation_id": p["conversation_id"],
                "user_id": p["user_id"],
                "assistant_id": p["assistant_id"],
                "source": p["source"],
                "user_chars": len(p["user"]),
                "assistant_chars": len(p["assistant"]),
                "cos_user_concat": cu,
                "cos_user_assistant": ua,
                "cos_assistant_concat": ac,
                "pollution_delta": cu - ua,
            }
        )

    def dense_matrix(qs, ds):
        return [[cos(qs[i], ds[j]) for j in range(n)] for i in range(n)]

    m_u_a = dense_matrix(u_vec, a_vec)
    m_u_c = dense_matrix(u_vec, c_vec)
    m_u_u = dense_matrix(u_vec, u_vec)
    m_a_a = dense_matrix(a_vec, a_vec)
    m_a_c = dense_matrix(a_vec, c_vec)
    own_ua = [m_u_a[i][i] for i in range(n)]
    own_uc = [m_u_c[i][i] for i in range(n)]
    own_uu = [m_u_u[i][i] for i in range(n)]
    own_aa = [m_a_a[i][i] for i in range(n)]
    own_ac = [m_a_c[i][i] for i in range(n)]

    t2 = ranks_from_scores(own_ua, m_u_a)
    t3 = ranks_from_scores(own_uc, m_u_c)
    t4 = ranks_from_scores(own_uu, m_u_u)
    t5a = ranks_from_scores(own_aa, m_a_a)
    t5c = ranks_from_scores(own_ac, m_a_c)

    lex_a = lexical_ranks(users, assts)
    lex_c = lexical_ranks(users, concats)
    lex_u = lexical_ranks(users, users)

    # T7: same conversation vs other, no hand labels
    by_cid: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs):
        by_cid[p["conversation_id"]].append(i)
    intra_c, inter_c, intra_a, inter_a = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            same = pairs[i]["conversation_id"] == pairs[j]["conversation_id"]
            cc = cos(c_vec[i], c_vec[j])
            aa = cos(a_vec[i], a_vec[j])
            if same:
                intra_c.append(cc)
                intra_a.append(aa)
            else:
                inter_c.append(cc)
                inter_a.append(aa)

    for i, r in enumerate(rows):
        r["rank_user_query_assistant_pool"] = t2[i]
        r["rank_user_query_concat_pool"] = t3[i]
        r["rank_user_query_user_pool"] = t4[i]
        r["rank_assistant_query_assistant_pool"] = t5a[i]
        r["rank_assistant_query_concat_pool"] = t5c[i]
        r["lex_rank_user_query_assistant_docs"] = lex_a[i]
        r["lex_rank_user_query_concat_docs"] = lex_c[i]
        r["lex_rank_user_query_user_docs"] = lex_u[i]

    report = {
        "protocol": "chunk-geometry-series-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embed_url": args.embed,
        "seed": args.seed,
        "n_pairs": n,
        "n_conversations": len(by_cid),
        "pool_size": n,
        "note": "Ranks are 1=best among n. No verdict. Lookup question is whether user text must be searchable.",
        "T1_pollution_delta": summarize(pollution),
        "T2_dense_user_query_assistant_pool": summarize([float(x) for x in t2]),
        "T3_dense_user_query_concat_pool": summarize([float(x) for x in t3]),
        "T4_dense_user_query_user_pool": summarize([float(x) for x in t4]),
        "T5_dense_assistant_query_assistant_pool": summarize([float(x) for x in t5a]),
        "T5_dense_assistant_query_concat_pool": summarize([float(x) for x in t5c]),
        "T6_lex_user_query_assistant_docs": summarize([float(x) for x in lex_a]),
        "T6_lex_user_query_concat_docs": summarize([float(x) for x in lex_c]),
        "T6_lex_user_query_user_docs": summarize([float(x) for x in lex_u]),
        "T7_conversation_geometry": {
            "concat_intra": summarize(intra_c),
            "concat_inter": summarize(inter_c),
            "concat_intra_minus_inter": None
            if mean(intra_c) is None or mean(inter_c) is None
            else mean(intra_c) - mean(inter_c),
            "assistant_intra": summarize(intra_a),
            "assistant_inter": summarize(inter_a),
            "assistant_intra_minus_inter": None
            if mean(intra_a) is None or mean(inter_a) is None
            else mean(intra_a) - mean(inter_a),
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in report if k != "rows"}
    print(json.dumps(summary, indent=2))
    print(f"wrote {out} ({n} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
