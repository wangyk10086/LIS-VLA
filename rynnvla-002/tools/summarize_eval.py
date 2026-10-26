"""Merge sharded eval results and print success rates with 95% Wilson intervals.

    python tools/summarize_eval.py /workspace/results/phase0 [--json out.json]
"""
import argparse
import collections
import glob
import json
import math
import os


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows, out = [], {}
    for job in sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)) and d != "logs"):
        recs = {}
        for f in glob.glob(os.path.join(args.root, job, "results_shard*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                recs[(r["task_index"], r["episode"])] = r  # de-duplicate across re-runs
        if not recs:
            continue
        by = collections.defaultdict(list)
        for r in recs.values():
            by[r["category"]].append(r)
            by["ALL"].append(r)
        for cat, rs in sorted(by.items()):
            if cat == "ALL" and len(by) == 2:
                continue
            k, n = sum(r["success"] for r in rs), len(rs)
            p, lo, hi = wilson(k, n)
            errs = sum(1 for r in rs if r.get("error"))
            lat = [r["query_latency_mean"] for r in rs if r.get("query_latency_mean")]
            ov = sum(r.get("overrides", 0) for r in rs) / max(1, sum(r.get("queries", 0) for r in rs))
            row = dict(job=job, category=cat, n=n, success=k, rate=round(100 * p, 1), ci95=f"{100*lo:.1f}-{100*hi:.1f}",
                       errors=errs, query_s=round(sum(lat) / len(lat), 3) if lat else None,
                       override_rate=round(100 * ov, 1), ep_s=round(sum(r["seconds"] for r in rs) / n, 1))
            rows.append(row)
            out.setdefault(job, {})[cat] = row
    hdr = ["job", "category", "n", "rate", "ci95", "errors", "query_s", "override_rate", "ep_s"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(str(r[h]) for h in hdr) + " |")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
