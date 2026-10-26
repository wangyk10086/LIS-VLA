"""Paired comparison of two evaluation runs on the episodes they share.

    python tools/compare_eval.py <baseline_dir> <new_dir> [--by category]

Episodes are matched on (task_index, episode). Reports both success rates on the shared set, the
difference with a 95% paired-bootstrap interval, and an exact McNemar test on the discordant pairs.
"""
import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "results_shard*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out[(r["task_index"], r["episode"])] = r
    return out


def mcnemar_exact(b, c):
    """two-sided exact binomial test on the discordant pairs (b: only new wins, c: only base wins)"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(pairs, rng):
    n = len(pairs)
    base = sum(a for a, _ in pairs) / n
    new = sum(b for _, b in pairs) / n
    diffs = [b - a for a, b in pairs]
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(2000))
    lo, hi = boots[49], boots[1949]
    b = sum(1 for a, x in pairs if x and not a)
    c = sum(1 for a, x in pairs if a and not x)
    return n, 100 * base, 100 * new, 100 * (new - base), 100 * lo, 100 * hi, b, c, mcnemar_exact(b, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--by", default="category")
    args = ap.parse_args()
    A, B = load(args.baseline), load(args.new)
    keys = sorted(set(A) & set(B))
    if not keys:
        raise SystemExit("no shared episodes")
    errs = sum(1 for k in keys if B[k].get("error"))
    groups = defaultdict(list)
    for k in keys:
        pair = (int(A[k]["success"]), int(B[k]["success"]))
        groups["ALL"].append(pair)
        groups[A[k].get(args.by, "?")].append(pair)
    rng = random.Random(0)
    print(f"shared episodes: {len(keys)} (baseline has {len(A)}, new has {len(B)}; errors in new: {errs})")
    print("| group | n | baseline | new | diff (pts) | 95% CI | new-only wins | base-only wins | McNemar p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g, pairs in sorted(groups.items(), key=lambda kv: kv[0] != "ALL"):
        n, a, b, d, lo, hi, w, l, p = compare(pairs, rng)
        print(f"| {g} | {n} | {a:.1f} | {b:.1f} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {w} | {l} | {p:.3g} |")
    lat = [B[k]["query_latency_mean"] for k in keys if B[k].get("query_latency_mean")]
    ov = [B[k]["overrides"] / max(B[k]["queries"], 1) for k in keys if B[k].get("queries")]
    if lat:
        print(f"new: mean query latency {sum(lat) / len(lat):.3f} s, System-2 override rate {100 * sum(ov) / len(ov):.1f}%")


if __name__ == "__main__":
    main()
