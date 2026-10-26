"""Head-room by intervention horizon, for tools/branch_horizons.py output.

    python tools/branch_horizons_summary.py DIR [--json out.json]

For each horizon H: share of failure branch points where >=1 of the alternative branches succeeds
while the System-1 replay fails ("rescuable"), share of failed episodes with a rescuable point, the
per-branch success rate, and the resulting oracle episode success if the base failure rate is known
(--base_success k/n: System-1 successes and episodes of the full branching run).
"""
import argparse
import glob
import json
import os

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k}/{n} = {p:.1f}% [{lo:.1f}, {hi:.1f}]"


def summarise(d, base_success=None):
    eps = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "ep*.npz")))]
    logs = [json.loads(l) for f in glob.glob(os.path.join(d, "log_shard*.jsonl")) for l in open(f) if l.strip()]
    fail = [e for e in eps if not bool(e["main_success"]) and e["alt_success"].shape[0] > 0]
    rep = {"episodes_run": len(eps), "errors": sum(1 for r in logs if r.get("error")),
           "main_reproduced_failure": fmt(len([e for e in eps if not bool(e["main_success"])]), len(eps)),
           "failed_episodes_with_points": len(fail), "by_horizon": {}}
    if not fail:
        return rep
    horizons = [int(h) for h in fail[0]["horizons"]]
    s1_fail = [(e, i) for e in fail for i in range(len(e["s1_success"])) if not e["s1_success"][i]]
    rep["system1_replay_fails"] = fmt(len(s1_fail), sum(len(e["s1_success"]) for e in fail))
    for j, h in enumerate(horizons):
        pts = [bool(e["alt_success"][i, j].any()) for e, i in s1_fail]
        per_branch = [bool(x) for e, i in s1_fail for x in e["alt_success"][i, j]]
        ep_rescued = sum(1 for e in fail if any(e["alt_success"][i, j].any() and not e["s1_success"][i]
                                                 for i in range(len(e["s1_success"]))))
        row = {"rescuable_points": fmt(sum(pts), len(pts)),
               "rescuable_failed_episodes": fmt(ep_rescued, len(fail)),
               "single_branch_success": fmt(sum(per_branch), len(per_branch)),
               "mean_alt_decisions": round(float(np.mean([e["n_alt"][i, j].mean() for e, i in s1_fail])), 1)}
        if base_success:
            k, n = base_success
            # failed episodes here are a sample of the n-k failures of the full run: scale the rescue share
            gain = (n - k) * ep_rescued / len(fail)
            row["oracle_episode_success_est"] = f"{100 * (k + gain) / n:.1f}% (System 1: {100 * k / n:.1f}%)"
        rep["by_horizon"]["all" if h < 0 else str(h)] = row
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--base_success", default=None, help="k/n System-1 successes in the full branching run")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    base = tuple(int(x) for x in a.base_success.split("/")) if a.base_success else None
    rep = summarise(a.dir, base)
    print(json.dumps(rep, indent=1))
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
