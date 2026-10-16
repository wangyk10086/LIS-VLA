"""Head-room summary for tools/branch_rollouts.py output.

    python tools/branch_summary.py /workspace/data/branches_goal [--json out.json]
"""
import argparse
import glob
import json
import os

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    logs = [json.loads(l) for f in glob.glob(os.path.join(args.dir, "log_shard*.jsonl")) for l in open(f) if l.strip()]
    errs = [r for r in logs if r.get("error")]
    eps = [np.load(f) for f in sorted(glob.glob(os.path.join(args.dir, "ep*.npz")))]
    fail = [e for e in eps if not bool(e["main_success"])]
    succ = [e for e in eps if bool(e["main_success"])]

    s1_repro = [bool(b[0]) == bool(e["main_success"]) for e in eps for b in e["branch_success"]]
    # failed episodes: can an alternative chunk at a branch point lead to success?
    f_pts = [b for e in fail for b in e["branch_success"]]
    f_pts_s1_failed = [b for b in f_pts if not b[0]]          # S1 branch reproduced the failure
    rescue_pts = sum(1 for b in f_pts_s1_failed if any(b[1:]))
    rescue_eps = sum(1 for e in fail if any(any(b[1:]) and not b[0] for b in e["branch_success"]))
    alt_rate_fail = np.mean([x for b in f_pts_s1_failed for x in b[1:]]) if f_pts_s1_failed else float("nan")
    # successful episodes: how often does a random alternative break it?
    s_pts = [b for e in succ for b in e["branch_success"] if b[0]]
    alt_rate_succ = np.mean([x for b in s_pts for x in b[1:]]) if s_pts else float("nan")
    # by position of the branch point inside failed episodes (early / middle / late)
    by_pos = {}
    for e in fail:
        n = len(e["branch_success"])
        for i, b in enumerate(e["branch_success"]):
            if b[0]:
                continue
            pos = "early" if i == 0 else ("late" if i == n - 1 and n > 1 else "middle")
            by_pos.setdefault(pos, []).append(any(b[1:]))

    def fmt(k, n):
        p, lo, hi = wilson(k, n)
        return f"{k}/{n} = {p:.1f}% [{lo:.1f}, {hi:.1f}]"

    rep = {
        "episodes": len(eps), "errors": len(errs),
        "system1_success": fmt(len(succ), len(eps)),
        "system1_branch_reproduces_main": fmt(sum(s1_repro), len(s1_repro)),
        "failed_eps_branch_points": len(f_pts_s1_failed),
        "rescuable_points (>=1 of the alternatives succeeds)": fmt(rescue_pts, len(f_pts_s1_failed)),
        "rescuable_failed_episodes (any point)": fmt(rescue_eps, len(fail)),
        "random_alternative_success_rate_in_failed_states": round(100 * float(alt_rate_fail), 1),
        "random_alternative_success_rate_in_successful_states": round(100 * float(alt_rate_succ), 1),
        "rescuable_by_position": {k: fmt(sum(v), len(v)) for k, v in sorted(by_pos.items())},
        "oracle_episode_success_upper_bound": fmt(len(succ) + rescue_eps, len(eps)),
    }
    for k, v in rep.items():
        print(f"{k}: {v}")
    if errs:
        print("first error:", errs[0]["error"])
    if args.json:
        json.dump(rep, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
