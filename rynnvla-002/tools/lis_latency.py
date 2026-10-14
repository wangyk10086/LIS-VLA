"""Latency of the System-2 search at 7B-backbone dimensions (heads only; the backbone pass is
shared with System 1 and excluded).

    cd rynnvla-002 && python tools/lis_latency.py --out ../paper/data/latency.json
"""
import argparse
import json
import os
import platform
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lis import LISHeads  # noqa: E402
from model.modeling_lis_vla import LIS_DEFAULTS, reset_lis_parameters  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../paper/data/latency.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg = types.SimpleNamespace(**LIS_DEFAULTS, action_dim=7, time_horizon=5)
    lis = LISHeads(cfg, 4096, 1024).eval()
    reset_lis_parameters(lis)
    n_params = sum(p.numel() for p in lis.parameters())
    ctx, g = torch.randn(1, 8, 512), torch.randn(1, 512)
    feats, z = torch.randn(1, 35, 1024), torch.randn(1, 16, 512)
    prior = torch.rand(5, 7) * 2 - 1
    rows = []
    for n, cem in [(1, 0), (4, 0), (8, 0), (16, 0), (32, 0), (64, 0), (16, 2)]:
        lis.search(ctx, g, feats, z, prior_actions=prior, num_candidates=n, cem_iters=cem)  # warm-up
        times = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            lis.search(ctx, g, feats, z, prior_actions=prior, num_candidates=n, cem_iters=cem)
            times.append(time.perf_counter() - t)
        times.sort()
        rows.append({"N": n, "cem": cem, "median_ms": 1000 * times[len(times) // 2]})
        print(rows[-1])
    out = {"device": f"CPU ({platform.processor() or platform.machine()}), {args.threads} threads, fp32",
           "torch": torch.__version__, "params_M": n_params / 1e6, "rows": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
