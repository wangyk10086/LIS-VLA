"""Sanity-check pre-tokenised LIS samples BEFORE spending GPU hours on training.

Loads records produced by pretoken_lis_model.py + concate_record, parses them with the exact
parser used in training, and fails loudly if the layout is not what LIS-VLA expects.
Optionally writes a small subset (for a smoke-test training run).

    python tools/check_lis_data.py --record ../processed_data/concate_tokens/<name>.json \
        --horizon 5 --views 2 [--subset_out ../configs/lis/smoke.yaml --subset_n 64]
"""
import argparse
import collections
import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lis.tokens import ACT_START, IMG_END, IMG_START, parse_sample  # noqa: E402


def load_tokens(item):
    if "token" in item:
        return item["token"], item["label"]
    with open(item["file"], "rb") as f:
        d = pickle.load(f)
    return d["token"], d["label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="concatenated record json used by the data config")
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--action_dim", type=int, default=7)
    ap.add_argument("--n", type=int, default=500, help="records to inspect")
    ap.add_argument("--subset_out", default=None, help="write a yaml data config over a small subset")
    ap.add_argument("--subset_n", type=int, default=64)
    args = ap.parse_args()

    records = json.load(open(args.record))
    print(f"{len(records)} records in {args.record}")
    stats = collections.Counter()
    lens, img_lens, problems = [], collections.Counter(), []
    lis_items = []
    for i, item in enumerate(records[: args.n]):
        try:
            tok, lab = load_tokens(item)
        except Exception as e:  # noqa: BLE001
            problems.append(f"record {i}: cannot load ({e})")
            continue
        ids = torch.tensor(tok)
        lens.append(len(tok))
        s = parse_sample(ids, args.views, args.horizon, args.action_dim)
        if s is None:
            stats["no_action_block (world-model sample?)"] += 1
            continue
        if not s.has_chunk:
            stats["action blocks but no full chunk"] += 1
            continue
        stats["full chunk"] += 1
        if s.has_future:
            stats["full chunk + future frames (LIS)"] += 1
            lis_items.append(item)
            cur = [e - a for a, e in s.cur_image_spans]
            fut = [e - a for a, e in s.future_image_spans]
            img_lens[tuple(cur + fut)] += 1
            if cur != fut:
                problems.append(f"record {i}: current/future view lengths differ {cur} vs {fut}")
            # labels must supervise actions and future image, not the prompt
            lab_t = torch.tensor(lab)
            if (lab_t[: s.ctx_end] != -100).any():
                problems.append(f"record {i}: prompt tokens are supervised")
            if lab_t[s.ctx_end] == -100:
                problems.append(f"record {i}: first action block is not supervised")
            fut_start = s.future_image_spans[0][0]
            if lab_t[fut_start] == -100:
                problems.append(f"record {i}: future frame is not supervised")
        a = s.action_tokens
        if (a <= ACT_START).any() or (a > ACT_START + 257).any():
            problems.append(f"record {i}: action token ids out of range")
        if (ids == IMG_START).sum() != (ids == IMG_END).sum():
            problems.append(f"record {i}: unbalanced image markers")

    print("sample types:", dict(stats))
    if lens:
        print(f"sequence length: min {min(lens)}, mean {sum(lens) / len(lens):.0f}, max {max(lens)}")
    print("image-block lengths (current views + future views):", dict(img_lens.most_common(3)))
    if problems:
        print(f"\n{len(problems)} PROBLEMS, first 10:")
        for p in problems[:10]:
            print("  -", p)
    if stats["full chunk + future frames (LIS)"] == 0:
        print("\nFAIL: no LIS samples found - System-2 world model/evaluator would never be trained.")
        sys.exit(1)
    if problems:
        sys.exit(1)
    print("\nOK: data layout matches what LIS-VLA expects.")

    if args.subset_out:
        sub = lis_items[: args.subset_n]
        sub_json = os.path.splitext(args.subset_out)[0] + "_records.json"
        json.dump(sub, open(sub_json, "w"))
        with open(args.subset_out, "w") as f:
            f.write(f"META:\n  - path: '{os.path.abspath(sub_json)}'\n")
        print(f"wrote smoke-test config with {len(sub)} LIS samples -> {args.subset_out}")


if __name__ == "__main__":
    main()
