"""Offline check of the System-2 scorer on held-out demonstrations (no simulator needed).

For each tokenised LIS sample (prompt + expert action chunk + future frames) it scores, with the
trained world model + evaluator, the expert chunk A*, the System-1 chunk, perturbed copies of A*
and proposer samples, and reports whether the score carries any signal:

  * pairwise accuracy  P(score(A*) > score(A* + noise))           (0.5 = no signal)
  * Spearman correlation between score and -||A - A*|| over proposer samples
  * how often the argmax over {System 1, proposer samples} is closer to A* than System 1 is
  * the score gap (in SDs) of the best candidate over System 1, to calibrate --override_z

    cd rynnvla-002/exps_lis   # token records use paths relative to this directory
    python ../tools/diag_evaluator.py --ckpt ../ckpts/RynnVLA-002/VLA_model_256/libero_goal \\
        --lis_weights ../outputs/.../epoch0-iter4999 --record ../processed_data/tokens/<val_dir>/record.json
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lis_weights", required=True)
    ap.add_argument("--record", required=True, help="record.json of a tokenised LIS split")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--time_horizon", type=int, default=5)
    ap.add_argument("--num_candidates", type=int, default=16)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from model import load_lis_model
    from model.chameleon import ChameleonForConditionalGeneration
    from model.lis import action_tokens_to_continuous, parse_sample
    from model.lis.tokens import ACT_START

    model = load_lis_model(args.ckpt, lis_weights=args.lis_weights, action_dim=7, time_horizon=args.time_horizon,
                           max_position_embeddings=8192, mask_image_logits=False, torch_dtype=torch.bfloat16,
                           attn_implementation="sdpa").cuda().eval()
    cfg = model.config
    recs = json.load(open(args.record))
    random.Random(0).shuffle(recs)

    acc = {s: [] for s in args.noise}
    rho, s1_err, best_err, gaps, s1_rank = [], [], [], [], []
    used = 0
    for rec in recs:
        if used >= args.n:
            break
        item = pickle.load(open(rec["file"], "rb"))  # paths are "../processed_data/...": run from exps_lis/
        ids = torch.tensor(item["token"], dtype=torch.long, device="cuda").unsqueeze(0)
        sample = parse_sample(ids[0], cfg.lis_num_views, args.time_horizon, 7)
        if sample is None or not sample.has_chunk:
            continue
        att = model.generate_att_mask_3(ids)
        out = ChameleonForConditionalGeneration.forward(model, input_ids=ids, attention_mask=att,
                                                        output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = (out.hidden_states if hasattr(out, "hidden_states") else out[1])[-1]
        prior, flag, feats, _ = model.action_head(hidden, ids, None, ACT_START, return_features=True)
        if not flag:
            continue
        prior = prior.reshape(args.time_horizon, 7).float()
        ctx, goal = model._lis_inputs(hidden, ids, [sample], [0])
        emb = model._image_embeds(ids)
        z_t = model.lis.obs_encoder([[emb[0, a:e] for a, e in sample.cur_image_spans]])
        gt = action_tokens_to_continuous(sample.action_tokens).float()

        def score(chunks):
            return model.lis.score_chunks(ctx, goal, z_t, chunks.to(ctx.dtype), prior)[0].float()

        s_gt = float(score(gt[None])[0])
        for s in args.noise:
            noisy = (gt[None] + s * torch.randn(8, *gt.shape, device=gt.device)).clamp(-1, 1)
            acc[s].append(float((score(noisy) < s_gt).float().mean()))

        cands = model.lis.proposer.sample(feats, ctx, args.num_candidates, cfg.lis_fm_steps, cfg.lis_noise_scale)[0].float()
        cands[0] = prior
        sc = score(cands).cpu().numpy()
        err = ((cands - gt[None]) ** 2).mean(dim=(1, 2)).cpu().numpy()
        rho.append(spearman(sc[1:], -err[1:]))
        b = int(sc.argmax())
        s1_err.append(float(err[0]))
        best_err.append(float(err[b]))
        gaps.append(float((sc[b] - sc[0]) / (sc.std() + 1e-6)))
        s1_rank.append(int((sc > sc[0]).sum()))
        used += 1

    s1_err, best_err = np.array(s1_err), np.array(best_err)
    report = {
        "samples": used,
        "pairwise_acc_gt_vs_noise": {str(s): round(float(np.mean(v)), 3) for s, v in acc.items()},
        "spearman_score_vs_closeness": round(float(np.nanmean(rho)), 3),
        "argmax_closer_than_system1": round(float(np.mean(best_err < s1_err - 1e-9)), 3),
        "argmax_farther_than_system1": round(float(np.mean(best_err > s1_err + 1e-9)), 3),
        "mse_system1_vs_expert": round(float(s1_err.mean()), 5),
        "mse_argmax_vs_expert": round(float(best_err.mean()), 5),
        "system1_rank_mean (0=best of N)": round(float(np.mean(s1_rank)), 2),
        "gap_z_percentiles(50,75,90,95)": [round(float(np.percentile(gaps, q)), 2) for q in (50, 75, 90, 95)],
    }
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
