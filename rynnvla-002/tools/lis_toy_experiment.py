"""Controlled toy experiment for the LIS-VLA search mechanism (CPU, tiny backbone).

This is a *mechanism sanity check*, not a robotics result. Sequences use the exact
RynnVLA-002 token layout, but observations are synthetic:

* each task's instruction asks for one of two "goal markers" (A for tasks 0,1; B for tasks 2,3);
* the future frame equals the current frame with the first M tokens of every view replaced by
  marker A if the mean of action dimension 0 over the chunk is positive, otherwise marker B;
* expert chunks always move toward the instructed marker.

Only the LIS heads are trained (random frozen tiny backbone, ``lis_only`` setting). On held-out
prompts we measure (i) latent prediction quality vs. a copy-the-present baseline, (ii) whether
the world model is action-sensitive, and (iii) whether imagination search picks a chunk with the
correct effect from N *uniformly random* candidates (chance = 0.5 per candidate).

    cd rynnvla-002 && python tools/lis_toy_experiment.py --out ../paper/data
"""
import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ChameleonXLLMXForConditionalGeneration_LIS, LISVLAConfig  # noqa: E402
from model.lis import action_tokens_to_continuous, parse_sample  # noqa: E402

H, A, V, L, M = 5, 7, 2, 12, 6
MARK = {0: list(range(100, 100 + M)), 1: list(range(200, 200 + M))}


def config(**kw):
    c = dict(vocab_size=65536, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=4, max_position_embeddings=4096, vocabulary_map={"<image>": 8711},
             mask_image_logits=False, action_dim=A, time_horizon=H, lis_dim=64, lis_num_latents=4, lis_ctx_tokens=2,
             lis_fm_layers=2, lis_wm_layers=2, lis_fm_steps=8, lis_smooth_weight=0.0, lis_include_prior=False,
             lis_train_scope="lis_only")
    c.update(kw)
    return LISVLAConfig(**c)


def img(tokens):
    return [8197] + tokens + [8196]


def act_block(a):
    return [10004] + (torch.bucketize(a, torch.linspace(-1, 1, 256)) + 10005).tolist() + [15004]


def sample(task, g, expert=True):
    goal = 0 if task < 2 else 1
    instr = [0] + [16500 + task * 50 + i for i in range(8)]
    state = [15504] + torch.randint(15505, 15760, (8,), generator=g).tolist() + [16004]
    frames = [[torch.randint(300, 8195, (L,), generator=g).tolist() for _ in range(V)] for _ in range(2)]
    prompt = instr + state + sum([img(v) for f in frames for v in f], [])
    acts = torch.rand(H, A, generator=g) * 2 - 1
    mag = 0.2 + 0.8 * torch.rand(H, generator=g)
    acts[:, 0] = mag if goal == 0 else -mag
    if not expert:
        acts[:, 0] = -acts[:, 0]
    reached = 0 if acts[:, 0].mean() > 0 else 1
    future = [MARK[reached] + v[M:] for v in frames[-1]]
    gpt = sum([act_block(a) for a in acts], []) + sum([img(v) for v in future], [])
    return prompt, gpt, goal


def build(n, g):
    data = []
    for i in range(n):
        p, gpt, _ = sample(i % 4, g)
        data.append((p + gpt, [-100] * len(p) + gpt))
    return data


@torch.no_grad()
def latents(model, ids_list):
    """Returns per-sample (ctx, goal, z_t, z_future, head_feats, prior) for parsed samples."""
    cfg = model.config
    out = []
    for ids in ids_list:
        x = torch.tensor([ids])
        hs = model.model(input_ids=x, output_hidden_states=True).hidden_states[-1]
        s = parse_sample(x[0], cfg.lis_num_views, H, A)
        ctx, goal = model._lis_inputs(hs, x, [s], [0])
        emb = model._image_embeds(x)
        z_t = model.lis.obs_encoder([[emb[0, a:e] for a, e in s.cur_image_spans]])
        z_f = model.lis.obs_encoder([[emb[0, a:e] for a, e in s.future_image_spans]]) if s.has_future else None
        _, _, feats, _ = model.action_head(hs, x, None, 10004, eval=True, return_features=True)
        out.append((ctx, goal, z_t, z_f, feats))
    return out


def evaluate(model, g, n_eval, Ns):
    model.eval()
    res = {"cos_pred": [], "cos_copy": [], "wm_action_sensitive": [], "search": {n: [] for n in Ns},
           "proposer_correct": []}
    for i in range(n_eval):
        task = i % 4
        p, gpt, goal = sample(task, g)
        ids = p + gpt
        ctx, gl, z_t, z_f, feats = latents(model, [ids])[0]
        s = parse_sample(torch.tensor(ids), V, H, A)
        a_exp = action_tokens_to_continuous(s.action_tokens)[None]
        wm = model.lis.world_model
        with torch.no_grad():
            z_hat = wm(z_t, ctx, a_exp)
            a_flip = a_exp.clone()
            a_flip[..., 0] = -a_flip[..., 0]
            z_hat_flip = wm(z_t, ctx, a_flip)
        res["cos_pred"].append(F.cosine_similarity(z_hat.flatten(), z_f.flatten(), dim=0).item())
        res["cos_copy"].append(F.cosine_similarity(z_t.flatten(), z_f.flatten(), dim=0).item())
        c_exp = F.cosine_similarity(z_hat.flatten(), z_f.flatten(), dim=0).item()
        c_flip = F.cosine_similarity(z_hat_flip.flatten(), z_f.flatten(), dim=0).item()
        res["wm_action_sensitive"].append(0.5 if abs(c_exp - c_flip) < 1e-6 else float(c_exp > c_flip))  # tie = chance
        # imagination search over uniformly random candidates (no proposer, no prior, no cost)
        for n in Ns:
            cands = torch.rand(n, H, A) * 2 - 1
            with torch.no_grad():
                zz = wm(z_t.expand(n, -1, -1), ctx.expand(n, -1, -1), cands)
                r = model.lis.evaluator(zz, gl.expand(n, -1))
            best = cands[int(r.argmax())]
            reached = 0 if best[:, 0].mean() > 0 else 1
            res["search"][n].append(float(reached == goal))
        samples = model.lis.proposer.sample(feats, ctx, 16, model.config.lis_fm_steps)[0]
        res["proposer_correct"].append(float(((samples[:, :, 0].mean(1) > 0).float() == (1 - goal)).float().mean()))
    summary = {
        "cos_pred": sum(res["cos_pred"]) / n_eval,
        "cos_copy": sum(res["cos_copy"]) / n_eval,
        "wm_action_sensitive_acc": sum(res["wm_action_sensitive"]) / n_eval,
        "search_acc": {str(n): sum(v) / n_eval for n, v in res["search"].items()},
        "proposer_correct_frac": sum(res["proposer_correct"]) / n_eval,
    }
    model.train()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../paper/data")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--n_train", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Ns = [1, 2, 4, 8, 16, 32]
    all_summ = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = ChameleonXLLMXForConditionalGeneration_LIS(config())
        del model.model.vqmodel
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lis.parameters():
            p.requires_grad_(True)
        g_train, g_eval = torch.Generator().manual_seed(1000 + seed), torch.Generator().manual_seed(9000 + seed)
        data = build(args.n_train, g_train)
        before = evaluate(model, torch.Generator().manual_seed(9000 + seed), args.n_eval, Ns)
        opt = torch.optim.AdamW(model.lis.parameters(), lr=3e-4, weight_decay=0.01)
        log = []
        t0 = time.time()
        for step in range(args.steps):
            idx = torch.randint(0, len(data), (args.batch,))
            ids, labels = [data[i][0] for i in idx], [data[i][1] for i in idx]
            _, add, *_ = model(input_ids=ids, labels=labels, output_hidden_states=True, training=True)
            loss = sum(v * w for v, w in add.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            log.append({"step": step, **{k: float(v[0]) for k, v in add.items() if k.startswith("lis")}})
        after = evaluate(model, g_eval, args.n_eval, Ns)
        summ = {"seed": seed, "train_seconds": time.time() - t0, "before": before, "after": after}
        print(json.dumps(summ, indent=1))
        all_summ.append(summ)
        with open(os.path.join(args.out, f"toy_losses_seed{seed}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
    with open(os.path.join(args.out, "toy_summary.json"), "w") as f:
        json.dump(all_summ, f, indent=1)


if __name__ == "__main__":
    main()
