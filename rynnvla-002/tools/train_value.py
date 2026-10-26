"""Train an outcome-trained System-2 scorer on cached rollout features (tools/collect_rollouts.py).

Target for the chunk executed at query time t of an episode that ends at step T:
    y = success * gamma ** ((T - t) / H)        (failures: 0)
so the scorer prefers chunks that lead to success, and sooner. Only the small scorer is trained;
the backbone and the LIS encoders / world model stay frozen (the world model is used to imagine
ẑ_{t+H} for the ``imag`` scorer).

Reports on held-out episodes: AUC for final success, and - on queries where an alternative chunk
was explored - whether the scorer ranks the executed chunk consistently with the outcome.

    python tools/train_value.py --rollouts /workspace/data/rollouts_goal --mode imag \\
        --ckpt ../ckpts/RynnVLA-002/VLA_model_256/libero_goal --lis_weights <dir> --out heads/value_imag.pt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_rollouts(path, gamma, H):
    eps = []
    for f in sorted(glob.glob(os.path.join(path, "ep*.npz"))):
        d = np.load(f)
        succ = bool(d["success"])
        cand, ex = d["candidates"].astype(np.float32), d["executed"]
        chunk = cand[np.arange(len(ex)), ex]
        y = (succ * gamma ** ((int(d["end_step"]) - d["t"]) / H)).astype(np.float32)
        eps.append(dict(z=d["z_t"], ctx=d["ctx"], goal=d["goal"], chunk=chunk, cand=cand, executed=ex, y=y,
                        success=succ, task=int(d["task_index"])))
    return eps


def stack(eps, key):
    return np.concatenate([e[key] for e in eps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--mode", choices=["imag", "direct"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lis_weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--time_horizon", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from model.lis import LISHeads
    from model.lis.value import ChunkValue
    from model.modeling_lis_vla import LISVLAConfig, read_lis_state_dict

    eps = load_rollouts(args.rollouts, args.gamma, args.time_horizon)
    if len(eps) < 20:
        raise SystemExit(f"only {len(eps)} episodes in {args.rollouts}")
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(eps))
    n_val = max(4, int(len(eps) * args.val_frac))
    val, tr = [eps[i] for i in idx[:n_val]], [eps[i] for i in idx[n_val:]]
    print(f"episodes: {len(tr)} train / {len(val)} val, success rate {np.mean([e['success'] for e in eps]):.3f}, "
          f"queries {sum(len(e['y']) for e in eps)}", flush=True)

    dev = torch.device(args.device)
    wm = None
    if args.mode == "imag":
        cfg = LISVLAConfig.from_pretrained(args.ckpt, action_dim=7, time_horizon=args.time_horizon)
        heads = LISHeads(cfg, cfg.hidden_size, int(cfg.hidden_size * 0.25))
        sd = {k[len("lis."):]: v for k, v in read_lis_state_dict(args.lis_weights).items()}
        missing, unexpected = heads.load_state_dict(sd, strict=False)
        assert not [m for m in missing if m.startswith("world_model")], missing
        wm = heads.world_model.to(dev).float().eval()

    def tensors(es, chunk_key="chunk"):
        return (torch.from_numpy(stack(es, "z")).float(), torch.from_numpy(stack(es, "ctx")).float(),
                torch.from_numpy(stack(es, "goal")).float(), torch.from_numpy(stack(es, chunk_key)).float(),
                torch.from_numpy(stack(es, "y")).float())

    @torch.no_grad()
    def imagine(z, ctx, a):
        out = []
        for i in range(0, len(z), 1024):
            out.append(wm(z[i:i + 1024].to(dev), ctx[i:i + 1024].to(dev), a[i:i + 1024].to(dev)).cpu())
        return torch.cat(out)

    Ztr, Ctr, Gtr, Atr, Ytr = tensors(tr)
    Zva, Cva, Gva, Ava, Yva = tensors(val)
    if wm is not None:
        Ztr_in, Zva_in = imagine(Ztr, Ctr, Atr), imagine(Zva, Cva, Ava)
    else:
        Ztr_in, Zva_in = Ztr, Zva

    model = ChunkValue(args.mode, d=Ztr.shape[-1], num_latents=Ztr.shape[1], num_ctx=Ctr.shape[1],
                       horizon=Atr.shape[1], action_dim=Atr.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def fwd(z, c, g, a):
        return model(g.to(dev), z.to(dev), c.to(dev), a.to(dev)) if args.mode == "direct" else model(g.to(dev), z.to(dev))

    @torch.no_grad()
    def predict(z, c, g, a):
        model.eval()
        out = [fwd(z[i:i + 2048], c[i:i + 2048], g[i:i + 2048], a[i:i + 2048]).cpu() for i in range(0, len(z), 2048)]
        return torch.cat(out)

    best, best_state = -1.0, None
    succ_val = np.concatenate([[e["success"]] * len(e["y"]) for e in val])
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Ytr))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss = F.binary_cross_entropy_with_logits(fwd(Ztr_in[b], Ctr[b], Gtr[b], Atr[b]), Ytr[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        pv = predict(Zva_in, Cva, Gva, Ava)
        vloss = float(F.binary_cross_entropy_with_logits(pv, Yva))
        a = auc(pv.numpy(), succ_val)
        print(f"epoch {ep:2d} train {tot / len(Ytr):.4f} val {vloss:.4f} auc {a:.3f}", flush=True)
        if a > best:
            best, best_state = a, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # ranking check on held-out queries: score all candidates; for explored queries, does the scorer
    # prefer the executed alternative over System 1 when the episode succeeded (and vice versa)?
    agree, n_rank, s1_rank = 0, 0, []
    for e in val:
        for q in range(len(e["y"])):
            cand = torch.from_numpy(e["cand"][q]).float()
            n = cand.shape[0]
            z = torch.from_numpy(e["z"][q]).float()[None].expand(n, -1, -1)
            c = torch.from_numpy(e["ctx"][q]).float()[None].expand(n, -1, -1)
            g = torch.from_numpy(e["goal"][q]).float()[None].expand(n, -1)
            zin = imagine(z, c, cand) if wm is not None else z
            s = predict(zin, c, g, cand).numpy()
            s1_rank.append(int((s > s[0]).sum()))
            k = int(e["executed"][q])
            if k:
                n_rank += 1
                agree += int((s[k] > s[0]) == e["success"])
    report = {"mode": args.mode, "episodes_train": len(tr), "episodes_val": len(val), "val_auc_success": round(best, 3),
              "explored_val_queries": n_rank, "ranking_agrees_with_outcome": round(agree / max(n_rank, 1), 3),
              "system1_rank_mean": round(float(np.mean(s1_rank)), 2)}
    print(json.dumps(report, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out, extra=report)
    json.dump(report, open(args.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
