"""Neural components of LIS-VLA (Latent Imagination Search VLA).

    z_t   = E_obs(I_t)                         latent tokens of the current observation
    c_t   = E_ctx(h_{<a})                      context tokens pooled from the VLA backbone
    g     = E_lang(h_instr)                    language-goal embedding
    A^i   ~ Proposer(. | h_act, c_t)           N candidate chunks (flow matching)
    z^i   = WM(z_t, c_t, A^i)                  imagined latent future  \\hat z_{t+k}
    R^i   = f(z^i, g) - Cost(A^i)              language-guided score
    A*    = argmax_i R^i
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- utils
def _encoder(dim: int, layers: int, heads: int, dropout: float = 0.0) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=4 * dim, dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)


def pad_sequences(seqs, pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """List of (L_i, D) -> (B, L_max, D) plus key-padding mask (True = pad)."""
    max_len = max(s.shape[0] for s in seqs)
    out = seqs[0].new_full((len(seqs), max_len, seqs[0].shape[-1]), pad_value)
    mask = torch.ones(len(seqs), max_len, dtype=torch.bool, device=seqs[0].device)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = False
    return out, mask


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AttentionPool(nn.Module):
    """Perceiver-style pooling: Q learnable queries cross-attend to a variable-length sequence."""

    def __init__(self, dim: int, num_queries: int, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        kv = self.norm_kv(x)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        out = out + q
        return out + self.ff(self.norm_ff(out))


# --------------------------------------------------------------------------- encoders
class ObservationEncoder(nn.Module):
    """Maps the VQ-token embeddings of the V current views to Q latent tokens z_t.

    Operates on the backbone's *input* embeddings of image tokens, so z_t is a pure
    function of the observation (no action leakage) and the same encoder can produce
    the stop-gradient target z_{t+k} from the real future frame.
    """

    def __init__(self, hidden_size: int, dim: int, num_latents: int, num_views: int, max_tokens_per_view: int = 1200):
        super().__init__()
        self.in_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.pos = nn.Embedding(max_tokens_per_view, dim)
        self.view = nn.Embedding(num_views, dim)
        self.pool = AttentionPool(dim, num_latents)
        self.refine = _encoder(dim, 1, 8)
        self.max_tokens_per_view = max_tokens_per_view
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.view.weight, std=0.02)

    def forward(self, view_embeds) -> torch.Tensor:
        """view_embeds: list (batch) of list (views) of (L_v, hidden) tensors."""
        seqs = []
        for views in view_embeds:
            parts = []
            for v, e in enumerate(views):
                e = e[: self.max_tokens_per_view]
                idx = torch.arange(e.shape[0], device=e.device)
                parts.append(self.in_proj(e) + self.pos(idx) + self.view.weight[v])
            seqs.append(torch.cat(parts, dim=0))
        x, mask = pad_sequences(seqs)
        return self.refine(self.pool(x, mask))


class SequencePooler(nn.Module):
    """Pools backbone hidden states (context or instruction) into K tokens of width ``dim``."""

    def __init__(self, hidden_size: int, dim: int, num_queries: int):
        super().__init__()
        self.in_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, dim))
        self.pool = AttentionPool(dim, num_queries)

    def forward(self, seqs) -> torch.Tensor:
        x, mask = pad_sequences([self.in_proj(s) for s in seqs])
        return self.pool(x, mask)


# --------------------------------------------------------------------------- proposer
class FlowMatchingProposer(nn.Module):
    """Rectified-flow action-chunk sampler conditioned on the action-head features and c_t.

    Training : x_tau = (1-tau) x_0 + tau a,   v* = a - x_0,   L = ||v(x_tau, tau) - v*||^2
    Sampling : Euler integration from x_0 ~ N(0, s^2 I) in ``steps`` steps, N chains in parallel.
    """

    def __init__(self, head_dim: int, action_dim: int, horizon: int, dim: int, layers: int = 3):
        super().__init__()
        self.action_dim, self.horizon = action_dim, horizon
        self.cond_proj = nn.Sequential(nn.LayerNorm(head_dim * action_dim), nn.Linear(head_dim * action_dim, dim))
        self.act_in = nn.Linear(action_dim, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.randn(horizon, dim) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.net = _encoder(dim, layers, 8)
        self.out = nn.Linear(dim, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.dim = dim

    def velocity(self, x: torch.Tensor, tau: torch.Tensor, cond: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x (B,H,A)  tau (B,)  cond (B,H,dim)  ctx (B,K,dim)
        t_emb = self.time_mlp(sinusoidal_embedding(tau, self.dim).to(x.dtype))[:, None]
        a_tok = self.act_in(x) + cond + self.pos.to(x.dtype) + t_emb + self.type_emb[0]
        c_tok = ctx + self.type_emb[1]
        h = self.net(torch.cat([c_tok, a_tok], dim=1))
        return self.out(h[:, ctx.shape[1]:])

    def encode_cond(self, head_feats: torch.Tensor) -> torch.Tensor:
        # head_feats: (B, H*A, head_dim) action-query outputs of the RynnVLA action head
        B = head_feats.shape[0]
        return self.cond_proj(head_feats.reshape(B, self.horizon, -1))

    def loss(self, head_feats: torch.Tensor, ctx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cond = self.encode_cond(head_feats)
        B = actions.shape[0]
        # logit-normal tau puts more weight on the hard middle of the path (as in SD3 / pi0)
        tau = torch.sigmoid(torch.randn(B, device=actions.device))
        x0 = torch.randn_like(actions)
        x_tau = (1 - tau[:, None, None]) * x0 + tau[:, None, None] * actions
        v = self.velocity(x_tau.to(cond.dtype), tau, cond, ctx)
        return F.mse_loss(v.float(), (actions - x0).float())

    @torch.no_grad()
    def sample(self, head_feats, ctx, num_samples: int, steps: int = 10, noise_scale: float = 1.0) -> torch.Tensor:
        """Returns (B, N, H, A) samples clipped to the normalised action range [-1, 1]."""
        cond = self.encode_cond(head_feats)
        B = cond.shape[0]
        cond = cond.repeat_interleave(num_samples, 0)
        ctx = ctx.repeat_interleave(num_samples, 0)
        x = torch.randn(B * num_samples, self.horizon, self.action_dim, device=cond.device, dtype=cond.dtype) * noise_scale
        dt = 1.0 / steps
        for k in range(steps):
            tau = torch.full((x.shape[0],), k * dt, device=x.device)
            x = x + dt * self.velocity(x, tau, cond, ctx)
        return x.clamp(-1, 1).reshape(B, num_samples, self.horizon, self.action_dim)


# --------------------------------------------------------------------------- world model
class LatentWorldModel(nn.Module):
    """Action-conditioned dynamics in latent space:  \\hat z_{t+k} = z_t + D([z_t; c_t; A]).

    The residual output projection is zero-initialised, so an untrained model is the
    identity map (a safe "nothing happens" prior) instead of random noise.
    """

    def __init__(self, dim: int, action_dim: int, horizon: int, layers: int = 4):
        super().__init__()
        self.act_in = nn.Linear(action_dim, dim)
        self.act_pos = nn.Parameter(torch.randn(horizon, dim) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.net = _encoder(dim, layers, 8)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, ctx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # z (B,Q,d)  ctx (B,K,d)  actions (B,H,A)
        a_tok = self.act_in(actions.to(z.dtype)) + self.act_pos[: actions.shape[1]].to(z.dtype) + self.type_emb[2]
        h = self.net(torch.cat([z + self.type_emb[0], ctx + self.type_emb[1], a_tok], dim=1))
        return z + self.out(h[:, : z.shape[1]])


# --------------------------------------------------------------------------- evaluator
class LanguageGuidedEvaluator(nn.Module):
    """R = f(\\hat z_{t+k}, g) = cos(phi(\\hat z), psi(g)) — semantic agreement of an imagined
    future with the instruction. ``logit_scale`` is only used by the training losses."""

    def __init__(self, dim: int, embed_dim: int = 256):
        super().__init__()
        self.state_pool = AttentionPool(dim, 1)
        self.phi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, embed_dim))
        self.psi = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def embed_state(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.phi(self.state_pool(z)[:, 0]).float(), dim=-1)

    def embed_goal(self, g: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.psi(g).float(), dim=-1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.float().clamp(max=math.log(100.0)).exp()

    def forward(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return (self.embed_state(z) * self.embed_goal(g)).sum(-1)


def action_cost(actions: torch.Tensor, prior: Optional[torch.Tensor], smooth_dims: int,
                smooth_w: float, prior_w: float) -> torch.Tensor:
    """Cost(A) = w_s * mean ||a_{i+1} - a_i||^2 (arm dims) + w_p * mean ||A - A_prior||^2."""
    a = actions.float()
    cost = torch.zeros(a.shape[:-2], device=a.device)
    if smooth_w > 0 and a.shape[-2] > 1:
        d = a[..., 1:, :smooth_dims] - a[..., :-1, :smooth_dims]
        cost = cost + smooth_w * d.pow(2).sum(-1).mean(-1)
    if prior_w > 0 and prior is not None:
        cost = cost + prior_w * (a - prior.float()).pow(2).sum(-1).mean(-1)
    return cost


# --------------------------------------------------------------------------- full system 2
class LISHeads(nn.Module):
    """Container for all System-2 modules + training losses + latent search."""

    def __init__(self, cfg, hidden_size: int, head_dim: int):
        super().__init__()
        self.cfg = cfg
        d = cfg.lis_dim
        self.action_dim, self.horizon = cfg.action_dim, cfg.time_horizon
        self.obs_encoder = ObservationEncoder(hidden_size, d, cfg.lis_num_latents, cfg.lis_num_views)
        self.ctx_encoder = SequencePooler(hidden_size, d, cfg.lis_ctx_tokens)
        self.lang_encoder = SequencePooler(hidden_size, d, 1)
        self.proposer = FlowMatchingProposer(head_dim, cfg.action_dim, cfg.time_horizon, d, cfg.lis_fm_layers)
        self.world_model = LatentWorldModel(d, cfg.action_dim, cfg.time_horizon, cfg.lis_wm_layers)
        self.evaluator = LanguageGuidedEvaluator(d)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _multi_positive_nce(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        pos = torch.logsumexp(log_prob.masked_fill(~pos_mask, float("-inf")), dim=-1)
        return -pos.mean()

    def _negative_actions(self, a: torch.Tensor) -> torch.Tensor:
        """Counterfactual chunks used to teach the evaluator which futures are *not* desirable."""
        negs = [(a + self.cfg.lis_noise_std * torch.randn_like(a)).clamp(-1, 1), a.flip(1)]
        if a.shape[0] > 1:
            negs.append(a.roll(1, dims=0))
        return torch.stack(negs, dim=1)  # (B, M, H, A)

    # ------------------------------------------------------------------ training
    def compute_losses(self, ctx_tokens, goal, head_feats, actions, z_t, z_future, fut_idx, goal_keys) -> Dict[str, torch.Tensor]:
        """
        ctx_tokens (B,K,d) goal (B,d) head_feats (B,H*A,hd) actions (B,H,A) in [-1,1]
        z_t / z_future: (F,Q,d) for the F samples that have a future frame (z_future is stop-grad)
        fut_idx: (F,) indices of those samples in the batch
        goal_keys: list (len F) of hashable instruction keys
        """
        cfg, out = self.cfg, {}
        out["lis_fm"] = self.proposer.loss(head_feats, ctx_tokens, actions)
        if z_t is None or z_t.shape[0] == 0:
            return out

        Fn = z_t.shape[0]
        ctx_f, goal_f, act_f = ctx_tokens[fut_idx], goal[fut_idx], actions[fut_idx]
        z_hat = self.world_model(z_t, ctx_f, act_f)

        # (1) latent prediction: cosine on every latent token
        out["lis_wm"] = (1 - F.cosine_similarity(z_hat.float(), z_future.float(), dim=-1)).mean()

        if Fn > 1:
            # (2) anti-collapse: predicted future must identify its own target within the batch
            p = F.normalize(z_hat.float().mean(1), dim=-1)
            t = F.normalize(z_future.float().mean(1), dim=-1)
            out["lis_wm_nce"] = F.cross_entropy(p @ t.T / 0.1, torch.arange(Fn, device=p.device))
            std = z_t.float().mean(1).std(0)
            out["lis_var"] = F.relu(1.0 - std).mean()

        # (3) language-goal grounding of latent futures (real + imagined)
        ev, scale = self.evaluator, self.evaluator.scale()
        e_g = ev.embed_goal(goal_f)
        uniq = {k: i for i, k in enumerate(dict.fromkeys(goal_keys))}
        gid = torch.tensor([uniq[k] for k in goal_keys], device=e_g.device)
        if len(uniq) > 1:
            pos_mask = gid[:, None] == gid[None, :]
            nce = 0.0
            for z in (z_future, z_hat.detach()):
                nce = nce + self._multi_positive_nce(scale * ev.embed_state(z) @ e_g.T, pos_mask)
            out["lis_lang_nce"] = nce / 2

        # (4) progress: the real future should match the goal better than the present
        r_now = (ev.embed_state(z_t.detach()) * e_g).sum(-1)
        r_fut = (ev.embed_state(z_future) * e_g).sum(-1)
        out["lis_prog"] = F.softplus(scale * (r_now - r_fut)).mean()

        # (5) action ranking in imagination: expert chunk > counterfactual chunks
        with torch.no_grad():
            negs = self._negative_actions(act_f)
            M = negs.shape[1]
            z_neg = self.world_model(z_t.detach().repeat_interleave(M, 0), ctx_f.detach().repeat_interleave(M, 0),
                                     negs.flatten(0, 1))
        r_pos = (ev.embed_state(z_hat.detach()) * e_g).sum(-1)
        r_neg = (ev.embed_state(z_neg) * e_g.repeat_interleave(M, 0)).sum(-1).view(Fn, M)
        out["lis_rank"] = F.softplus(scale * (r_neg - r_pos[:, None])).mean()
        return out

    # ------------------------------------------------------------------ inference: latent search
    @torch.no_grad()
    def search(self, ctx_tokens, goal, head_feats, z_t, prior_actions=None, num_candidates=None,
               cem_iters=None, return_all: bool = False):
        """Imagination search for a single observation (B = 1).

        Returns the best chunk (H, A) in normalised space and a diagnostics dict.
        """
        cfg = self.cfg
        N = num_candidates or cfg.lis_num_candidates
        iters = cfg.lis_cem_iters if cem_iters is None else cem_iters
        n_elite = max(1, min(cfg.lis_cem_elites, N))

        cands = self.proposer.sample(head_feats, ctx_tokens, N, cfg.lis_fm_steps, cfg.lis_noise_scale)[0]  # (N,H,A)
        if prior_actions is not None and cfg.lis_include_prior:
            cands[0] = prior_actions.to(cands.dtype)

        def score(c):
            n = c.shape[0]
            z_hat = self.world_model(z_t.expand(n, -1, -1), ctx_tokens.expand(n, -1, -1), c)
            r = self.evaluator(z_hat, goal.expand(n, -1))
            cost = action_cost(c, prior_actions, cfg.lis_smooth_dims, cfg.lis_smooth_weight, cfg.lis_prior_weight)
            return r - cost, r, cost

        total, sem, cost = score(cands)
        for _ in range(iters):  # optional CEM refinement around the elites
            elite_idx = total.topk(n_elite).indices
            elites = cands[elite_idx]
            mu, std = elites.float().mean(0), elites.float().std(0, unbiased=False) + 1e-3
            new = (mu + std * torch.randn(N - n_elite, *mu.shape, device=mu.device)).clamp(-1, 1).to(cands.dtype)
            cands = torch.cat([elites, new], 0)
            total, sem, cost = score(cands)

        best = int(total.argmax())
        info = {"best_index": best, "best_score": float(total[best]), "best_semantic": float(sem[best]),
                "best_cost": float(cost[best]), "score_mean": float(total.mean()), "score_std": float(total.std()),
                "prior_score": float(total[0]) if (prior_actions is not None and cfg.lis_include_prior and iters == 0) else None}
        if return_all:
            info.update(candidates=cands, scores=total, semantic=sem, cost=cost)
        return cands[best], info
