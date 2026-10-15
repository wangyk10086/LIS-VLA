"""Outcome-trained scorers for System 2 (step 2 after the language-similarity evaluator failed).

Both predict the probability that the episode eventually succeeds if action chunk A is executed now,
and are trained offline on cached features from System-1 rollouts with exploration
(``tools/collect_rollouts.py`` -> ``tools/train_value.py``). The rest of LIS-VLA stays frozen.

  * ``imag``  : V(ẑ_{t+H}, g)       - scores the *imagined* future from the latent world model
  * ``direct``: Q(z_t, ctx, g, A)   - scores the action directly, no imagination (ablation)

Comparing the two isolates the contribution of imagination.
"""
from typing import Dict

import torch
from torch import nn


class ChunkValue(nn.Module):
    def __init__(self, mode: str, d: int = 512, num_latents: int = 16, num_ctx: int = 8,
                 horizon: int = 5, action_dim: int = 7, width: int = 512, dropout: float = 0.1):
        super().__init__()
        assert mode in ("imag", "direct"), mode
        self.mode = mode
        self.hparams = dict(mode=mode, d=d, num_latents=num_latents, num_ctx=num_ctx, horizon=horizon,
                            action_dim=action_dim, width=width, dropout=dropout)
        self.state_in = nn.Sequential(nn.LayerNorm(num_latents * d), nn.Linear(num_latents * d, width))
        self.goal_in = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, width))
        if mode == "direct":
            self.ctx_in = nn.Sequential(nn.LayerNorm(num_ctx * d), nn.Linear(num_ctx * d, width))
            self.act_in = nn.Linear(horizon * action_dim, width)
        n_in = 2 if mode == "imag" else 4
        self.head = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(n_in * width, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, goal: torch.Tensor, state: torch.Tensor, ctx: torch.Tensor = None,
                actions: torch.Tensor = None) -> torch.Tensor:
        """Returns success logits (B,).

        imag  : state = imagined future latents ẑ (B, Q, d)
        direct: state = current latents z_t (B, Q, d), plus ctx (B, K, d) and actions (B, H, A)
        """
        x = [self.state_in(state.flatten(1).float()), self.goal_in(goal.float())]
        if self.mode == "direct":
            x += [self.ctx_in(ctx.flatten(1).float()), self.act_in(actions.flatten(1).float())]
        return self.head(torch.cat(x, -1)).squeeze(-1)

    def save(self, path: str, extra: Dict = None):
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "ChunkValue":
        ck = torch.load(path, map_location=map_location)
        m = cls(**ck["hparams"])
        m.load_state_dict(ck["state_dict"])
        return m.eval()
