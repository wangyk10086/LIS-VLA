"""LIS-VLA: Latent Imagination Search on top of RynnVLA-002.

System 1 (unchanged RynnVLA-002 path): Chameleon backbone -> continuous action head.
System 2 (new): flow-matching proposer -> latent world model -> language-guided
evaluator -> argmax, all in latent space and sharing one backbone forward pass.

The class is a drop-in replacement for ``ChameleonXLLMXForConditionalGeneration_ck_action_head``:
the action head keeps its parameter names, so released RynnVLA-002 checkpoints load directly
and only the ``lis.*`` parameters start from scratch.
"""
import logging
import math
from typing import Dict, List, Optional

import torch
from torch import nn

from .chameleon import ChameleonForConditionalGeneration
from .configuration_xllmx_chameleon import ChameleonXLLMXConfig
from .modeling_xllmx_chameleon_ck_action_head import (
    ChameleonXLLMXForConditionalGeneration_ck_action_head,
    L1RegressionActionHead,
)
from .lis import LISHeads, action_tokens_to_continuous, instruction_key, parse_sample
from .lis.tokens import ACT_END, ACT_START, IMG_END, IMG_START, _pair_blocks
from .lis.modules import pad_sequences

logger = logging.getLogger(__name__)

__all__ = ["LISVLAConfig", "ChameleonXLLMXForConditionalGeneration_LIS"]


LIS_DEFAULTS = dict(
    lis_dim=512,                 # width of the latent space
    lis_num_latents=16,          # Q latent tokens per observation (z_t)
    lis_num_views=2,             # views per timestep (third-person + wrist)
    lis_ctx_tokens=8,            # K context tokens pooled from the backbone
    lis_fm_layers=3,
    lis_wm_layers=4,
    lis_fm_steps=10,             # Euler steps for flow-matching sampling
    lis_noise_scale=1.0,
    lis_num_candidates=16,       # N
    lis_cem_iters=0,             # optional CEM refinement rounds
    lis_cem_elites=4,
    lis_override_z=0.0,          # >0: replace System 1 only if the best candidate wins by this many score SDs
    lis_include_prior=True,      # candidate #0 = deterministic System-1 chunk
    lis_smooth_weight=0.05,
    lis_smooth_dims=6,           # smoothness is applied to arm dims, not the gripper
    lis_prior_weight=0.0,
    lis_noise_std=0.3,           # perturbation for counterfactual chunks (ranking loss)
    lis_detach_backbone=True,    # LIS losses do not back-propagate into the 7B backbone
    lis_enable_search=True,
    # which parameters the trainer updates: "all" | "lis_and_head" | "lis_only"
    # ("lis_only" trains System 2 on top of a frozen RynnVLA-002 checkpoint, ~1 GPU-day scale)
    lis_train_scope="all",
    # loss weights (added to the trainer's additional_loss_dict)
    lis_fm_weight=1.0,
    lis_wm_weight=1.0,
    lis_wm_nce_weight=0.1,
    lis_var_weight=0.1,
    lis_lang_nce_weight=0.1,
    lis_prog_weight=0.05,
    lis_rank_weight=0.1,
)


class LISVLAConfig(ChameleonXLLMXConfig):
    def __init__(self, **kwargs):
        for k, v in LIS_DEFAULTS.items():
            setattr(self, k, kwargs.pop(k, v))
        super().__init__(**kwargs)


class LISActionHead(nn.Module):
    """RynnVLA-002 ``ActionHead`` with (i) configurable hidden size, (ii) padded context
    positions masked out, and (iii) optional return of the action-query features that
    condition the flow-matching proposer. Parameter names are unchanged."""

    def __init__(self, action_dim=7, time_horizon=8, hidden_size=4096, hidden_size_factor=0.25, num_encoder_layers=2):
        super().__init__()
        self.action_dim, self.time_horizon = action_dim, time_horizon
        self.num_encoder_layers = num_encoder_layers
        self.hidden_size = hidden_size
        self.reduced_hidden_size = int(hidden_size * hidden_size_factor)
        self.action_token_embeddings = nn.Embedding(1, time_horizon * action_dim * hidden_size)
        nn.init.normal_(self.action_token_embeddings.weight, std=0.02)
        self.hidden_projection = nn.Linear(hidden_size, self.reduced_hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=self.reduced_hidden_size, nhead=4,
                                           dim_feedforward=self.reduced_hidden_size * 4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers,
                                                         norm=nn.LayerNorm(self.reduced_hidden_size))
        self.output_projection = L1RegressionActionHead(self.reduced_hidden_size, self.reduced_hidden_size,
                                                        time_horizon, action_dim)

    def forward(self, hidden_states, input_ids, attention_mask=None, target_token_id=ACT_START, eval=False,
                return_features=False):
        T = self.time_horizon * self.action_dim
        act_tok = self.action_token_embeddings.weight.view(T, self.hidden_size)
        ctx, idx = [], []
        for i in range(hidden_states.shape[0]):
            pos = (input_ids[i] == target_token_id).nonzero(as_tuple=True)[0]
            if len(pos) > 1 or (eval and len(pos) > 0):
                ctx.append(hidden_states[i, : int(pos[0])])
                idx.append(i)
        flag = len(ctx) > 0
        if not flag:
            ctx, idx = [hidden_states[0, 0:1]], [0]

        seqs = [torch.cat([c, act_tok.to(c.dtype)], dim=0) for c in ctx]
        x, pad = pad_sequences(seqs)
        h = self.transformer_encoder(self.hidden_projection(x), src_key_padding_mask=pad)
        feats = torch.stack([h[j, c.shape[0]: c.shape[0] + T] for j, c in enumerate(ctx)], dim=0)
        actions = self.output_projection(feats).reshape(-1, self.action_dim)
        if return_features:
            return actions, flag, feats, idx
        return actions, flag


class ChameleonXLLMXForConditionalGeneration_LIS(ChameleonXLLMXForConditionalGeneration_ck_action_head):
    config_class = LISVLAConfig

    def __init__(self, config):
        ChameleonForConditionalGeneration.__init__(self, config)
        self.init_input_ids = None
        self.action_dim = config.action_dim
        self.action_head = LISActionHead(config.action_dim, config.time_horizon, hidden_size=config.hidden_size)
        self.lis = LISHeads(config, config.hidden_size, self.action_head.reduced_hidden_size)
        self.last_search_info: Optional[Dict] = None
        self.post_init()
        reset_lis_parameters(self.lis)

    # ------------------------------------------------------------------ attention mask
    def generate_att_mask_3(self, input_ids):
        """Same rule as RynnVLA-002 (action blocks of one chunk cannot attend to each other), but
        the reference image is the last image *before the first action block*. This keeps the
        inference-time mask for LIS samples that append the future frame after the chunk, and is
        identical to the original rule for all existing VLA / world-model sample layouts."""
        B, L = input_ids.shape
        mask = torch.tril(torch.ones(L, L, device=input_ids.device)).unsqueeze(0).repeat(B, 1, 1).bool()
        for b in range(B):
            acts = [(s, e if e != -1 else L - 1) for s, e in _pair_blocks(input_ids[b], ACT_START, ACT_END)]
            if len(acts) < 2:
                continue
            imgs = [(s, e if e != -1 else L - 1) for s, e in _pair_blocks(input_ids[b], IMG_START, IMG_END)]
            prior = [e for _, e in imgs if e < acts[0][0]]
            ref = prior[-1] if prior else -1
            for i, (s, e) in enumerate(acts):
                if s > ref:
                    for ps, pe in acts[:i]:
                        if pe < s:
                            mask[b, s: e + 1, ps: pe + 1] = 0
        return mask

    # ------------------------------------------------------------------ LIS features
    def _lis_inputs(self, hidden_states, input_ids, samples: List, batch_idx: List[int]):
        cfg = self.config
        hs = hidden_states.detach() if cfg.lis_detach_backbone else hidden_states
        ctx = self.lis.ctx_encoder([hs[b, : s.ctx_end] for b, s in zip(batch_idx, samples)])
        goal = self.lis.lang_encoder([hs[b, : s.instr_end] for b, s in zip(batch_idx, samples)])[:, 0]
        return ctx, goal

    def _image_embeds(self, input_ids):
        with torch.no_grad():
            return self.model.embed_tokens(input_ids)

    def _lis_training_losses(self, hidden_states, input_ids, head_feats, head_idx):
        cfg = self.config
        keep, samples = [], []
        for j, b in enumerate(head_idx):
            s = parse_sample(input_ids[b], cfg.lis_num_views, cfg.time_horizon, cfg.action_dim)
            if s is not None and s.has_chunk:
                keep.append(j)
                samples.append(s)
        if not keep:
            return {}
        batch_idx = [head_idx[j] for j in keep]
        feats = head_feats[keep]
        if cfg.lis_detach_backbone:
            feats = feats.detach()
        ctx, goal = self._lis_inputs(hidden_states, input_ids, samples, batch_idx)
        actions = action_tokens_to_continuous(torch.stack([s.action_tokens for s in samples])).to(ctx.dtype)

        fut = [k for k, s in enumerate(samples) if s.has_future]
        z_t = z_f = None
        keys = []
        if fut:
            emb = self._image_embeds(input_ids)
            cur = [[emb[batch_idx[k], a:e] for a, e in samples[k].cur_image_spans] for k in fut]
            nxt = [[emb[batch_idx[k], a:e] for a, e in samples[k].future_image_spans] for k in fut]
            z_t = self.lis.obs_encoder(cur)
            with torch.no_grad():
                z_f = self.lis.obs_encoder(nxt)
            keys = [instruction_key(input_ids[batch_idx[k]], samples[k]) for k in fut]
        fut_idx = torch.tensor(fut, device=ctx.device, dtype=torch.long)
        return self.lis.compute_losses(ctx, goal, feats, actions, z_t, z_f, fut_idx, keys)

    def _lis_zero(self, ref: torch.Tensor) -> torch.Tensor:
        # keeps every LIS parameter in the autograd graph (safe for FSDP / DDP on empty batches)
        return sum(p.sum() for p in self.lis.parameters()) * 0.0 + ref.sum() * 0.0

    # ------------------------------------------------------------------ training forward
    def forward(self, input_ids=None, labels=None, training=False, att_mask=True, **kwargs):
        if not training:
            return super().forward(input_ids=input_ids, labels=labels, training=False, att_mask=att_mask, **kwargs)

        max_tokens = min(max(len(x) for x in input_ids), self.config.max_position_embeddings)
        input_ids = [x[:max_tokens] for x in input_ids]
        labels = [x[:max_tokens] for x in labels]
        input_ids = torch.tensor([x + [0] * (max_tokens - len(x)) for x in input_ids], dtype=torch.int64, device=self.device)
        labels = torch.tensor([x + [-100] * (max_tokens - len(x)) for x in labels], dtype=torch.int64, device=self.device)
        attention_mask = self.generate_att_mask_3(input_ids) if att_mask else None

        result = ChameleonForConditionalGeneration.forward(
            self, input_ids=input_ids, labels=labels, use_cache=False, attention_mask=attention_mask, **kwargs
        )
        c_loss = result[0]
        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:
            logits = result[1]
            shift_logits, shift_labels = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[shift_labels >= 0].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)

        if "output_hidden_states" not in kwargs:
            return c_loss, additional_loss_dict

        hidden_states = result[2][-1]
        predicted_actions, flag, head_feats, head_idx = self.action_head(
            hidden_states=hidden_states, input_ids=input_ids, attention_mask=None,
            target_token_id=ACT_START, return_features=True,
        )
        if not flag:
            additional_loss_dict["lis_fm"] = (self._lis_zero(predicted_actions), self.config.lis_fm_weight)
            return (c_loss, additional_loss_dict, result[1], hidden_states, labels, predicted_actions,
                    predicted_actions.mean() * 0)

        labels_action_dis, _ = self.get_action_hs_label(hidden_states, labels)
        labels_action_ct = self.decode_token_ids_to_actions(labels_action_dis)
        loss_ct = torch.nn.functional.l1_loss(predicted_actions, labels_action_ct)

        lis_losses = self._lis_training_losses(hidden_states, input_ids, head_feats, head_idx)
        zero = self._lis_zero(predicted_actions)
        if "lis_fm" not in lis_losses:
            lis_losses["lis_fm"] = zero
        else:
            lis_losses["lis_fm"] = lis_losses["lis_fm"] + zero
        for name, value in lis_losses.items():
            additional_loss_dict[name] = (value, getattr(self.config, f"{name}_weight"))

        return c_loss, additional_loss_dict, result[1], hidden_states, labels, predicted_actions, loss_ct

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def generate_lis_action(self, input_ids, generation_config, num_candidates=None, cem_iters=None,
                            enable_search=None, return_all=False):
        """System 1 + System 2. Returns a (H, action_dim) chunk in normalised [-1, 1] space.
        Diagnostics of the last search are stored in ``self.last_search_info``."""
        self.init_input_ids = None
        res = ChameleonForConditionalGeneration.generate(
            self, input_ids=input_ids, generation_config=generation_config,
            output_hidden_states=True, training=False, return_dict_in_generate=True,
        )
        full_ids = res["sequences"]
        hidden = torch.cat([step[-1] for step in res["hidden_states"]], dim=1)
        prior, flag, feats, _ = self.action_head(hidden, full_ids, None, ACT_START, eval=True, return_features=True)
        H, A = self.action_head.time_horizon, self.action_head.action_dim
        if not flag:
            logger.warning("No action-start token found; returning zero actions")
            return torch.zeros(H, A, device=input_ids.device)
        prior = prior.reshape(H, A)

        search = self.config.lis_enable_search if enable_search is None else enable_search
        sample = parse_sample(full_ids[0], self.config.lis_num_views, H, A)
        if not search or sample is None:
            self.last_search_info = {"search": False}
            return prior

        ctx, goal = self._lis_inputs(hidden, full_ids, [sample], [0])
        emb = self._image_embeds(full_ids)
        z_t = self.lis.obs_encoder([[emb[0, a:e] for a, e in sample.cur_image_spans]])
        best, info = self.lis.search(ctx, goal, feats, z_t, prior_actions=prior, num_candidates=num_candidates,
                                     cem_iters=cem_iters, return_all=return_all)
        info["search"] = True
        self.last_search_info = info
        return best.to(prior.dtype)

    # Note on FSDP: the LIS heads are deliberately *not* in ``get_fsdp_wrap_module_list``; they are
    # small and stay in the root unit, so their non-``forward`` methods (losses / search) always
    # see unsharded parameters.

    def get_trainable_params(self):
        """Hook used by the RynnVLA-002 trainer to decide ``requires_grad``."""
        scope = self.config.lis_train_scope
        prefixes = {"all": ("",), "lis_and_head": ("lis.", "action_head."), "lis_only": ("lis.",)}[scope]
        # a dict: the trainer's ``--only_save_trainable`` path calls ``.keys()`` on it
        return {n: p for n, p in self.named_parameters() if n.startswith(prefixes)}


@torch.no_grad()
def reset_lis_parameters(lis: nn.Module) -> None:
    """Deterministic initialisation of all LIS parameters (also used after ``from_pretrained``
    on a checkpoint that does not contain them, where HF would leave raw Parameters empty)."""
    for m in lis.modules():
        if isinstance(m, nn.MultiheadAttention):
            m._reset_parameters()
        elif isinstance(m, (nn.Linear, nn.LayerNorm)):
            m.reset_parameters()
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    for name, p in lis.named_parameters():
        leaf = name.split(".")[-1]
        if leaf in ("queries", "pos", "act_pos", "type_emb"):
            nn.init.normal_(p, std=0.02)
        elif leaf == "logit_scale":
            p.fill_(math.log(1 / 0.07))
    for mod in (lis.proposer.out, lis.world_model.out):
        nn.init.zeros_(mod.weight)
        nn.init.zeros_(mod.bias)


def read_lis_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Read the ``lis.*`` tensors from a checkpoint directory or file (safetensors or torch .bin).
    Training with ``--only_save_trainable`` writes exactly these tensors (~50M parameters)."""
    import glob
    import os

    files = [path] if os.path.isfile(path) else (
        sorted(glob.glob(os.path.join(path, "*.safetensors"))) or sorted(glob.glob(os.path.join(path, "*.bin"))))
    if not files:
        raise FileNotFoundError(f"no weights found in {path}")
    sd = {}
    for f in files:
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            part = load_file(f)
        else:
            part = torch.load(f, map_location="cpu")
        sd.update({k: v for k, v in part.items() if k.startswith("lis.")})
    if not sd:
        raise ValueError(f"{path} contains no lis.* tensors")
    return sd


def load_lis_model(path: str, lis_weights: Optional[str] = None, **kwargs):
    """``from_pretrained`` that (re)initialises the LIS heads when the checkpoint lacks them,
    e.g. when fine-tuning from a released RynnVLA-002 checkpoint. ``lis_weights`` then loads
    trained System-2 weights (a small checkpoint with only ``lis.*``) on top of the base model."""
    model, info = ChameleonXLLMXForConditionalGeneration_LIS.from_pretrained(path, output_loading_info=True, **kwargs)
    missing = [k for k in info.get("missing_keys", []) if k.startswith("lis.")]
    if missing:
        logger.info(f"LIS heads not found in {path} ({len(missing)} tensors); initialising from scratch")
        reset_lis_parameters(model.lis)
    if lis_weights:
        sd = read_lis_state_dict(lis_weights)
        params = {k for k, _ in model.named_parameters() if k.startswith("lis.")}
        absent = params - set(sd)
        unexpected = set(sd) - {k for k in model.state_dict() if k.startswith("lis.")}
        if absent or unexpected:
            raise ValueError(f"LIS weights in {lis_weights} do not match the model: "
                             f"missing {sorted(absent)[:5]}, unexpected {sorted(unexpected)[:5]}")
        ref = next(model.lis.parameters())
        model.load_state_dict({k: v.to(ref.dtype) for k, v in sd.items()}, strict=False)
        logger.info(f"loaded {len(sd)} trained LIS tensors from {lis_weights}")
    return model
