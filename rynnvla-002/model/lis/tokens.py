"""Token-level parsing of RynnVLA-002 sequences for LIS-VLA.

RynnVLA-002 serialises every modality into one Chameleon token stream:

    <bos> text ... [state block] [image block]* [action block]* [image block]*
    image block  : 8197 <vq tokens + row separators> 8196
    action block : 10004 <action_dim bins> 15004
    state block  : 15504 <state_dim bins> 16004

LIS-VLA training samples have the layout

    instruction | state | images_t (V views x his) | H action blocks | images_{t+H} (V views)

and inference prompts stop right after the first ``10004``.  This module finds
the spans that the LIS heads need, without assuming fixed lengths.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

IMG_START = 8197
IMG_END = 8196
ACT_START = 10004
ACT_END = 15004
STATE_START = 15504


def _pair_blocks(ids: torch.Tensor, start_id: int, end_id: int) -> List[Tuple[int, int]]:
    """Return (start, end) index pairs. ``end`` is -1 for an unterminated block."""
    starts = (ids == start_id).nonzero(as_tuple=True)[0].tolist()
    ends = (ids == end_id).nonzero(as_tuple=True)[0].tolist()
    blocks, j = [], 0
    for s in starts:
        while j < len(ends) and ends[j] < s:
            j += 1
        if j < len(ends):
            blocks.append((s, ends[j]))
            j += 1
        else:
            blocks.append((s, -1))
    return blocks


@dataclass
class ParsedSample:
    ctx_end: int                                   # index of the first action-start token
    instr_end: int                                 # instruction occupies [0, instr_end)
    cur_image_spans: List[Tuple[int, int]]         # token spans (exclusive of markers) of z_t views
    action_tokens: Optional[torch.Tensor] = None   # (H, action_dim) discrete ids, if a full chunk exists
    future_image_spans: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def has_chunk(self) -> bool:
        return self.action_tokens is not None

    @property
    def has_future(self) -> bool:
        return self.has_chunk and len(self.future_image_spans) > 0


def parse_sample(ids: torch.Tensor, num_views: int, horizon: int, action_dim: int) -> Optional[ParsedSample]:
    """Parse a single 1-D token sequence. Returns None if there is no action block."""
    act_blocks = _pair_blocks(ids, ACT_START, ACT_END)
    if len(act_blocks) == 0:
        return None
    ctx_end = act_blocks[0][0]

    img_blocks = [b for b in _pair_blocks(ids, IMG_START, IMG_END) if b[1] != -1]
    cur = [(s + 1, e) for s, e in img_blocks if e < ctx_end][-num_views:]
    if len(cur) == 0:
        return None

    special = [ctx_end]
    for tok in (IMG_START, STATE_START):
        pos = (ids[:ctx_end] == tok).nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            special.append(int(pos[0]))
    instr_end = max(1, min(special))

    sample = ParsedSample(ctx_end=ctx_end, instr_end=instr_end, cur_image_spans=cur)

    chunk = act_blocks[:horizon]
    if len(chunk) == horizon and all(e != -1 and e - s - 1 == action_dim for s, e in chunk):
        sample.action_tokens = torch.stack([ids[s + 1:e] for s, e in chunk], dim=0)
        last_end = chunk[-1][1]
        fut = [(s + 1, e) for s, e in img_blocks if s > last_end][:num_views]
        if len(fut) == num_views:
            sample.future_image_spans = fut
    return sample


def action_tokens_to_continuous(tok: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """Same de-tokenisation as ``decode_token_ids_to_actions`` in RynnVLA-002 (range [-1, 1])."""
    bins = torch.linspace(-1, 1, n_bins, device=tok.device)
    centers = (bins[:-1] + bins[1:]) / 2.0
    idx = torch.clamp(tok - 1 - ACT_START - 1, min=0, max=centers.shape[0] - 1).long()
    return centers[idx]


def instruction_key(ids: torch.Tensor, sample: ParsedSample) -> Tuple[int, ...]:
    """Hashable key used to group samples that share the same language goal g."""
    return tuple(ids[: sample.instr_end].tolist())
