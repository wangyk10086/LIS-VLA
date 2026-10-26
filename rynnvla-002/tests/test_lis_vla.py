"""CPU smoke tests for LIS-VLA with a tiny Chameleon backbone.

    cd rynnvla-002 && python -m pytest -q tests/test_lis_vla.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import ChameleonXLLMXForConditionalGeneration_ck_action_head as Baseline  # noqa: E402
from model import ChameleonXLLMXForConditionalGeneration_LIS, LISVLAConfig  # noqa: E402
from model.lis import parse_sample  # noqa: E402

H, A, V, IMG_LEN = 5, 7, 2, 12


def tiny_config(**kw):
    cfg = dict(
        vocab_size=65536, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
        vocabulary_map={"<image>": 8711}, mask_image_logits=False, action_dim=A, time_horizon=H,
        lis_dim=32, lis_num_latents=4, lis_ctx_tokens=2, lis_fm_layers=1, lis_wm_layers=1,
        lis_num_candidates=8, lis_fm_steps=4,
    )
    cfg.update(kw)
    return LISVLAConfig(**cfg)


def image_block(g):
    return [8197] + torch.randint(4, 8195, (IMG_LEN,), generator=g).tolist() + [8196]


def action_block(a):  # a: (A,) in [-1, 1]
    bins = torch.linspace(-1, 1, 256)
    return [10004] + (torch.bucketize(a, bins) + 10005).tolist() + [15004]


def make_sample(task_id, g, with_future=True):
    instr = [0] + [16500 + task_id * 50 + i for i in range(8)]
    state = [15504] + torch.randint(15505, 15760, (8,), generator=g).tolist() + [16004]
    prompt = instr + state + sum([image_block(g) for _ in range(2 * V)], [])
    acts = torch.rand(H, A, generator=g) * 2 - 1
    gpt = sum([action_block(a) for a in acts], [])
    if with_future:
        gpt += sum([image_block(g) for _ in range(V)], [])
    return prompt + gpt, [-100] * len(prompt) + gpt


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = ChameleonXLLMXForConditionalGeneration_LIS(tiny_config())
    del m.model.vqmodel
    return m


def test_parse_sample():
    g = torch.Generator().manual_seed(1)
    ids, _ = make_sample(0, g)
    s = parse_sample(torch.tensor(ids), V, H, A)
    assert s.has_chunk and s.has_future
    assert len(s.cur_image_spans) == V and len(s.future_image_spans) == V
    assert s.action_tokens.shape == (H, A)
    assert ids[s.instr_end] == 15504


def test_attention_mask_matches_original_and_fixes_lis_layout(model):
    g = torch.Generator().manual_seed(2)
    vla, _ = make_sample(0, g, with_future=False)
    lis, _ = make_sample(0, g, with_future=True)
    for ids, expect_same in ((vla, True), (vla[: vla.index(10004) + 1], True), (lis, False)):
        x = torch.tensor([ids])
        mine = model.generate_att_mask_3(x)
        theirs = Baseline.generate_att_mask_3(type("M", (), {"device": "cpu"})(), x)
        assert torch.equal(mine, theirs) == expect_same
    # in the LIS layout, chunk actions must not see each other (as at inference time)
    x = torch.tensor([lis])
    starts = (x[0] == 10004).nonzero().flatten()
    m = model.generate_att_mask_3(x)
    assert not m[0, starts[1], starts[0]]


def test_training_forward_backward(model):
    g = torch.Generator().manual_seed(3)
    batch = [make_sample(t % 3, g) for t in range(4)] + [make_sample(1, g, with_future=False)]
    ids, labels = [b[0] for b in batch], [b[1] for b in batch]
    out = model(input_ids=ids, labels=labels, output_hidden_states=True, training=True, att_mask=True)
    c_loss, add, _, _, _, pred, loss_ct = out
    assert pred.shape == (5 * H, A)
    expected = {"lis_fm", "lis_wm", "lis_wm_nce", "lis_var", "lis_lang_nce", "lis_prog", "lis_rank"}
    assert expected <= set(add), set(add)
    total = c_loss + 10 * loss_ct + sum(v * w for v, w in add.values())
    assert torch.isfinite(total)
    model.zero_grad()
    total.backward()
    # zero-initialised output layers block gradients to upstream modules on step 0 (by design);
    # perturb them once to check that every LIS module is wired into the graph.
    with torch.no_grad():
        for mod in (model.lis.proposer.out, model.lis.world_model.out):
            mod.weight.normal_(std=0.02)
    out = model(input_ids=ids, labels=labels, output_hidden_states=True, training=True, att_mask=True)
    model.zero_grad()
    sum(v * w for v, w in out[1].values()).backward()
    for name in ("proposer", "world_model", "evaluator", "obs_encoder", "ctx_encoder", "lang_encoder"):
        grads = [p.grad for p in getattr(model.lis, name).parameters() if p.grad is not None]
        assert grads and any(gr.abs().sum() > 0 for gr in grads), name


def test_lis_losses_decrease():
    torch.manual_seed(0)
    m = ChameleonXLLMXForConditionalGeneration_LIS(tiny_config())
    del m.model.vqmodel
    opt = torch.optim.AdamW(m.lis.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(4)
    data = [make_sample(t % 4, g) for t in range(8)]
    ids, labels = [d[0] for d in data], [d[1] for d in data]
    hist = []
    for _ in range(60):
        _, add, *_ = m(input_ids=ids, labels=labels, output_hidden_states=True, training=True)
        loss = sum(v * w for v, w in add.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append({k: float(v[0]) for k, v in add.items()})
    for k in ("lis_fm", "lis_wm", "lis_lang_nce"):
        assert hist[-1][k] < hist[0][k], (k, hist[0][k], hist[-1][k])


def test_latent_search(model):
    from transformers import GenerationConfig

    g = torch.Generator().manual_seed(5)
    ids, _ = make_sample(2, g, with_future=False)
    prompt = torch.tensor([ids[: ids.index(10004) + 1]])
    gen = GenerationConfig(max_new_tokens=1, do_sample=False, eos_token_id=[8710])
    model.eval()
    chunk = model.generate_lis_action(prompt, gen, return_all=True)
    info = model.last_search_info
    assert chunk.shape == (H, A) and info["search"]
    assert info["candidates"].shape[0] == model.config.lis_num_candidates
    assert chunk.abs().max() <= 1.0
    chunk = model.generate_lis_action(prompt, gen, cem_iters=2)
    assert chunk.shape == (H, A)
    prior = model.generate_lis_action(prompt, gen, enable_search=False)
    assert prior.shape == (H, A)

    # conservative override: an unreachable margin always keeps the System-1 chunk
    model.config.lis_override_z = 1e9
    try:
        gated = model.generate_lis_action(prompt, gen)
        info = model.last_search_info
        assert info["gated"] and info["best_index"] == 0 and info["gap_z"] is not None
        assert torch.allclose(gated.float(), prior.float(), atol=1e-6)
    finally:
        model.config.lis_override_z = 0.0
    model.generate_lis_action(prompt, gen)
    assert not model.last_search_info["gated"]


def test_action_head_checkpoint_compatible():
    from model.modeling_lis_vla import LISActionHead
    from model.modeling_xllmx_chameleon_ck_action_head import ActionHead

    torch.manual_seed(0)
    old = ActionHead(action_dim=1, time_horizon=1).eval()  # hidden size is hard-coded to 4096 upstream
    new = LISActionHead(action_dim=1, time_horizon=1, hidden_size=4096).eval()
    sd_old, sd_new = old.state_dict(), new.state_dict()
    assert {k: v.shape for k, v in sd_old.items()} == {k: v.shape for k, v in sd_new.items()}
    new.load_state_dict(sd_old)
    hs = torch.randn(1, 6, 4096)
    ids = torch.tensor([[1, 2, 3, 4, 5, 10004]])
    a_old, _ = old(hs, ids, eval=True)
    a_new, _ = new(hs, ids, eval=True)
    assert torch.allclose(a_old, a_new, atol=1e-5)


def test_load_checkpoint_without_lis_weights(tmp_path):
    from model import load_lis_model

    torch.manual_seed(0)
    m = ChameleonXLLMXForConditionalGeneration_LIS(tiny_config())
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("lis.")}
    m.save_pretrained(tmp_path, state_dict=sd, safe_serialization=True)
    loaded = load_lis_model(str(tmp_path))
    for n, p in loaded.lis.named_parameters():
        assert torch.isfinite(p).all(), n
    assert loaded.lis.world_model.out.weight.abs().sum() == 0
    assert torch.equal(loaded.action_head.hidden_projection.weight, m.action_head.hidden_projection.weight)


def test_lis_weights_roundtrip(tmp_path):
    """only-trainable checkpoint (lis.*) saved by the trainer loads on top of a base checkpoint"""
    from safetensors.torch import save_file

    from model import load_lis_model
    from model.modeling_lis_vla import ChameleonXLLMXForConditionalGeneration_LIS

    torch.manual_seed(0)
    base = ChameleonXLLMXForConditionalGeneration_LIS(tiny_config(lis_train_scope="lis_only"))
    trainable = base.get_trainable_params()
    assert isinstance(trainable, dict) and trainable and all(k.startswith("lis.") for k in trainable)
    base.save_pretrained(tmp_path / "base")
    trained = {k: torch.randn_like(p) for k, p in trainable.items()}
    (tmp_path / "lis").mkdir()
    save_file(trained, str(tmp_path / "lis" / "model.safetensors"))
    m = load_lis_model(str(tmp_path / "base"), lis_weights=str(tmp_path / "lis"))
    sd = m.state_dict()
    for k, v in trained.items():
        assert torch.equal(sd[k], v.to(sd[k].dtype)), k
    for k, v in base.state_dict().items():
        if not k.startswith("lis."):
            assert torch.equal(sd[k], v), k


def test_value_head_scoring_and_training(model, tmp_path):
    """outcome-trained scorer: plugs into search, gates on P(success), trains from rollout files"""
    import subprocess

    import numpy as np
    from safetensors.torch import save_file
    from transformers import GenerationConfig

    from model.lis.value import ChunkValue

    cfg = model.config
    g = torch.Generator().manual_seed(7)
    ids, _ = make_sample(1, g, with_future=False)
    prompt = torch.tensor([ids[: ids.index(10004) + 1]])
    gen = GenerationConfig(max_new_tokens=1, do_sample=False, eos_token_id=[8710])
    model.eval()
    for mode in ("imag", "direct"):
        model.lis.value_head = ChunkValue(mode, d=cfg.lis_dim, num_latents=cfg.lis_num_latents,
                                          num_ctx=cfg.lis_ctx_tokens, horizon=H, action_dim=A, width=16).eval()
        try:
            model.generate_lis_action(prompt, gen, return_all=True)
            info = model.last_search_info
            assert info["gap_p"] is not None and -1 <= info["gap_p"] <= 1
            assert info["z_t"].shape[-2:] == (cfg.lis_num_latents, cfg.lis_dim)
            cfg.lis_override_margin_p = 2.0  # unreachable: always keep System 1
            model.generate_lis_action(prompt, gen)
            assert model.last_search_info["gated"]
        finally:
            cfg.lis_override_margin_p = 0.0
            del model.lis.value_head

    # synthetic rollouts with the real feature shapes -> train_value.py end to end
    model.save_pretrained(tmp_path / "base")
    (tmp_path / "lis").mkdir()
    save_file({k: v.contiguous() for k, v in model.state_dict().items() if k.startswith("lis.")},
              str(tmp_path / "lis" / "model.safetensors"))
    roll = tmp_path / "roll"
    roll.mkdir()
    rng = np.random.RandomState(0)
    for e in range(24):
        q = 6
        np.savez_compressed(roll / f"ep{e:05d}.npz",
                            z_t=rng.randn(q, cfg.lis_num_latents, cfg.lis_dim).astype(np.float16),
                            ctx=rng.randn(q, cfg.lis_ctx_tokens, cfg.lis_dim).astype(np.float16),
                            goal=rng.randn(q, cfg.lis_dim).astype(np.float16),
                            candidates=rng.uniform(-1, 1, (q, 4, H, A)).astype(np.float16),
                            executed=rng.randint(0, 4, q), t=np.arange(10, 10 + 5 * q, 5),
                            success=np.array(e % 2 == 0), end_step=np.array(60), task_index=np.array(e % 3),
                            sigma=np.array(0.1), joint_noise=np.zeros(7, np.float32), error=np.array(""))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for mode in ("imag", "direct"):
        out = tmp_path / f"v_{mode}.pt"
        r = subprocess.run([sys.executable, os.path.join(root, "tools", "train_value.py"), "--rollouts", str(roll),
                            "--mode", mode, "--ckpt", str(tmp_path / "base"), "--lis_weights", str(tmp_path / "lis"),
                            "--out", str(out), "--epochs", "2", "--device", "cpu"],
                           capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-2000:]
        head = ChunkValue.load(str(out))
        assert head.mode == mode
