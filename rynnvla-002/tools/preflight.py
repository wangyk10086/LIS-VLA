"""Pod preflight: GPU, model load, prompt tokenisation and simulator rendering speed.
Fails fast so a mis-configured pod is stopped before it burns hours.

    python tools/preflight.py --ckpt ../ckpts/RynnVLA-002/VLA_model_256/libero_goal --tokenizer_path ../ckpts/lumina_tokenizer
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time_horizon", type=int, default=5)
    ap.add_argument("--skip_model", action="store_true")
    args = ap.parse_args()
    report = {}

    import torch
    assert torch.cuda.is_available(), "no CUDA GPU visible"
    report["gpu"] = torch.cuda.get_device_name(0)
    report["torch"] = torch.__version__

    # simulator + rendering speed (EGL on the GPU should be well under 30 ms/step)
    from libero.libero import benchmark
    from libero_util.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
    bench = benchmark.get_benchmark_dict()[args.suite]()
    task = bench.get_task(0)
    env, _ = get_libero_env(task, resolution=256)
    env.reset()
    obs = env.set_init_state(bench.get_task_init_states(0)[0])
    env.step(get_libero_dummy_action())
    t = time.time()
    for _ in range(30):
        obs, *_ = env.step(get_libero_dummy_action())
    ms = (time.time() - t) / 30 * 1000
    report["sim_ms_per_step"] = round(ms, 1)
    try:
        from OpenGL import GL
        report["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
    except Exception as e:  # noqa: BLE001
        report["gl_renderer"] = f"unknown ({type(e).__name__})"
    img = get_libero_image(obs, 256)
    assert img.shape == (256, 256, 3)

    if not args.skip_model:
        from PIL import Image
        from data.pre_tokenize_action_state import ItemProcessor
        from libero_util.lis_utils import build_prompt_tokens
        from model import load_lis_model
        ip = ItemProcessor(target_size=256, tokenizer=args.tokenizer_path)
        toks = build_prompt_tokens(Image.fromarray(img), Image.fromarray(img), task.language, ip, [], [],
                                   np.zeros(8), "his_2_third_view_wrist_w_state")
        report["prompt_tokens"] = len(toks)
        t = time.time()
        model = load_lis_model(args.ckpt, action_dim=7, time_horizon=args.time_horizon, max_position_embeddings=8192,
                               mask_image_logits=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        report["model_load_s"] = round(time.time() - t, 1)
        from libero_util.lis_utils import get_action_lis
        for search in (False, True):
            get_action_lis(model, Image.fromarray(img), Image.fromarray(img), task.language, ip, [], [], np.zeros(8),
                           "his_2_third_view_wrist_w_state", enable_search=search)  # warm-up
            torch.cuda.synchronize()
            t = time.time()
            a = get_action_lis(model, Image.fromarray(img), Image.fromarray(img), task.language, ip, [], [], np.zeros(8),
                               "his_2_third_view_wrist_w_state", enable_search=search)
            torch.cuda.synchronize()
            report[f"query_s_search{int(search)}"] = round(time.time() - t, 3)
            assert tuple(a.shape) == (args.time_horizon, 7) and torch.isfinite(a.float()).all()
        report["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)

    print(json.dumps(report, indent=1))
    if report["sim_ms_per_step"] > 60:
        print("FAIL: simulator renders slowly (likely CPU/llvmpipe rendering). Restart the pod with "
              "NVIDIA_DRIVER_CAPABILITIES=all so EGL can use the GPU.")
        sys.exit(2)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
