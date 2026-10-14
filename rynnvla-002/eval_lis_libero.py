"""Resumable, shardable LIBERO / LIBERO-Plus evaluation for RynnVLA-002 and LIS-VLA.

Observation preprocessing, prompt format, warm-up steps, action de-normalisation and chunk
execution are the same as RynnVLA-002's ``eval_solver_libero_continous_w_state.py``.
Additions: explicit episode lists (so N machines can split the work), one JSON line per
episode (so a pre-empted pod can resume and shards can be merged), LIBERO-Plus perturbation
categories, and System-2 search statistics.

Instruction used for LIBERO-Plus variants: LIBERO-Plus derives ``task.language`` from the variant
file name (e.g. "... cabinet view 0 0 100 0 0 initstate 1"). For every category except
"Language Instructions" we therefore give the model the *original* task instruction (the
unperturbed task name, which is what RynnVLA-002 was trained on); for "Language Instructions"
we use the perturbed instruction provided by LIBERO-Plus.

Example (shard 2 of 6, System 1 only):
    python eval_lis_libero.py --ckpt ../ckpts/RynnVLA-002/VLA_model_256/libero_goal --suite libero_goal \
        --benchmark plus --categories "Robot Initial States" "Objects Layout" \
        --time_horizon 5 --search 0 --shard 2 --num_shards 6 --out ../eval_outputs/phase0/goal_plus_s1
"""
import argparse
import json
import os
import random
import re
import time
import traceback

import numpy as np

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
# action / state normalisation used by RynnVLA-002's LIBERO evaluation
ACT_MIN = np.array([-0.9375, -0.9375, -0.9375, -0.24214286, -0.375, -0.36428571, -1.0])
ACT_MAX = np.array([0.9375, 0.9375, 0.9375, 0.34821429, 0.375, 0.375, 1.0])


def unnorm_action(a):
    a = np.asarray(a)[:7]
    return (a + 1) / 2 * (ACT_MAX - ACT_MIN + 1e-8) + ACT_MIN


def libero_language(name):
    """Same rule as LIBERO's grab_language_from_filename (strips the SCENE prefix of LIBERO-10/90 tasks)."""
    if name[0].isupper():
        k = name.find("SCENE")
        name = name[k + 8:] if "SCENE10" in name else name[k + 7:]
    return " ".join(name.split("_"))


def vanilla_task_names(suite, vanilla_root):
    ns = {}
    exec(open(os.path.join(vanilla_root, "libero/libero/benchmark/libero_suite_task_map.py")).read(), ns)
    return ns["libero_task_map"][suite]


def build_episode_list(args):
    """Deterministic list of episodes: dicts with task_index, episode, category, instruction."""
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[args.suite]()
    eps = []
    if args.benchmark == "vanilla":
        for ti in range(bench.n_tasks):
            task = bench.get_task(ti)
            for e in range(args.episodes_per_task):
                eps.append({"task_index": ti, "episode": e, "category": "standard",
                            "task_name": task.name, "instruction": task.language})
    else:
        cls_path = os.path.join(os.path.dirname(benchmark.__file__), "task_classification.json")
        cls = json.load(open(cls_path))[args.suite]
        bases = sorted(vanilla_task_names(args.suite, args.vanilla_root), key=len, reverse=True)
        for item in cls:
            if args.categories and item["category"] not in args.categories:
                continue
            ti = item["id"] - 1
            task = bench.get_task(ti)
            assert task.name == item["name"], (task.name, item["name"])
            if item["category"] == "Language Instructions":
                instr = task.language
            else:
                base = next((b for b in bases if item["name"].startswith(b)), None)
                if base is None:
                    raise ValueError(f"cannot map {item['name']} to an original task")
                instr = libero_language(base)
            eps.append({"task_index": ti, "episode": 0, "category": item["category"],
                        "difficulty": item.get("difficulty_level"), "task_name": task.name, "instruction": instr})
        if args.max_per_category:
            rng = random.Random(0)
            by_cat = {}
            for e in eps:
                by_cat.setdefault(e["category"], []).append(e)
            eps = []
            for cat in sorted(by_cat):
                items = by_cat[cat]
                rng.shuffle(items)
                eps += sorted(items[: args.max_per_category], key=lambda e: e["task_index"])
    if args.limit:
        eps = eps[: args.limit]
    return bench, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", required=True, choices=list(MAX_STEPS))
    ap.add_argument("--benchmark", choices=["vanilla", "plus"], default="vanilla")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--episodes_per_task", type=int, default=50, help="vanilla only")
    ap.add_argument("--max_per_category", type=int, default=0, help="plus: random subset per category (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N episodes only")
    ap.add_argument("--vanilla_root", default=os.environ.get("VANILLA_LIBERO_ROOT", "../third_party/LIBERO"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time_horizon", type=int, required=True)
    ap.add_argument("--his", default="his_2_third_view_wrist_w_state")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--search", type=int, default=0, help="0 = System 1 only, 1 = LIS search")
    ap.add_argument("--num_candidates", type=int, default=None)
    ap.add_argument("--cem_iters", type=int, default=None)
    ap.add_argument("--save_video", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, f"results_shard{args.shard}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            done.add((r["task_index"], r["episode"]))

    bench, episodes = build_episode_list(args)
    mine = [e for i, e in enumerate(episodes) if i % args.num_shards == args.shard]
    todo = [e for e in mine if (e["task_index"], e["episode"]) not in done]
    print(f"{len(episodes)} episodes total, {len(mine)} in this shard, {len(todo)} to run", flush=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    if not todo:
        return

    import torch
    from PIL import Image

    from data.pre_tokenize_action_state import ItemProcessor
    from libero_util.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle, save_rollout_video
    from libero_util.lis_utils import get_action_lis
    from model import load_lis_model

    try:
        import flash_attn  # noqa: F401
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"  # at inference the prompt mask is plain causal, so the backend does not change results
    model = load_lis_model(args.ckpt, action_dim=7, time_horizon=args.time_horizon, max_position_embeddings=8192,
                           mask_image_logits=False, torch_dtype=torch.bfloat16,
                           attn_implementation=attn).to(args.device).eval()
    item_processor = ItemProcessor(target_size=args.resolution, tokenizer=args.tokenizer_path)
    max_steps = MAX_STEPS[args.suite]

    cur_task, env = None, None
    for ep in todo:
        if cur_task != ep["task_index"]:
            if env is not None:
                env.close()
            task = bench.get_task(ep["task_index"])
            env, _ = get_libero_env(task, resolution=args.resolution)
            init_states = bench.get_task_init_states(ep["task_index"])
            cur_task = ep["task_index"]
        t0 = time.time()
        env.reset()
        obs = env.set_init_state(init_states[ep["episode"]])
        t, done_flag, err = 0, False, None
        his_img, his_wrist, chunk = [], [], []
        replay, n_queries, overrides, q_lat = [], 0, 0, []
        while t < max_steps + 10:
            try:
                if t < 10:
                    obs, _, done_flag, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue
                img = get_libero_image(obs, args.resolution)
                wrist = get_libero_image(obs, args.resolution, "robot0_eye_in_hand_image")
                if args.save_video:
                    replay.append(img)
                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                cur_img, cur_wrist = Image.fromarray(img), Image.fromarray(wrist)
                if not chunk:
                    tq = time.time()
                    a = get_action_lis(model, cur_img, cur_wrist, ep["instruction"], item_processor, his_img, his_wrist,
                                       state, args.his, num_candidates=args.num_candidates,
                                       cem_iters=args.cem_iters, enable_search=bool(args.search))
                    q_lat.append(time.time() - tq)
                    n_queries += 1
                    info = model.last_search_info or {}
                    if info.get("search") and info.get("best_index", 0) != 0:
                        overrides += 1
                    chunk = [x.float().cpu().numpy() for x in a]
                act = chunk.pop(0)
                obs, _, done_flag, _ = env.step(unnorm_action(act).tolist())
                his_img.append(cur_img)
                his_wrist.append(cur_wrist)
                if done_flag:
                    break
                t += 1
            except Exception as e:  # same policy as the original evaluator: an exception ends the episode
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                break
        rec = {**ep, "success": bool(done_flag), "steps": t, "seconds": round(time.time() - t0, 2),
               "queries": n_queries, "overrides": overrides,
               "query_latency_mean": round(float(np.mean(q_lat)), 4) if q_lat else None, "error": err,
               "search": args.search, "num_candidates": args.num_candidates}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{ep['category']}] {ep['task_name'][:60]} success={rec['success']} t={t} {rec['seconds']}s", flush=True)
        if args.save_video:
            save_rollout_video(os.path.join(args.out, "videos"), replay, f"{ep['task_index']}_{ep['episode']}",
                               success=done_flag, task_description=ep["instruction"])


if __name__ == "__main__":
    main()
