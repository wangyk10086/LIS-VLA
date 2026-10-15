"""Collect System-1 rollouts with exploration on the LIBERO *training* tasks, for outcome-trained
System-2 scorers (``tools/train_value.py``).

Episodes use the standard LIBERO init states with a random perturbation of the robot's initial
joint positions (our own perturbations - LIBERO-Plus test variants are never used). At every
query, System 2 proposes N candidate chunks (candidate 0 = System 1); with probability --explore
a random non-System-1 candidate is executed instead, so the data contains alternative actions
and their outcomes. Per query we store the frozen LIS features (z_t, ctx, goal), all candidate
chunks, which one was executed and when; per episode the outcome. One .npz file per episode,
resumable, shardable, with a wall-clock deadline.

    python tools/collect_rollouts.py --ckpt ... --lis_weights ... --out /workspace/data/rollouts_goal \\
        --episodes 1000 --shard 0 --num_shards 2 --deadline <unix time>
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def perturb_robot(env, sigma, rng):
    """Add N(0, sigma) to the robot's arm joint positions (gripper untouched). Returns the offsets."""
    inner = getattr(env, "env", env)
    robot = inner.robots[0]
    idx = list(robot._ref_joint_pos_indexes)
    sim = inner.sim
    noise = rng.normal(0.0, sigma, len(idx))
    if sigma > 0:
        q = sim.data.qpos[idx].copy() + noise
        robot.set_robot_joint_positions(q)
        sim.forward()
    return noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lis_weights", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time_horizon", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--explore", type=float, default=0.3)
    ap.add_argument("--num_candidates", type=int, default=16)
    ap.add_argument("--sigma_max", type=float, default=0.3, help="joint noise std ~ U(0, sigma_max) rad per episode")
    ap.add_argument("--deadline", type=float, default=0, help="unix time after which no new episode starts")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--his", default="his_2_third_view_wrist_w_state")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from PIL import Image

    from data.pre_tokenize_action_state import ItemProcessor
    from eval_lis_libero import MAX_STEPS, unnorm_action
    from libero.libero import benchmark
    from libero_util.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle
    from libero_util.lis_utils import build_prompt_tokens
    from model import load_lis_model
    from transformers import GenerationConfig

    bench = benchmark.get_benchmark_dict()[args.suite]()
    # deterministic episode list: (task, init state, sigma, seed); sigma = 0 for every 8th episode
    plan = []
    rng0 = np.random.RandomState(1234)
    for i in range(args.episodes):
        ti = i % bench.n_tasks
        n_init = len(bench.get_task_init_states(ti))
        plan.append({"id": i, "task_index": ti, "init": int(rng0.randint(n_init)),
                     "sigma": 0.0 if i % 8 == 0 else float(rng0.uniform(0.03, args.sigma_max)),
                     "seed": int(rng0.randint(2 ** 31))})
    mine = [p for p in plan if p["id"] % args.num_shards == args.shard
            and not os.path.exists(os.path.join(args.out, f"ep{p['id']:05d}.npz"))]
    print(f"{len(plan)} planned, {len(mine)} to run in shard {args.shard}", flush=True)

    model = load_lis_model(args.ckpt, lis_weights=args.lis_weights, action_dim=7, time_horizon=args.time_horizon,
                           max_position_embeddings=8192, mask_image_logits=False, torch_dtype=torch.bfloat16,
                           attn_implementation="sdpa").to(args.device).eval()
    ip = ItemProcessor(target_size=args.resolution, tokenizer=args.tokenizer_path)
    gen = GenerationConfig(max_new_tokens=1, max_length=model.config.max_position_embeddings, temperature=1,
                           top_k=None, do_sample=False, eos_token_id=[8710])
    max_steps = MAX_STEPS[args.suite]

    cur_task, env = None, None
    for p in mine:
        if args.deadline and time.time() > args.deadline:
            print("deadline reached", flush=True)
            break
        rng = np.random.RandomState(p["seed"])
        torch.manual_seed(p["seed"])
        task = bench.get_task(p["task_index"])
        if cur_task != p["task_index"]:
            if env is not None:
                env.close()
            env, _ = get_libero_env(task, resolution=args.resolution)
            cur_task = p["task_index"]
        t0 = time.time()
        env.reset()
        obs = env.set_init_state(bench.get_task_init_states(p["task_index"])[p["init"]])
        noise = perturb_robot(env, p["sigma"], rng)
        Z, C, G, CAND, EXEC, TQ = [], [], [], [], [], []
        his_img, his_wrist, chunk = [], [], []
        t, done, err = 0, False, None
        while t < max_steps + 10:
            try:
                if t < 10:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue
                img = get_libero_image(obs, args.resolution)
                wrist = get_libero_image(obs, args.resolution, "robot0_eye_in_hand_image")
                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                ci, cw = Image.fromarray(img), Image.fromarray(wrist)
                if not chunk:
                    toks = build_prompt_tokens(ci, cw, task.language, ip, his_img, his_wrist, state, args.his)
                    ids = torch.tensor(toks, dtype=torch.int64, device=model.device).unsqueeze(0)
                    model.generate_lis_action(ids, gen, num_candidates=args.num_candidates, enable_search=True,
                                              return_all=True)
                    info = model.last_search_info
                    if not info.get("search"):
                        raise RuntimeError("search unavailable for this prompt")
                    cands = info["candidates"].float()
                    k = int(rng.randint(1, cands.shape[0])) if rng.rand() < args.explore else 0
                    Z.append(info["z_t"][0].half().cpu().numpy())
                    C.append(info["ctx"][0].half().cpu().numpy())
                    G.append(info["goal"][0].half().cpu().numpy())
                    CAND.append(cands.cpu().numpy().astype(np.float16))
                    EXEC.append(k)
                    TQ.append(t)
                    chunk = [x for x in cands[k].cpu().numpy()]
                act = chunk.pop(0)
                obs, _, done, _ = env.step(unnorm_action(act).tolist())
                his_img.append(ci)
                his_wrist.append(cw)
                if done:
                    break
                t += 1
            except Exception as e:  # an exception ends the episode (as in the evaluator)
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                break
        if not Z:
            print(f"episode {p['id']} produced no queries ({err})", flush=True)
            if err:
                raise SystemExit(1)  # fail fast on systematic errors
            continue
        np.savez_compressed(
            os.path.join(args.out, f"ep{p['id']:05d}.npz"),
            z_t=np.stack(Z), ctx=np.stack(C), goal=np.stack(G), candidates=np.stack(CAND),
            executed=np.array(EXEC), t=np.array(TQ), success=np.array(bool(done)), end_step=np.array(t),
            task_index=np.array(p["task_index"]), sigma=np.array(p["sigma"]), joint_noise=noise.astype(np.float32),
            error=np.array(err or ""))
        with open(os.path.join(args.out, f"log_shard{args.shard}.jsonl"), "a") as f:
            f.write(json.dumps({"id": p["id"], "task": p["task_index"], "sigma": round(p["sigma"], 3),
                                "success": bool(done), "steps": t, "queries": len(Z),
                                "explored": int(sum(1 for k in EXEC if k)), "seconds": round(time.time() - t0, 1),
                                "error": err}) + "\n")
        print(f"ep {p['id']} task {p['task_index']} sigma {p['sigma']:.2f} success={done} t={t} "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
