"""Counterfactual branching: can a different action chunk rescue an episode System 1 fails?

On LIBERO training tasks with our own robot initial-joint perturbations (never LIBERO-Plus test
variants), run System 1 and save simulator snapshots at every --every-th query. After the episode:
  * failed episode   -> branch at up to --points_fail snapshots
  * succeeded episode -> branch at --points_success snapshots (to see how often alternatives break it)
At a branch point the simulator is restored and 1 + --alts branches are played to the end: the
System-1 chunk and random alternative chunks from the System-2 proposer, each followed by System 1.

Per branch point we store the frozen LIS features (z_t, ctx, goal), all candidates, the branched
chunk indices and their outcomes - the "same state, different action, own outcome" data an action
scorer needs - and summarise the head-room with tools/branch_summary.py.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/ (collect_rollouts.perturb_robot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lis_weights", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time_horizon", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed_offset", type=int, default=100000, help="keeps scenes disjoint from collect_rollouts")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--every", type=int, default=4, help="snapshot every N-th query")
    ap.add_argument("--points_fail", type=int, default=3)
    ap.add_argument("--points_success", type=int, default=1)
    ap.add_argument("--alts", type=int, default=3)
    ap.add_argument("--num_candidates", type=int, default=16)
    ap.add_argument("--sigma_max", type=float, default=0.3)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--his", default="his_2_third_view_wrist_w_state")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from PIL import Image

    from collect_rollouts import perturb_robot
    from data.pre_tokenize_action_state import ItemProcessor
    from eval_lis_libero import MAX_STEPS, unnorm_action
    from libero.libero import benchmark
    from libero_util.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle
    from libero_util.lis_utils import build_prompt_tokens
    from model import load_lis_model
    from transformers import GenerationConfig

    bench = benchmark.get_benchmark_dict()[args.suite]()
    plan, rng0 = [], np.random.RandomState(4321)
    for i in range(args.episodes):
        ti = i % bench.n_tasks
        plan.append({"id": args.seed_offset + i, "task_index": ti,
                     "init": int(rng0.randint(len(bench.get_task_init_states(ti)))),
                     "sigma": float(rng0.uniform(0.05, args.sigma_max)), "seed": int(rng0.randint(2 ** 31))})
    mine = [p for k, p in enumerate(plan) if k % args.num_shards == args.shard
            and not os.path.exists(os.path.join(args.out, f"ep{p['id']:06d}.npz"))]
    print(f"{len(mine)} episodes to run in shard {args.shard}", flush=True)

    model = load_lis_model(args.ckpt, lis_weights=args.lis_weights, action_dim=7, time_horizon=args.time_horizon,
                           max_position_embeddings=8192, mask_image_logits=False, torch_dtype=torch.bfloat16,
                           attn_implementation="sdpa").to(args.device).eval()
    ip = ItemProcessor(target_size=args.resolution, tokenizer=args.tokenizer_path)
    gen = GenerationConfig(max_new_tokens=1, max_length=model.config.max_position_embeddings, temperature=1,
                           top_k=None, do_sample=False, eos_token_id=[8710])
    max_steps = MAX_STEPS[args.suite]

    def sim_state(env):
        return env.get_sim_state() if hasattr(env, "get_sim_state") else env.env.sim.get_state().flatten()

    def query(task, ci, cw, state, his_img, his_wrist, search):
        toks = build_prompt_tokens(ci, cw, task.language, ip, his_img, his_wrist, state, args.his)
        ids = torch.tensor(toks, dtype=torch.int64, device=model.device).unsqueeze(0)
        return model.generate_lis_action(ids, gen, num_candidates=args.num_candidates, enable_search=search,
                                         return_all=search)

    def play(env, task, obs, t, his_img, his_wrist, first_chunk=None, record=None):
        """System 1 from (obs, t) to the end, with the same per-step history as eval_lis_libero.py.
        Optionally execute first_chunk at the first query, and record a snapshot at every
        --every-th query. Returns (success, end_step)."""
        chunk, n_q, done = [], 0, False
        his_img, his_wrist = list(his_img), list(his_wrist)
        while t < max_steps + 10:
            img = get_libero_image(obs, args.resolution)
            wrist = get_libero_image(obs, args.resolution, "robot0_eye_in_hand_image")
            state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"]))
            ci, cw = Image.fromarray(img), Image.fromarray(wrist)
            if not chunk:
                if first_chunk is not None and n_q == 0:
                    chunk = [np.asarray(a, dtype=np.float32) for a in first_chunk]
                else:
                    snap = record is not None and n_q % args.every == 0
                    s1 = query(task, ci, cw, state, his_img, his_wrist, search=snap)
                    info = model.last_search_info or {}
                    if snap and info.get("search"):
                        s1 = info["prior"]
                        record.append({"t": t, "sim": sim_state(env), "his_img": list(his_img),
                                       "his_wrist": list(his_wrist), "z_t": info["z_t"][0].half().cpu().numpy(),
                                       "ctx": info["ctx"][0].half().cpu().numpy(),
                                       "goal": info["goal"][0].half().cpu().numpy(),
                                       "cands": info["candidates"].float().cpu().numpy()})
                    chunk = [x for x in s1.float().cpu().numpy()]
                n_q += 1
            act = chunk.pop(0)
            obs, _, done, _ = env.step(unnorm_action(act).tolist())
            his_img.append(ci)
            his_wrist.append(cw)
            if done:
                break
            t += 1
        return bool(done), t

    cur_task, env = None, None
    for p in mine:
        if args.deadline and time.time() > args.deadline:
            print("deadline reached", flush=True)
            break
        t0 = time.time()
        rng = np.random.RandomState(p["seed"])
        torch.manual_seed(p["seed"])
        task = bench.get_task(p["task_index"])
        if cur_task != p["task_index"]:
            if env is not None:
                env.close()
            env, _ = get_libero_env(task, resolution=args.resolution)
            cur_task = p["task_index"]
        main_success = None
        try:
            env.reset()
            obs = env.set_init_state(bench.get_task_init_states(p["task_index"])[p["init"]])
            perturb_robot(env, p["sigma"], rng)
            for t in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            record = []
            main_success, main_end = play(env, task, obs, 10, [], [], record=record)
            n_pts = args.points_fail if not main_success else args.points_success
            # spread the branch points over the episode (skip the very last snapshot: little left to change)
            cand_pts = record[:-1] if len(record) > 1 else record
            if len(cand_pts) > n_pts:
                pick = np.linspace(0, len(cand_pts) - 1, n_pts).round().astype(int)
                pts = [cand_pts[i] for i in sorted(set(pick))]
            else:
                pts = cand_pts
            Z, C, G, CAND, BIDX, BSUC, BEND, TT = [], [], [], [], [], [], [], []
            for pt in pts:
                alts = rng.choice(np.arange(1, pt["cands"].shape[0]), size=args.alts, replace=False)
                idx = [0] + [int(a) for a in alts]
                suc, end = [], []
                for k in idx:
                    # reset first: an episode that ran to the horizon leaves robosuite's "done" flag set, and
                    # restoring a sim state alone does not clear it ("executing action in terminated episode")
                    env.reset()
                    obs_b = env.set_init_state(pt["sim"])
                    s, e = play(env, task, obs_b, pt["t"], pt["his_img"], pt["his_wrist"], first_chunk=pt["cands"][k])
                    suc.append(s)
                    end.append(e)
                Z.append(pt["z_t"]); C.append(pt["ctx"]); G.append(pt["goal"])
                CAND.append(pt["cands"].astype(np.float16)); BIDX.append(idx); BSUC.append(suc); BEND.append(end)
                TT.append(pt["t"])
            if Z:
                np.savez_compressed(os.path.join(args.out, f"ep{p['id']:06d}.npz"),
                                    z_t=np.stack(Z), ctx=np.stack(C), goal=np.stack(G), candidates=np.stack(CAND),
                                    branch_idx=np.array(BIDX), branch_success=np.array(BSUC), branch_end=np.array(BEND),
                                    t=np.array(TT), main_success=np.array(main_success), main_end=np.array(main_end),
                                    task_index=np.array(p["task_index"]), sigma=np.array(p["sigma"]))
            rec = {"id": p["id"], "task": p["task_index"], "sigma": round(p["sigma"], 3), "main_success": main_success,
                   "points": len(pts), "s1_branch_success": [bool(b[0]) for b in BSUC],
                   "alt_success": [int(sum(b[1:])) for b in BSUC], "seconds": round(time.time() - t0, 1), "error": None}
        except Exception as e:  # keep going; log the failure
            traceback.print_exc()
            rec = {"id": p["id"], "task": p["task_index"], "main_success": main_success,
                   "error": f"{type(e).__name__}: {e}",
                   "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(args.out, f"log_shard{args.shard}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
