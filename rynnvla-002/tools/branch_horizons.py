"""Multi-step counterfactual branching: does a failure become recoverable if alternative chunks keep
control for several consecutive decision points, instead of one?

Same scenes as tools/branch_rollouts.py (same episode plan: seed 4321, ids from --seed_offset), so
the failed episodes of that run can be targeted with --only_ids. For every failed episode System 1 is
replayed with snapshots; at up to --points branch points the simulator is restored and we play
  * the System-1 branch (horizon 0), and
  * for each horizon H in --horizons: --alts branches in which, at each of the next H decision points,
    a random non-System-1 candidate of the proposer is executed; System 1 then takes over.
    H = -1 means the proposer's random candidates keep control until the episode ends.
The H = 1 branches reproduce the single-chunk protocol of branch_rollouts.py.

    python tools/branch_horizons.py --ckpt ... --lis_weights ... --tokenizer_path ... --out DIR \\
        --only_ids ids.txt --horizons 1,2,4,-1 --alts 3 --points 2
Summarise with tools/branch_horizons_summary.py.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def episode_plan(n_tasks, init_counts, episodes, seed_offset, sigma_max):
    """Identical to the plan in branch_rollouts.py (same RNG calls in the same order)."""
    plan, rng0 = [], np.random.RandomState(4321)
    for i in range(episodes):
        ti = i % n_tasks
        plan.append({"id": seed_offset + i, "task_index": ti, "init": int(rng0.randint(init_counts[ti])),
                     "sigma": float(rng0.uniform(0.05, sigma_max)), "seed": int(rng0.randint(2 ** 31))})
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lis_weights", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--time_horizon", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed_offset", type=int, default=100000)
    ap.add_argument("--only_ids", default=None, help="file with one episode id per line (e.g. failed episodes)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--every", type=int, default=4, help="snapshot every N-th query")
    ap.add_argument("--points", type=int, default=2, help="branch points per failed episode")
    ap.add_argument("--horizons", default="1,2,4,-1")
    ap.add_argument("--alts", type=int, default=3, help="alternative branches per horizon and point")
    ap.add_argument("--num_candidates", type=int, default=16)
    ap.add_argument("--sigma_max", type=float, default=0.3)
    ap.add_argument("--deadline", type=float, default=0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--his", default="his_2_third_view_wrist_w_state")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)
    horizons = [int(h) for h in args.horizons.split(",")]
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
    plan = episode_plan(bench.n_tasks, [len(bench.get_task_init_states(t)) for t in range(bench.n_tasks)],
                        args.episodes, args.seed_offset, args.sigma_max)
    if args.only_ids:
        keep = {int(l) for l in open(args.only_ids) if l.strip()}
        plan = [p for p in plan if p["id"] in keep]
    mine = [p for k, p in enumerate(plan) if k % args.num_shards == args.shard
            and not os.path.exists(os.path.join(args.out, f"ep{p['id']:06d}.npz"))]
    print(f"{len(mine)} episodes to run in shard {args.shard}, horizons {horizons}", flush=True)

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

    def play(env, task, obs, t, his_img, his_wrist, first_chunk=None, alt_queries=0, rng=None, record=None):
        """Run from (obs, t) to the end with the same per-step history as eval_lis_libero.py.
        first_chunk: executed at the first decision point (counts as one alternative decision).
        alt_queries: number of decision points (including the first) at which a random non-System-1
        proposer candidate is executed; -1 = all of them. Returns (success, end_step, n_alt_executed)."""
        chunk, n_q, done, n_alt = [], 0, False, 0
        his_img, his_wrist = list(his_img), list(his_wrist)
        while t < max_steps + 10:
            img = get_libero_image(obs, args.resolution)
            wrist = get_libero_image(obs, args.resolution, "robot0_eye_in_hand_image")
            state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"]))
            ci, cw = Image.fromarray(img), Image.fromarray(wrist)
            if not chunk:
                alt_now = alt_queries < 0 or n_q < alt_queries
                if first_chunk is not None and n_q == 0:
                    chunk = [np.asarray(a, dtype=np.float32) for a in first_chunk]
                    n_alt += 1
                else:
                    snap = record is not None and n_q % args.every == 0
                    search = snap or alt_now
                    s1 = query(task, ci, cw, state, his_img, his_wrist, search=search)
                    info = model.last_search_info or {}
                    if search and info.get("search"):
                        cands = info["candidates"].float().cpu().numpy()
                        if snap:
                            record.append({"t": t, "sim": sim_state(env), "his_img": list(his_img),
                                           "his_wrist": list(his_wrist), "cands": cands})
                        if alt_now:
                            s1 = torch.from_numpy(cands[int(rng.randint(1, cands.shape[0]))])
                            n_alt += 1
                        else:
                            s1 = info["prior"]
                    chunk = [x for x in s1.float().cpu().numpy()]
                n_q += 1
            act = chunk.pop(0)
            obs, _, done, _ = env.step(unnorm_action(act).tolist())
            his_img.append(ci)
            his_wrist.append(cw)
            if done:
                break
            t += 1
        return bool(done), t, n_alt

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
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            record = []
            main_success, main_end, _ = play(env, task, obs, 10, [], [], record=record)
            pts = []
            if not main_success:
                cand_pts = record[:-1] if len(record) > 1 else record
                if len(cand_pts) > args.points:
                    pick = np.linspace(0, len(cand_pts) - 1, args.points).round().astype(int)
                    pts = [cand_pts[i] for i in sorted(set(pick))]
                else:
                    pts = cand_pts
            # results[point][horizon] = list of successes; horizon 0 = System-1 replay
            S1, RES, END, NALT, TT = [], [], [], [], []
            for pt in pts:
                def restore():
                    env.reset()  # clears robosuite's "done" flag left by a horizon-terminated episode
                    return env.set_init_state(pt["sim"])
                s, e, _ = play(env, task, restore(), pt["t"], pt["his_img"], pt["his_wrist"],
                               first_chunk=pt["cands"][0])
                S1.append(s)
                res_h, end_h, nalt_h = [], [], []
                for h in horizons:
                    alts = rng.choice(np.arange(1, pt["cands"].shape[0]), size=args.alts, replace=False)
                    rs, es, ns = [], [], []
                    for a in alts:
                        brng = np.random.RandomState(int(rng.randint(2 ** 31)))
                        s, e, n = play(env, task, restore(), pt["t"], pt["his_img"], pt["his_wrist"],
                                       first_chunk=pt["cands"][int(a)], alt_queries=h, rng=brng)
                        rs.append(s); es.append(e); ns.append(n)
                    res_h.append(rs); end_h.append(es); nalt_h.append(ns)
                RES.append(res_h); END.append(end_h); NALT.append(nalt_h); TT.append(pt["t"])
            np.savez_compressed(os.path.join(args.out, f"ep{p['id']:06d}.npz"),
                                horizons=np.array(horizons), s1_success=np.array(S1, dtype=bool),
                                alt_success=np.array(RES, dtype=bool).reshape(len(pts), len(horizons), args.alts),
                                alt_end=np.array(END).reshape(len(pts), len(horizons), args.alts),
                                n_alt=np.array(NALT).reshape(len(pts), len(horizons), args.alts),
                                t=np.array(TT), main_success=np.array(main_success), main_end=np.array(main_end),
                                task_index=np.array(p["task_index"]), sigma=np.array(p["sigma"]))
            rec = {"id": p["id"], "task": p["task_index"], "main_success": main_success, "points": len(pts),
                   "s1": [bool(x) for x in S1],
                   "alt_by_h": {str(h): [int(sum(RES[i][j])) for i in range(len(pts))] for j, h in enumerate(horizons)},
                   "seconds": round(time.time() - t0, 1), "error": None}
        except Exception as e:  # keep going; log the failure
            traceback.print_exc()
            rec = {"id": p["id"], "task": p["task_index"], "main_success": main_success,
                   "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 1)}
        with open(os.path.join(args.out, f"log_shard{args.shard}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
