"""Generate LIS-VLA conversations from LIBERO trajectories.

Each sample couples the VLA prompt with the *future observation reached after executing
the action chunk*, so a single sequence supervises all LIS components:

    human: "What action should the robot take to {task}?" <|state|> <|image|> x (his * V)
    gpt  : <|action|> x H  <|image|> x V        # images at t + H  (k = H)

  * action head + flow-matching proposer   <- the H action blocks
  * latent world model                     <- (z_t, A_{t:t+H})  ->  z_{t+H}
  * language-guided evaluator               <- (z_{t+H}, instruction)
  * the backbone additionally learns to generate the frame at t+H (action-world-model CE)

Input layout is the one produced by ``libero_util/regenerate_libero_dataset_save_img_action_state_wrist.py``:
    <base_dir>/<task_name>/<trj_k>/{action, imgs_third_view, imgs_wrist, eef_gripper_state}/*.npy|png

Usage:
    python lis_conv_generation.py --base_dir ../processed_data/libero_goal_image_state_action_t_256 \
        --his 2 --len_action 5 --task_name goal --resolution 256 --with_state \
        --img_names imgs_third_view imgs_wrist --output_dir ../processed_data/convs
"""
import argparse
import json
import math
import os


def _indexed(folder, prefix, suffix):
    out = {}
    for f in os.listdir(folder):
        if f.startswith(prefix) and f.endswith(suffix):
            out[int(f[len(prefix): -len(suffix)])] = os.path.join(folder, f)
    return out


def collect_trajectory(trj_path, img_names, with_state):
    action = _indexed(os.path.join(trj_path, "action"), "action_", ".npy")
    imgs = [_indexed(os.path.join(trj_path, n), "image_", ".png") for n in img_names]
    keys = set(action)
    for d in imgs:
        keys &= set(d)
    state = None
    if with_state:
        state = _indexed(os.path.join(trj_path, "eef_gripper_state"), "eef_gripper_state_", ".npy")
        keys &= set(state)
    idx = sorted(keys)
    return (
        [action[i] for i in idx],
        [[d[i] for d in imgs] for i in idx],          # per frame: one path per view
        [state[i] for i in idx] if with_state else None,
    )


def build_convs(task_readable, actions, frames, states, his, H, with_state):
    convs = []
    T = len(actions)
    for j in range(T - H):  # need frame j + H to exist
        hist = [p for f in frames[max(0, j - his + 1): j + 1] for p in f]
        future = list(frames[j + H])
        human = f"What action should the robot take to {task_readable}?"
        if with_state:
            human += "<|state|>"
        human += "<|image|>" * len(hist)
        conv = {
            "conversations": [
                {"from": "human", "value": human},
                {"from": "gpt", "value": "<|action|>" * H + "<|image|>" * len(future)},
            ],
            "image": hist + future,
            "action": actions[j: j + H],
        }
        if with_state:
            conv["state"] = [states[j]]
        convs.append(conv)
    return convs


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--his", type=int, default=2)
    ap.add_argument("--len_action", type=int, default=5, help="H, also the imagination horizon k")
    ap.add_argument("--task_name", required=True, help="goal / object / spatial / 10")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--with_state", action="store_true")
    ap.add_argument("--img_names", nargs="+", default=["imgs_third_view", "imgs_wrist"])
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    splits = {"train": [], "val_ind": [], "val_ood": []}
    tasks = sorted(os.listdir(args.base_dir))
    split_ood = math.ceil(len(tasks) * 0.9)  # same split rule as the original generators
    for t_id, task in enumerate(tasks):
        task_dir = os.path.join(args.base_dir, task)
        trjs = sorted(os.listdir(task_dir))
        split_ind = math.ceil(len(trjs) * 0.9)
        for k, trj in enumerate(trjs):
            try:
                a, f, s = collect_trajectory(os.path.join(task_dir, trj), args.img_names, args.with_state)
            except FileNotFoundError as e:
                print(f"skip {task}/{trj}: {e}")
                continue
            convs = build_convs(task.replace("_", " "), a, f, s, args.his, args.len_action, args.with_state)
            if t_id < split_ood:
                splits["train" if k < split_ind else "val_ind"] += convs
            else:
                splits["val_ood"] += convs

    # same naming scheme as action_state_model_conv_generation.py, with a "_lis" suffix
    img_item = "_".join(n.replace("imgs_", "") for n in args.img_names)
    state_item = "w_state" if args.with_state else "wo_state"
    name = f"libero_{args.task_name}_his_{args.his}_{{}}_{img_item}_{state_item}_{args.len_action}_{args.resolution}_lis"
    for split, convs in splits.items():
        path = os.path.join(args.output_dir, name.format(split) + ".json")
        with open(path, "w") as fp:
            json.dump(convs, fp)
        print(f"{split}: {len(convs)} samples -> {path}")


if __name__ == "__main__":
    main()
