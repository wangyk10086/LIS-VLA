"""Pre-tokenise LIS-VLA conversations (output of ``lis_conv_generation.py``).

Re-uses ``pre_tokenize_action_state_local.py`` from RynnVLA-002 (the item processor already
handles state, action and image placeholders in any order), only file names differ.

    python pretoken_lis_model.py --task goal --resolution 256 --with_state \
        --img_names imgs_third_view imgs_wrist --his 2 --len_action 5 \
        --tokenizer_path ../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/<hash>
"""
import argparse
import os
from multiprocessing import Process


def run(rank, world, gpus, resolution, in_file, out_dir, with_state, tokenizer):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % gpus)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    script = "pre_tokenize_action_state_local.py" if with_state else "pre_tokenize_action_local.py"
    os.system(f"python -u {script} --splits={world} --rank={rank} --in_filename {in_file} "
              f"--out_dir {out_dir} --tokenizer {tokenizer} --target_size {resolution}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--his", type=int, default=2)
    ap.add_argument("--len_action", type=int, default=5)
    ap.add_argument("--with_state", action="store_true")
    ap.add_argument("--img_names", nargs="+", default=["imgs_third_view", "imgs_wrist"])
    ap.add_argument("--num_procs", type=int, default=32)
    ap.add_argument("--num_gpus", type=int, default=4)
    ap.add_argument("--conv_dir", default="../processed_data/convs")
    ap.add_argument("--out_root", default="../processed_data/tokens")
    args = ap.parse_args()

    img_item = "_".join(n.replace("imgs_", "") for n in args.img_names)
    state_item = "w_state" if args.with_state else "wo_state"
    for split in ("val_ind", "val_ood", "train"):
        name = f"libero_{args.task}_his_{args.his}_{split}_{img_item}_{state_item}_{args.len_action}_{args.resolution}_lis"
        in_file = os.path.join(args.conv_dir, name + ".json")
        out_dir = os.path.join(args.out_root, name)
        procs = [Process(target=run, args=(r, args.num_procs, args.num_gpus, args.resolution, in_file, out_dir,
                                           args.with_state, args.tokenizer_path)) for r in range(args.num_procs)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
