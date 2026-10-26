#!/bin/bash
# Evaluate LIS-VLA on LIBERO.
#   bash eval_lis.sh goal 5 ../outputs/libero_goal/<exp>/epoch9 [N=16] [CEM=0] [SEARCH=1]
# SEARCH=0 runs System 1 only on the same checkpoint (the main ablation).
cd "$(dirname "$0")"
SUITE=${1:-goal}
H=${2:-5}
CKPT=$3
N=${4:-16}
CEM=${5:-0}
SEARCH=${6:-1}
export LIBGL_DRIVERS_PATH=${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri/}
OUT=../eval_outputs/libero_${SUITE}/lis/$(basename $(dirname $CKPT))_$(basename $CKPT)/search${SEARCH}_N${N}_cem${CEM}
mkdir -p $OUT
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$((29600 + RANDOM % 1000)) ../eval_solver_libero_lis.py \
  --device 0 --task_suite_name libero_${SUITE} --his his_2_third_view_wrist_w_state \
  --no_auto_resume --resume_path $CKPT \
  --tokenizer_path ../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af \
  --eval_only True --model_size 7B --batch_size 1 --accum_iter 1 --epochs 1 \
  --action_dim 7 --time_horizon $H --output_dir $OUT --max_seq_len 8192 --unmask_image_logits \
  --dropout 0.05 --z_loss_weight 1e-5 --ckpt_max_keep 0 \
  --lis_enable_search $SEARCH --lis_num_candidates $N --lis_cem_iters $CEM \
  2>&1 | tee -a $OUT/output.log
