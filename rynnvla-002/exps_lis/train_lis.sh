#!/bin/bash
# Train LIS-VLA on one LIBERO suite.
#   bash train_lis.sh goal 5 lis_only ../ckpts/RynnVLA-002/VLA_model_256/libero_goal
#
# SCOPE: lis_only      - freeze the RynnVLA-002 checkpoint, train System 2 only (cheapest)
#        lis_and_head  - also fine-tune the continuous action head
#        all           - full fine-tuning (as in RynnVLA-002 training)
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")"
SUITE=${1:-goal}
H=${2:-5}
SCOPE=${3:-lis_only}
INIT_FROM=${4:-../ckpts/RynnVLA-002/VLA_model_256/libero_${SUITE}}
NPROC=${NPROC_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-16667}

if [ "$SCOPE" = "all" ]; then lr=5e-6; epochs=40; else lr=1e-4; epochs=10; fi
DATA=../configs/lis/libero_${SUITE}_lis.yaml
EXP=lis_${SCOPE}_his_2_third_view_wrist_w_state_${H}_256
OUT=../outputs/libero_${SUITE}/$EXP
mkdir -p $OUT

torchrun --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$NPROC --nnodes=$NNODES --node_rank=$RANK \
  ../pretrain_solver_lis_vla.py \
  --train_only True --disable_length_clustering \
  --init_from $INIT_FROM \
  --tokenizer_path ../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af \
  --ablation 0 --model_size 7B --batch_size 8 --accum_iter 1 --epochs $epochs --warmup_epochs 0.01 \
  --lr $lr --min_lr $lr --wd 0.1 --clip_grad 4 \
  --action_dim 7 --time_horizon $H \
  --data_config_train $DATA --data_config_val_ind $DATA --data_config_val_ood $DATA \
  --num_workers 8 --output_dir $OUT --checkpointing --max_seq_len 4096 --unmask_image_logits \
  --dropout 0.08 --z_loss_weight 1e-5 --ckpt_max_keep 0 \
  --lis_train_scope $SCOPE --lis_num_views 2 --lis_dim 512 --lis_num_latents 16 \
  --lis_fm_weight 1.0 --lis_wm_weight 1.0 --lis_wm_nce_weight 0.1 --lis_var_weight 0.1 \
  --lis_lang_nce_weight 0.1 --lis_prog_weight 0.05 --lis_rank_weight 0.1 \
  2>&1 | tee -a $OUT/output.log
