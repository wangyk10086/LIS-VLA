#!/bin/bash
# Pilot: LIS search vs the Phase-0 System-1 baseline on the same LIBERO-Plus episodes.
#
#   bash pilot_eval.sh <lis_weights_dir> [N_PER_CATEGORY=200] [SHARD NUM_SHARDS]
#
# Runs Goal / "Robot Initial States" with search on; compare with
#   python tools/compare_eval.py /workspace/results/phase0/goal_plus_s1 /workspace/results/pilot/goal_rob_search
set -uo pipefail
LISW=${1:?lis weights dir}
N=${2:-200}
SHARD=${3:-0}
NUM=${4:-1}
RV=/workspace/LIS-VLA/rynnvla-002
OUT=/workspace/results/pilot
mkdir -p $OUT/logs
cd $RV/exps_lis
source $RV/exps_lis/runpod/pod_env.sh plus >/dev/null
python $RV/tools/preflight.py --ckpt $RV/ckpts/RynnVLA-002/VLA_model_256/libero_goal --lis_weights $LISW \
  --tokenizer_path $RV/ckpts/lumina_tokenizer | tee $OUT/logs/preflight_shard$SHARD.log
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "preflight failed"; exit 1; }
python $RV/eval_lis_libero.py --ckpt $RV/ckpts/RynnVLA-002/VLA_model_256/libero_goal --lis_weights $LISW \
  --suite libero_goal --time_horizon 5 --tokenizer_path $RV/ckpts/lumina_tokenizer \
  --benchmark plus --categories "Robot Initial States" --max_per_category $N \
  --search 1 --shard $SHARD --num_shards $NUM --out $OUT/goal_rob_search >> $OUT/logs/goal_rob_search_shard$SHARD.log 2>&1
echo "pilot shard $SHARD exit=$?"
touch $OUT/logs/DONE_shard$SHARD
