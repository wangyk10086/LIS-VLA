#!/bin/bash
# Phase 0 (no training): RynnVLA-002 System 1 on standard LIBERO (sanity) and LIBERO-Plus
# (Robot Initial States + Objects Layout), for Goal and Long. Run one shard per pod:
#
#   bash phase0_shard.sh SHARD NUM_SHARDS          # e.g. 0 6 ... 5 6
#   SMOKE=1 bash phase0_shard.sh 0 1               # 2 episodes per job, to test a fresh pod
#
# Results: /workspace/results/phase0/<job>/results_shard*.jsonl (resumable: just re-run).
set -uo pipefail
SHARD=${1:?shard}
NUM=${2:?num_shards}
SMOKE=${SMOKE:-0}
RV=/workspace/LIS-VLA/rynnvla-002
OUT=/workspace/results/phase0
TOK=$RV/ckpts/lumina_tokenizer
CK=$RV/ckpts/RynnVLA-002/VLA_model_256
LIMIT=""; [ "$SMOKE" = 1 ] && LIMIT="--limit 2" && OUT=/workspace/results/phase0_smoke
mkdir -p $OUT/logs
cd $RV/exps_lis   # RynnVLA-002 loads tokenizers via "../ckpts/..." relative to the CWD

run() {  # name kind suite H extra-args...
  local name=$1 kind=$2 suite=$3 H=$4; shift 4
  source $RV/exps_lis/runpod/pod_env.sh $kind >/dev/null
  echo "[$(date +%H:%M)] job $name (shard $SHARD/$NUM)"
  python $RV/eval_lis_libero.py --ckpt $CK/$suite --suite $suite --time_horizon $H --tokenizer_path $TOK \
    --search 0 --shard $SHARD --num_shards $NUM --out $OUT/$name $LIMIT "$@" \
    >> $OUT/logs/${name}_shard${SHARD}.log 2>&1
  echo "[$(date +%H:%M)] job $name exit=$?"
}

source $RV/exps_lis/runpod/pod_env.sh plus
python $RV/tools/preflight.py --ckpt $CK/libero_goal --tokenizer_path $TOK | tee $OUT/logs/preflight_shard${SHARD}.log
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "preflight failed - stopping"; exit 1; }

CATS=("Robot Initial States" "Objects Layout")
run goal_plus_s1    plus    libero_goal 5  --benchmark plus --categories "${CATS[@]}"
run long_plus_s1    plus    libero_10   10 --benchmark plus --categories "${CATS[@]}"
run goal_vanilla_s1 vanilla libero_goal 5  --benchmark vanilla --episodes_per_task 10
run long_vanilla_s1 vanilla libero_10   10 --benchmark vanilla --episodes_per_task 10
echo "[$(date +%H:%M)] shard $SHARD done"
touch $OUT/logs/DONE_shard${SHARD}
