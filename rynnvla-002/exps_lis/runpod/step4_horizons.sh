#!/bin/bash
# Step 4: multi-step counterfactual branching (reviewer request). Re-runs the failed episodes of step 3
# and branches with interventions of horizon 1, 2, 4 and "until the end" (3 alternatives each).
#   nohup bash step4_horizons.sh <lis_weights_dir> [shard num_shards] < /dev/null > /workspace/results/step4.out 2>&1 &
# Writes /workspace/results/step4/summary_<shard>.txt, holds HOLD_HOURS, then stops the pod.
set -uo pipefail
LISW=${1:?lis weights dir}
SHARD=${2:-0}; NSH=${3:-1}
HOURS=${HOURS:-7}
HOLD_HOURS=${HOLD_HOURS:-1}
W=/workspace
RV=$W/LIS-VLA/rynnvla-002
OUT=$W/results/step4
DATA=$W/data/horizons_goal
CK=$RV/ckpts/RynnVLA-002/VLA_model_256/libero_goal
TOK=$RV/ckpts/lumina_tokenizer
mkdir -p $OUT/logs $DATA
cd $RV/exps_lis
source $RV/exps_lis/runpod/pod_env.sh vanilla >/dev/null
DEADLINE=$(python -c "import time; print(int(time.time() + float('$HOURS') * 3600))")
echo "[$(date +%H:%M)] horizons branching shard $SHARD/$NSH until $(date -d @$DEADLINE +%H:%M)"
# two streams per A40: split this pod's shard once more
for s in 0 1; do
  python $RV/tools/branch_horizons.py --ckpt $CK --lis_weights $LISW --tokenizer_path $TOK --out $DATA \
    --only_ids $RV/exps_lis/runpod/step4_failed_ids.txt --horizons 1,2,4,-1 --alts 3 --points 2 \
    --shard $((SHARD * 2 + s)) --num_shards $((NSH * 2)) --deadline $DEADLINE > $OUT/logs/h_$((SHARD * 2 + s)).log 2>&1 &
  sleep 90
done
wait
python $RV/tools/branch_horizons_summary.py $DATA --base_success 55/115 --json $OUT/summary.json > $OUT/summary_$SHARD.txt 2>&1
cat $OUT/summary_$SHARD.txt
touch $OUT/DONE_$SHARD
sleep $(python -c "print(int(float('$HOLD_HOURS') * 3600))")
[ "${NO_STOP:-0}" = 1 ] || runpodctl stop pod $RUNPOD_POD_ID
