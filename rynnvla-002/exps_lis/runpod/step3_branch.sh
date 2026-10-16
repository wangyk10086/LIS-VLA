#!/bin/bash
# Step 3: head-room test by counterfactual branching (unattended on one A40, ~5 h, ~$3).
#   nohup bash step3_branch.sh <lis_weights_dir> < /dev/null > /workspace/results/step3.out 2>&1 &
# Writes /workspace/results/step3/summary.txt, keeps the pod up HOLD_HOURS for reading, then stops it.
set -uo pipefail
LISW=${1:?lis weights dir}
HOURS=${HOURS:-5}
HOLD_HOURS=${HOLD_HOURS:-2}
W=/workspace
RV=$W/LIS-VLA/rynnvla-002
OUT=$W/results/step3
DATA=$W/data/branches_goal
CK=$RV/ckpts/RynnVLA-002/VLA_model_256/libero_goal
TOK=$RV/ckpts/lumina_tokenizer
mkdir -p $OUT/logs $DATA
cd $RV/exps_lis
source $RV/exps_lis/runpod/pod_env.sh vanilla >/dev/null

DEADLINE=$(python -c "import time; print(int(time.time() + float('$HOURS') * 3600))")
echo "[$(date +%H:%M)] branching until $(date -d @$DEADLINE +%H:%M)"
for s in 0 1; do
  python $RV/tools/branch_rollouts.py --ckpt $CK --lis_weights $LISW --tokenizer_path $TOK --out $DATA \
    --episodes 400 --shard $s --num_shards 2 --deadline $DEADLINE > $OUT/logs/branch_$s.log 2>&1 &
  sleep 90
done
wait
echo "[$(date +%H:%M)] summary"
{
  echo "### step 3 (counterfactual branching) $(date -u +%F' '%H:%M) UTC"
  python $RV/tools/branch_summary.py $DATA --json $OUT/summary.json
  echo "== error tails"; for f in $OUT/logs/*.log; do echo "-- $f"; grep -iE "error|traceback" $f | tail -3; done
} > $OUT/summary.txt 2>&1
cat $OUT/summary.txt
touch $OUT/DONE
sleep $(python -c "print(int(float('$HOLD_HOURS') * 3600))")
[ "${NO_STOP:-0}" = 1 ] || runpodctl stop pod $RUNPOD_POD_ID
