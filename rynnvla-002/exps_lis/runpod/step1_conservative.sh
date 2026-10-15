#!/bin/bash
# Step 1 after the negative pilot: inference-only fixes, no retraining.
#   1. offline diagnostic of the System-2 scorer on held-out demonstrations
#   2. two conservative search settings, in parallel on one A40, on the SAME 100 episodes as the
#      pilot (Goal / Robot Initial States, --max_per_category 200, shard 0 of 2):
#        A: override only when the winner beats System 1 by the 90th-percentile score gap
#        B: 75th-percentile gap + trust region (prior_weight) + candidates closer to System 1
#   3. paired comparison with the Phase-0 baseline, summary file, then the pod stops itself
#
#   nohup bash step1_conservative.sh <lis_weights_dir> < /dev/null > /workspace/results/step1.out 2>&1 &
set -uo pipefail
LISW=${1:?lis weights dir}
W=/workspace
RV=$W/LIS-VLA/rynnvla-002
OUT=$W/results/step1
CK=$RV/ckpts/RynnVLA-002/VLA_model_256/libero_goal
TOK=$RV/ckpts/lumina_tokenizer
mkdir -p $OUT/logs /root/pd
cd $RV/exps_lis
source $RV/exps_lis/runpod/pod_env.sh plus >/dev/null

echo "[$(date +%H:%M)] 1/3 scorer diagnostic"
if [ ! -d /root/pd/tokens ]; then
  tar -C /root/pd -xf $W/data/lis_goal_H5.tar --wildcards 'tokens/libero_goal_his_2_val_ind*'
fi
ln -sfn /root/pd $RV/processed_data
python ../tools/diag_evaluator.py --ckpt $CK --lis_weights $LISW --n 300 \
  --record ../processed_data/tokens/libero_goal_his_2_val_ind_third_view_wrist_w_state_5_256_lis/record.json \
  --out $OUT/diag.json > $OUT/logs/diag.log 2>&1
cat $OUT/diag.json || { tail -20 $OUT/logs/diag.log; echo "diagnostic failed - continuing with fixed margins"; }
read P75 P90 < <(python -c "
import json
try:
    p = json.load(open('$OUT/diag.json'))['gap_z_percentiles(50,75,90,95)']; print(max(p[1], 0.5), max(p[2], 1.0))
except Exception:
    print(2.0, 3.0)")
echo "margins: B=$P75 A=$P90"

run() {  # name override_z prior_weight noise_scale
  python $RV/eval_lis_libero.py --ckpt $CK --lis_weights $LISW --suite libero_goal --time_horizon 5 \
    --tokenizer_path $TOK --benchmark plus --categories "Robot Initial States" --max_per_category 200 \
    --shard 0 --num_shards 2 --search 1 --override_z $2 --prior_weight $3 --noise_scale $4 \
    --out $OUT/$1 > $OUT/logs/$1.log 2>&1
  echo "[$(date +%H:%M)] $1 exit=$?"
}
echo "[$(date +%H:%M)] 2/3 evaluation (2 settings in parallel)"
run A_gate_p90 $P90 0.0 1.0 &
sleep 90
run B_gate_p75_trust $P75 0.5 0.5 &
wait

echo "[$(date +%H:%M)] 3/3 comparison"
{
  echo "margins: A override_z=$P90 | B override_z=$P75, prior_weight=0.5, noise_scale=0.5"
  for c in A_gate_p90 B_gate_p75_trust; do
    echo "== $c"
    python $RV/tools/compare_eval.py $W/results/phase0/goal_plus_s1 $OUT/$c
  done
  echo "== pilot (plain argmax, for reference)"
  python $RV/tools/compare_eval.py $W/results/phase0/goal_plus_s1 $W/results/pilot/goal_rob_search
} > $OUT/summary.txt 2>&1
cat $OUT/summary.txt
touch $OUT/DONE
sync; sleep 30
[ "${NO_STOP:-0}" = 1 ] || runpodctl stop pod $RUNPOD_POD_ID
