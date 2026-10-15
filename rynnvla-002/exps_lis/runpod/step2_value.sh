#!/bin/bash
# Step 2: outcome-trained System-2 scorer, unattended on one A40 (~8 h, ~$5).
#   1. collect System-1 rollouts with exploration on LIBERO-Goal *training* tasks with our own robot
#      initial-joint perturbations (2 streams, wall-clock cap COLLECT_HOURS)
#   2. train two scorers on the cached features: imag = V(imagined future), direct = Q(state, action)
#   3. evaluate both, in parallel, on the same 100 LIBERO-Plus episodes as the pilot / step 1
#   4. paired comparison with the System-1 baseline -> /workspace/results/step2/summary.txt; pod stops
#
#   nohup bash step2_value.sh <lis_weights_dir> < /dev/null > /workspace/results/step2.out 2>&1 &
set -uo pipefail
LISW=${1:?lis weights dir}
COLLECT_HOURS=${COLLECT_HOURS:-4.5}
MARGIN=${MARGIN:-0.05}
W=/workspace
RV=$W/LIS-VLA/rynnvla-002
OUT=$W/results/step2
ROLL=$W/data/rollouts_goal
CK=$RV/ckpts/RynnVLA-002/VLA_model_256/libero_goal
TOK=$RV/ckpts/lumina_tokenizer
mkdir -p $OUT/logs $OUT/heads $ROLL
cd $RV/exps_lis

finish() {  # write the summary (whatever exists) and stop the pod
  {
    echo "### step 2 summary  $(date -u +%F' '%H:%M) UTC"
    echo "rollouts: $(ls $ROLL/ep*.npz 2>/dev/null | wc -l) episodes"
    cat $ROLL/log_shard*.jsonl 2>/dev/null | python -c "
import sys, json
r = [json.loads(l) for l in sys.stdin if l.strip()]
if r:
    import statistics as s
    print(f'  success rate {s.mean(x[\"success\"] for x in r):.3f}  (sigma=0: {s.mean([x[\"success\"] for x in r if x[\"sigma\"]==0] or [0]):.3f})'
          f'  explored queries {sum(x[\"explored\"] for x in r)} / {sum(x[\"queries\"] for x in r)}  errors {sum(1 for x in r if x[\"error\"])}')
"
    for m in imag direct; do echo "== scorer $m (held-out rollouts)"; cat $OUT/heads/value_$m.pt.json 2>/dev/null || echo "  (not trained)"; done
    for m in imag direct; do
      echo "== LIBERO-Plus Goal / Robot Initial States, scorer $m, margin $MARGIN"
      [ -d $OUT/eval_$m ] && python $RV/tools/compare_eval.py $W/results/phase0/goal_plus_s1 $OUT/eval_$m || echo "  (not run)"
    done
    echo "== reference: step-1 language-similarity scorer, gated (A)"
    python $RV/tools/compare_eval.py $W/results/phase0/goal_plus_s1 $W/results/step1/A_gate_p90 2>/dev/null | head -4
    echo "== errors (tails)"; for f in $OUT/logs/*.log; do echo "-- $f"; grep -iE "error|traceback" $f | tail -3; done
  } > $OUT/summary.txt 2>&1
  cat $OUT/summary.txt
  touch $OUT/DONE
  sync; sleep 30
  [ "${NO_STOP:-0}" = 1 ] || runpodctl stop pod $RUNPOD_POD_ID
  exit 0
}

echo "[$(date +%H:%M)] 1/4 rollouts (cap ${COLLECT_HOURS} h)"
source $RV/exps_lis/runpod/pod_env.sh vanilla >/dev/null
DEADLINE=$(python -c "import time; print(int(time.time() + float('$COLLECT_HOURS') * 3600))")
for s in 0 1; do
  python $RV/tools/collect_rollouts.py --ckpt $CK --lis_weights $LISW --tokenizer_path $TOK --out $ROLL \
    --episodes 1200 --shard $s --num_shards 2 --deadline $DEADLINE > $OUT/logs/collect_$s.log 2>&1 &
  sleep 90
done
wait
N=$(ls $ROLL/ep*.npz 2>/dev/null | wc -l)
echo "[$(date +%H:%M)] collected $N episodes"
if [ "$N" -lt 100 ]; then echo "too few rollouts - stopping"; tail -30 $OUT/logs/collect_0.log; finish; fi

echo "[$(date +%H:%M)] 2/4 train scorers"
for m in imag direct; do
  python $RV/tools/train_value.py --rollouts $ROLL --mode $m --ckpt $CK --lis_weights $LISW \
    --out $OUT/heads/value_$m.pt > $OUT/logs/train_$m.log 2>&1 || echo "train $m failed"
  tail -12 $OUT/logs/train_$m.log
done

echo "[$(date +%H:%M)] 3/4 evaluate"
source $RV/exps_lis/runpod/pod_env.sh plus >/dev/null
for m in imag direct; do
  [ -f $OUT/heads/value_$m.pt ] || continue
  python $RV/eval_lis_libero.py --ckpt $CK --lis_weights $LISW --value_head $OUT/heads/value_$m.pt \
    --margin_p $MARGIN --suite libero_goal --time_horizon 5 --tokenizer_path $TOK --benchmark plus \
    --categories "Robot Initial States" --max_per_category 200 --shard 0 --num_shards 2 --search 1 \
    --out $OUT/eval_$m > $OUT/logs/eval_$m.log 2>&1 &
  sleep 90
done
wait
echo "[$(date +%H:%M)] 4/4 summary"
finish
