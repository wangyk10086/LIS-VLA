#!/bin/bash
# Run on the step-1 pod: take over from step1_conservative.sh (so it does not stop the pod), wait for
# its evaluations, write the step-1 summary, then run step 2 on the same pod.
#   bash chain_step2.sh <lis_weights_dir>
LISW=${1:?lis weights dir}
W=/workspace
RV=$W/LIS-VLA/rynnvla-002
pkill -f step1_conservative.sh   # its python children keep running
nohup bash -c "
  while pgrep -f '[e]val_lis_libero.py' > /dev/null; do sleep 30; done
  source $RV/exps_lis/runpod/pod_env.sh plus > /dev/null
  { for c in A_gate_p90 B_gate_p75_trust; do echo == \$c; python $RV/tools/compare_eval.py $W/results/phase0/goal_plus_s1 $W/results/step1/\$c; done
    echo == pilot; python $RV/tools/compare_eval.py $W/results/phase0/goal_plus_s1 $W/results/pilot/goal_rob_search; } > $W/results/step1/summary.txt 2>&1
  touch $W/results/step1/DONE
  bash $RV/exps_lis/runpod/step2_value.sh $LISW
" < /dev/null > $W/results/step2.out 2>&1 &
sleep 2
echo "chained; step-1 evals still running: $(pgrep -fc '[e]val_lis_libero.py')"
