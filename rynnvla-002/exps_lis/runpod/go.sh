#!/bin/bash
# Short entry point for the RunPod web terminal (long typed commands are unreliable there):
#   bash go.sh prep  [suite H]            -> prep_data.sh in the background
#   bash go.sh tok   [suite H]            -> restart prep at tokenisation (steps 1-3 done)
#   bash go.sh train [suite H max_hours]  -> train_runpod.sh in the background
#   bash go.sh pilot <lis_weights> [N]    -> pilot_eval.sh, two streams on one GPU
#   bash go.sh step3 [hours hold_hours]   -> step3_branch.sh with the latest LIS checkpoint
#   bash go.sh status                     -> tails of all logs
R=/workspace/results
S=/workspace/LIS-VLA/rynnvla-002/exps_lis/runpod
mkdir -p $R
case "${1:-status}" in
  prep)  nohup bash $S/prep_data.sh ${2:-goal} ${3:-5} < /dev/null > $R/prep_${2:-goal}.out 2>&1 & ;;
  tok)   pkill -f prep_data.sh; pkill -f pretoken_lis_model; pkill -f pre_tokenize_action_state_local; sleep 5
         nohup env START_STEP=4 bash $S/prep_data.sh ${2:-goal} ${3:-5} < /dev/null >> $R/prep_${2:-goal}.out 2>&1 & ;;
  train) nohup env MAX_HOURS=${4:-5} bash $S/train_runpod.sh ${2:-goal} ${3:-5} < /dev/null > $R/train_${2:-goal}.out 2>&1 & ;;
  pilot) # the whole launcher runs detached: closing the web terminal must not stop the staggered start
         nohup bash -c "for s in 0 1; do bash $S/pilot_eval.sh $2 ${3:-200} \$s 2 < /dev/null > $R/pilot_shard\$s.out 2>&1 & sleep 120; done; wait" \
           < /dev/null > $R/pilot_launcher.out 2>&1 & ;;
  step3) LISW=$(ls -d /workspace/LIS-VLA/rynnvla-002/outputs/libero_goal/lis_lis_only*/epoch* | sort -V | tail -1)
         HOURS=${2:-4} HOLD_HOURS=${3:-1} nohup bash $S/step3_branch.sh $LISW < /dev/null > $R/step3.out 2>&1 & ;;
  status) ;;
  *) echo "unknown command $1"; exit 1 ;;
esac
sleep 3
for f in $R/*.out; do echo "== $f"; grep -v "Skipping no-op" $f | tail -n 4 | cut -c1-200; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
df -h / | tail -1
