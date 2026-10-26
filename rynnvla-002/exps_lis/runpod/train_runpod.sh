#!/bin/bash
# Train LIS-VLA System 2 (frozen RynnVLA-002) on one GPU, with a hard wall-clock cap.
#
#   MAX_HOURS=5 bash train_runpod.sh goal 5
#
# Needs /workspace/data/lis_<suite>_H<H>.tar from prep_data.sh. Checkpoints contain only lis.*
# (~0.2 GB) and are written every SAVE_EVERY steps, so a run stopped by the cap is still usable:
#   ls /workspace/LIS-VLA/rynnvla-002/outputs/libero_<suite>/lis_only_*/
set -uo pipefail
SUITE=${1:-goal}
H=${2:-5}
MAX_HOURS=${MAX_HOURS:-5}
W=/workspace
RV=$W/LIS-VLA/rynnvla-002
LOCAL=${LOCAL_DATA:-/root/pd}
TAR=$W/data/lis_${SUITE}_H${H}.tar

source $RV/exps_lis/runpod/pod_env.sh vanilla
[ -f $TAR ] || { echo "ERROR: $TAR not found - run prep_data.sh first"; exit 1; }
if [ ! -f $LOCAL/.unpacked_${SUITE}_H${H} ]; then
  echo "[$(date +%H:%M)] unpacking training data to local disk"
  mkdir -p $LOCAL && tar -C $LOCAL -xf $TAR && touch $LOCAL/.unpacked_${SUITE}_H${H}
fi
if [ -e $RV/processed_data ] && [ ! -L $RV/processed_data ]; then
  echo "ERROR: $RV/processed_data exists and is not a symlink"; exit 1
fi
ln -sfn $LOCAL $RV/processed_data

echo "[$(date +%H:%M)] training (cap ${MAX_HOURS} h) on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
# own process group, so the cap stops torchrun and every worker
setsid bash $RV/exps_lis/train_lis.sh $SUITE $H lis_only < /dev/null &
PG=$!
deadline=$(( $(date +%s) + $(python -c "print(int(float('$MAX_HOURS') * 3600))") ))
while kill -0 $PG 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 30; done
if kill -0 $PG 2>/dev/null; then
  echo "[$(date +%H:%M)] ${MAX_HOURS} h cap reached - stopping (latest checkpoint is kept)"
  kill -INT -- -$PG 2>/dev/null; sleep 60; kill -KILL -- -$PG 2>/dev/null
  rc=124
else
  wait $PG; rc=$?
fi
ls -d $RV/outputs/libero_${SUITE}/lis_only_*/epoch* 2>/dev/null
echo "[$(date +%H:%M)] train_runpod.sh exit=$rc"
