#!/bin/bash
# Build LIS-VLA training data for one LIBERO suite on a single-GPU pod, then pack it onto the volume.
#
#   bash prep_data.sh goal 5        # suite (goal|10|object|spatial), action chunk H
#
# Steps: download raw LIBERO demos -> replay them without no-op actions at 256 px (one process per
# task) -> dump images/actions/states -> LIS conversations -> VQ tokens -> one tar on the volume:
#   /workspace/data/lis_<suite>_H<H>.tar   (unpacked by train_runpod.sh onto the training pod)
# The many small intermediate files stay on the pod's local disk (use a container disk >= 60 GB).
set -euo pipefail
SUITE=${1:-goal}
H=${2:-5}
W=/workspace
RV=$W/LIS-VLA/rynnvla-002
LOCAL=${LOCAL_DATA:-/root/pd}
LOG=$W/results/prep_${SUITE}
OUT_TAR=$W/data/lis_${SUITE}_H${H}.tar
mkdir -p $LOCAL $LOG $W/data/libero_raw

# training demos come from standard LIBERO
source $RV/exps_lis/runpod/pod_env.sh vanilla

if [ -e $RV/processed_data ] && [ ! -L $RV/processed_data ]; then
  echo "ERROR: $RV/processed_data exists and is not a symlink - refusing to touch it"; exit 1
fi
ln -sfn $LOCAL $RV/processed_data

echo "[$(date +%H:%M)] 1/5 download raw demos"
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
                  allow_patterns=["libero_${SUITE}/*"], local_dir="$W/data/libero_raw")
EOF
RAW=$W/data/libero_raw/libero_${SUITE}
ls $RAW | head -3

echo "[$(date +%H:%M)] 2/5 replay demos without no-ops (parallel, one process per task)"
NOOP=$LOCAL/libero_${SUITE}_no_noops_t_256
mkdir -p $NOOP
cd $RV/libero_util
NT=$(python -c "from libero.libero import benchmark; print(benchmark.get_benchmark_dict()['libero_${SUITE}']().n_tasks)" | tail -1)   # LIBERO also prints log lines
pids=()
for t in $(seq 0 $((NT - 1))); do
  python regenerate_libero_dataset_filter_no_op.py --libero_task_suite libero_${SUITE} --libero_raw_data_dir $RAW \
    --libero_target_dir $NOOP --image_resolution 256 --task_ids $t --yes > $LOG/noop_task$t.log 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || { echo "ERROR: a replay process failed, see $LOG/noop_task*.log"; exit 1; }
python - <<EOF
import glob, h5py
n = 0
for f in sorted(glob.glob("$NOOP/*.hdf5")):
    k = len(h5py.File(f, "r")["data"]); n += k
    print(f"{k:3d} demos  {f.split('/')[-1]}")
print("total successful replays:", n)
assert n > 0
EOF

echo "[$(date +%H:%M)] 3/5 dump images, actions and states"
IMG=$LOCAL/libero_${SUITE}_image_state_action_t_256
python regenerate_libero_dataset_save_img_action_state_wrist.py --libero_task_suite libero_${SUITE} \
  --image_resolution 256 --raw_data_dir $NOOP --save_dir $IMG > $LOG/dump.log 2>&1
rm -rf $NOOP   # 15-20 GB of uncompressed frames, no longer needed
echo "frames: $(find $IMG -name 'image_*.png' -path '*third_view*' | wc -l)"

echo "[$(date +%H:%M)] 4/5 LIS conversations + VQ tokens"
LIS_ONLY_DATA=1 bash $RV/exps_lis/prepare_data.sh $SUITE $H > $LOG/prepare.log 2>&1 || {
  tail -30 $LOG/prepare.log; exit 1; }
grep -E "samples ->|find [0-9]+ sub-records" $LOG/prepare.log || true
CONCAT=$LOCAL/concate_tokens/libero_${SUITE}_his_2_third_view_wrist_w_state_${H}_256_lis_awm.json
python -c "import json; d=json.load(open('$CONCAT')); print('training records:', len(d)); assert len(d) > 1000"

echo "[$(date +%H:%M)] 5/5 pack to $OUT_TAR"
tar -C $LOCAL -cf $OUT_TAR.tmp tokens concate_tokens && mv $OUT_TAR.tmp $OUT_TAR
ls -la $OUT_TAR
touch $W/data/lis_${SUITE}_H${H}.DONE
echo "[$(date +%H:%M)] done"
