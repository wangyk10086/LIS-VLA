#!/bin/bash
# Phase 0 on ONE 48 GB GPU (e.g. A40): two shard streams share the GPU (~16 GB each), then the
# results are summarised and the pod stops itself (falls back to terminate for network-volume pods;
# everything is on /workspace).
#
#   nohup bash phase0_single_pod.sh < /dev/null > /workspace/results/orchestrate.out 2>&1 &
P=/workspace/LIS-VLA/rynnvla-002/exps_lis/runpod/phase0_shard.sh
RV=/workspace/LIS-VLA/rynnvla-002
mkdir -p /workspace/results && cd /workspace/results
(for s in 0 2 4; do bash $P $s 6; done) < /dev/null > streamA.out 2>&1 &
A=$!
(sleep 120; for s in 1 3 5; do bash $P $s 6; done) < /dev/null > streamB.out 2>&1 &
B=$!
while kill -0 $A 2>/dev/null || kill -0 $B 2>/dev/null; do sleep 120; done
source $RV/exps_lis/runpod/pod_env.sh plus
python $RV/tools/summarize_eval.py /workspace/results/phase0 --json /workspace/results/phase0/summary.json \
  > /workspace/results/phase0/summary.txt 2>&1
sync; sleep 30
[ "${NO_STOP:-0}" = 1 ] || runpodctl stop pod $RUNPOD_POD_ID || runpodctl remove pod $RUNPOD_POD_ID
