# Running LIS-VLA experiments on RunPod

## Resources
- **Network volume**: 200 GB, in a datacenter that has both RTX A5000 and H100 SXM available. Mounted at `/workspace` on every pod.
- **Phase 0 (no training)**: 6 × RTX A5000 24 GB pods, any PyTorch template, container disk 30 GB, environment variable `NVIDIA_DRIVER_CAPABILITIES=all` (lets MuJoCo render with EGL on the GPU; without it rendering falls back to the CPU and is ~20× slower).

## 1. First pod only: prepare the volume (~30-60 min)
```bash
cd /workspace && git clone https://github.com/wangyk10086/LIS-VLA.git   # private repo: use a token or make it public
bash /workspace/LIS-VLA/rynnvla-002/exps_lis/runpod/setup_volume.sh
```

## 2. Smoke test on that pod (~10 min)
```bash
SMOKE=1 bash /workspace/LIS-VLA/rynnvla-002/exps_lis/runpod/phase0_shard.sh 0 1
python /workspace/LIS-VLA/rynnvla-002/tools/summarize_eval.py /workspace/results/phase0_smoke
```
The preflight prints GPU, simulator ms/step (should be < 30), model query latency with and without search, and GPU memory. Check that every job in the smoke run finished with `errors = 0`.

## 3. Start the 6 shards (one per pod, in tmux so closing the browser does not stop them)
```bash
tmux new -d -s run "bash /workspace/LIS-VLA/rynnvla-002/exps_lis/runpod/phase0_shard.sh <SHARD> 6"
```
with `<SHARD>` = 0 … 5. Progress: `tail -f /workspace/results/phase0/logs/*_shard<SHARD>.log`. A pod that was interrupted can simply run the same command again; finished episodes are skipped.

**Single-pod alternative** (when only one 48 GB GPU is available): `phase0_single_pod.sh` runs two shard streams on the same GPU (~16 GB each), then writes `summary.txt` and stops the pod itself. Measured on an A40: ~30 s per episode per stream, ~8 h for all six shards.

## 4. Results
```bash
python /workspace/LIS-VLA/rynnvla-002/tools/summarize_eval.py /workspace/results/phase0 --json /workspace/results/phase0/summary.json
```
Stop the pods as soon as `logs/DONE_shard*` exists for all six shards. The volume keeps the results.
