#!/bin/bash
# One-time setup of the shared RunPod network volume (mounted at /workspace).
# Run on ONE pod; every later pod only needs `source pod_env.sh`.
#
#   bash setup_volume.sh            # idempotent: re-running skips finished steps
#
# Layout created on the volume:
#   /workspace/venv               python 3.10 env (torch 2.2.0 / transformers 4.43.0, pinned)
#   /workspace/third_party/{LIBERO,LIBERO-plus}
#   /workspace/libero_cfg/{vanilla,plus}/config.yaml
#   /workspace/LIS-VLA/rynnvla-002/ckpts/{RynnVLA-002,chameleon/tokenizer,lumina_tokenizer}
set -euo pipefail
W=/workspace
REPO=$W/LIS-VLA
RV=$REPO/rynnvla-002
mkdir -p $W/third_party $W/libero_cfg/vanilla $W/libero_cfg/plus $W/hf
export HF_HOME=$W/hf

step() { echo; echo "=== $*"; }

step "system packages (needed for wand / rendering checks)"
apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libmagickwand-dev libexpat1 libfontconfig1 unzip git >/dev/null

step "python environment"
pip install -q uv
if [ ! -x $W/venv/bin/python ]; then uv venv -q -p 3.10 $W/venv; fi
cat > $W/constraints.txt <<'EOF'
torch==2.2.0
numpy==1.24.4
transformers==4.43.0
EOF
uv pip install -q -p $W/venv/bin/python -c $W/constraints.txt \
  torch==2.2.0 torchvision==0.17.0
uv pip install -q -p $W/venv/bin/python -c $W/constraints.txt \
  transformers==4.43.0 accelerate==0.33.0 sentencepiece fairscale==0.4.13 numpy==1.24.4 pillow einops safetensors \
  huggingface_hub tqdm pyyaml h5py tensorboard \
  mujoco==3.3.5 robosuite==1.4.1 bddl==3.6.0 easydict gym==0.26.2 imageio imageio-ffmpeg opencv-python-headless \
  scikit-image numba wand future cloudpickle matplotlib pytest
uv pip uninstall -q -p $W/venv/bin/python libero hf-libero 2>/dev/null || true   # a pip 'libero' would shadow the checkouts
$W/venv/bin/python -c "import torch, transformers, numpy; print('torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__)"

step "code"
if [ ! -d $REPO/.git ]; then
  echo "Clone the LIS-VLA repo to $REPO first (see RUNPOD.md)"; exit 1
fi
uv pip install -q -p $W/venv/bin/python -c $W/constraints.txt -e $REPO   # xllmx

step "LIBERO (vanilla) and LIBERO-plus"
[ -d $W/third_party/LIBERO ] || git clone -q --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git $W/third_party/LIBERO
[ -d $W/third_party/LIBERO-plus ] || git clone -q --depth 1 https://github.com/sylvestf/LIBERO-plus.git $W/third_party/LIBERO-plus
PLUS_ASSETS=$W/third_party/LIBERO-plus/libero/libero/assets
if [ ! -d $PLUS_ASSETS/new_objects ]; then
  $W/venv/bin/python - <<'EOF'
from huggingface_hub import hf_hub_download
p = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", local_dir="/workspace/hf/libero_plus")
print(p)
EOF
  rm -rf $PLUS_ASSETS && mkdir -p /workspace/hf/libero_plus/unz
  unzip -q -o /workspace/hf/libero_plus/assets.zip -d /workspace/hf/libero_plus/unz
  A=$(dirname $(find /workspace/hf/libero_plus/unz -type d -name new_objects | head -1))
  mv "$A" $PLUS_ASSETS
fi
ls $PLUS_ASSETS | head
for kind in vanilla plus; do
  [ $kind = vanilla ] && R=$W/third_party/LIBERO/libero/libero || R=$W/third_party/LIBERO-plus/libero/libero
  cat > $W/libero_cfg/$kind/config.yaml <<EOF
benchmark_root: $R
bddl_files: $R/bddl_files
init_states: $R/init_files
datasets: $R/../datasets
assets: $R/assets
EOF
done

step "model weights and tokenizers"
$W/venv/bin/python - <<EOF
from huggingface_hub import snapshot_download
rv = "$RV"
snapshot_download("Alibaba-DAMO-Academy/RynnVLA-002", local_dir=f"{rv}/ckpts/RynnVLA-002",
                  allow_patterns=["VLA_model_256/libero_goal/*", "VLA_model_256/libero_10/*"])
snapshot_download("Alibaba-DAMO-Academy/WorldVLA", local_dir=f"{rv}/ckpts/worldvla_tmp",
                  allow_patterns=["chameleon/tokenizer/*"])
snapshot_download("Alpha-VLLM/Lumina-mGPT-7B-768", local_dir=f"{rv}/ckpts/lumina_tokenizer",
                  allow_patterns=["*.json", "*.model", "*.txt"], ignore_patterns=["*.safetensors*", "*.bin*"])
EOF
mkdir -p $RV/ckpts/chameleon
[ -d $RV/ckpts/chameleon/tokenizer ] || mv $RV/ckpts/worldvla_tmp/chameleon/tokenizer $RV/ckpts/chameleon/tokenizer
ls $RV/ckpts/RynnVLA-002/VLA_model_256 $RV/ckpts/chameleon/tokenizer $RV/ckpts/lumina_tokenizer

step "done - volume is ready. Every pod: source $RV/exps_lis/runpod/pod_env.sh"
