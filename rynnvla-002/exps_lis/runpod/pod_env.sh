#!/bin/bash
# Source on every pod (after the volume has been set up once):
#   source /workspace/LIS-VLA/rynnvla-002/exps_lis/runpod/pod_env.sh [vanilla|plus]
# Installs the few system libraries the container lacks, activates the shared venv,
# selects which LIBERO build is importable, and checks that MuJoCo renders on the GPU.
W=/workspace
KIND=${1:-plus}
if ! ldconfig -p | grep -q libMagickWand; then
  apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libmagickwand-dev libexpat1 libfontconfig1 >/dev/null
fi
source $W/venv/bin/activate
export HF_HOME=$W/hf
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export VANILLA_LIBERO_ROOT=$W/third_party/LIBERO
if [ "$KIND" = vanilla ]; then
  export PYTHONPATH=$W/third_party/LIBERO:$W/LIS-VLA/rynnvla-002:$W/LIS-VLA
  export LIBERO_CONFIG_PATH=$W/libero_cfg/vanilla
else
  export PYTHONPATH=$W/third_party/LIBERO-plus:$W/LIS-VLA/rynnvla-002:$W/LIS-VLA
  export LIBERO_CONFIG_PATH=$W/libero_cfg/plus
fi
echo "LIBERO build: $KIND  |  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
