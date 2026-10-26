#!/bin/bash
# Source on every pod (after the volume has been set up once):
#   source /workspace/LIS-VLA/rynnvla-002/exps_lis/runpod/pod_env.sh [vanilla|plus]
# Installs the few system libraries the container lacks, activates the shared venv,
# selects which LIBERO build is importable, and checks that MuJoCo renders on the GPU.
W=/workspace
KIND=${1:-plus}
if ! ldconfig -p | grep -q libMagickWand; then
  apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libglib2.0-0 libmagickwand-dev libexpat1 libfontconfig1 unzip >/dev/null
fi
# LIBERO-plus assets live on local disk (see setup_volume.sh); unpack once per pod (~2 min)
if [ "$KIND" != vanilla ] && [ ! -f /root/lpassets/.complete ] && [ -f $W/hf/libero_plus/assets.zip ]; then
  echo "unpacking LIBERO-plus assets to local disk..."
  rm -rf /root/lpassets /root/lpassets_tmp && mkdir -p /root/lpassets_tmp
  # the archive root is nested ~11 levels deep (…/LIBERO-plus-0/assets/…); extract, then locate it
  unzip -q -o $W/hf/libero_plus/assets.zip -d /root/lpassets_tmp
  NEWOBJ=$(find /root/lpassets_tmp -type d -path '*/assets/new_objects' -print -quit)
  if [ -z "$NEWOBJ" ]; then
    echo "ERROR: assets/new_objects not found in assets.zip - not touching anything else"
    return 1 2>/dev/null || exit 1
  fi
  mv "$(dirname "$NEWOBJ")" /root/lpassets && rm -rf /root/lpassets_tmp   # same local filesystem: a rename
  if [ ! -d /root/lpassets/scenes ]; then
    echo "ERROR: unpacked assets look incomplete (no scenes/)"; return 1 2>/dev/null || exit 1
  fi
  touch /root/lpassets/.complete
  echo "assets ready: $(ls /root/lpassets | tr '\n' ' ')"
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
