#!/bin/bash
# Build LIS-VLA training data for one LIBERO suite.
#   bash prepare_data.sh goal 5      (suite in goal|object|spatial|10, H = action chunk = imagination horizon)
# Prerequisite: Step 1 of the RynnVLA-002 README (regenerated LIBERO images/actions/states, and the
# standard VLA + world-model conversations/tokens if you want to mix them in).
set -e
SUITE=${1:-goal}
H=${2:-5}
RES=256
TOKENIZER=${TOKENIZER:-../ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af}
BASE=../processed_data/libero_${SUITE}_image_state_action_t_${RES}

cd "$(dirname "$0")/../data"
python lis_conv_generation.py --base_dir $BASE --his 2 --len_action $H --task_name $SUITE \
    --resolution $RES --with_state --img_names imgs_third_view imgs_wrist --output_dir ../processed_data/convs
python pretoken_lis_model.py --task $SUITE --resolution $RES --with_state \
    --img_names imgs_third_view imgs_wrist --his 2 --len_action $H --tokenizer_path $TOKENIZER
bash concate_record_libero.sh

PATTERNS="libero_${SUITE}_his_2_{}_third_view_wrist_w_state_${H}_${RES}_lis"
# mix in the original world-model and VLA data when they exist
[ -d ../processed_data/tokens/libero_${SUITE}_his_1_train_third_view_wrist_a2i_${RES} ] && \
    PATTERNS="$PATTERNS libero_${SUITE}_his_1_{}_third_view_wrist_a2i_${RES}"
[ -d ../processed_data/tokens/libero_${SUITE}_his_2_train_third_view_wrist_w_state_${H}_${RES} ] && \
    PATTERNS="$PATTERNS libero_${SUITE}_his_2_{}_third_view_wrist_w_state_${H}_${RES}"
python concate_action_world_model_data_libero.py --source_dir_patterns $PATTERNS \
    --all_patterns libero_${SUITE}_his_2_third_view_wrist_w_state_${H}_${RES}_lis_awm
