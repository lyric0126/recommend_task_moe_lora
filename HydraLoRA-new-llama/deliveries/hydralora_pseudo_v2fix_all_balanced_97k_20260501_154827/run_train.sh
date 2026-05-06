#!/bin/bash
set -euo pipefail
cd "/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama"
export CUDA_VISIBLE_DEVICES=1
export OUTPUT_ROOT="/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints"
export EXP_NAME="hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827"
export MAX_SAMPLES=0
export MAX_SAMPLES_PER_USER_DOMAIN=20
export NUM_TRAIN_EPOCHS=1
export MAX_STEPS=-1
echo "[run_train] start $(date '+%F %T %Z')"
echo "[run_train] pid=$$"
echo "[run_train] output=${OUTPUT_ROOT}/${EXP_NAME}"
bash HydraLoRA/fine-tuning_pseudo_user.sh
echo "[run_train] done $(date '+%F %T %Z')"
