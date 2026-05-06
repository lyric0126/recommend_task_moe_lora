#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

REPO_DIR="/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama"
PYTHON_BIN="${PYTHON_BIN:-/home/liushaokun/miniconda3/envs/hydralora/bin/python}"

BASE_MODEL="${BASE_MODEL:-/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"

VERSION_DIR="${VERSION_DIR:-/vepfs-cnbja62d5d769987/liushaokun/sys_work/dataset_human_like/final_data_use/data/data/final_versions/v2fix_all}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/data/hydralora_pseudo_v2fix_all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/test_use_lora/output}"
EXP_NAME="${EXP_NAME:-hydralora_pseudo_v2fix_all_llama32_1b}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

MAX_SAMPLES="${MAX_SAMPLES:-50000}"
MAX_SAMPLES_PER_USER_DOMAIN="${MAX_SAMPLES_PER_USER_DOMAIN:-20}"
MAX_STEPS="${MAX_STEPS:--1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

mkdir -p "${DATASET_DIR}" "${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${REPO_DIR}/HydraLoRA/prepare_pseudo_user_sft.py" \
  --version-dir "${VERSION_DIR}" \
  --output-dir "${DATASET_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-samples-per-user-domain "${MAX_SAMPLES_PER_USER_DOMAIN}"

rm -rf "${DATASET_DIR}/train_512" "${DATASET_DIR}/valid_512" "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${REPO_DIR}/HydraLoRA/fine-tuning.py" \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --dataset_dir "${DATASET_DIR}" \
  --train_file "${DATASET_DIR}/train.json" \
  --validation_file "${DATASET_DIR}/valid.json" \
  --output_dir "${OUTPUT_DIR}" \
  --do_train \
  --do_eval \
  --seed 41 \
  --bf16 \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --logging_strategy steps \
  --logging_steps 10 \
  --logging_first_step True \
  --save_strategy steps \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --max_seq_length 512 \
  --trainable "q_proj,k_proj,v_proj,o_proj" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_nums 4 \
  --load_in_kbits 16 \
  --report_to tensorboard \
  --logging_dir "${OUTPUT_DIR}/runs" \
  --overwrite_output_dir
