#!/usr/bin/env bash
set -euo pipefail

cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PWD}/peft-main/src:${PWD}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/liushaokun/miniconda3/envs/lavispy310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B}"
TRAIN_FILE="${TRAIN_FILE:-data/movielens1m_train_debug/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/movielens1m_mocle_small}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${OUTPUT_DIR}/tb_logs}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-50}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"

"${PYTHON_BIN}" recommendation/movielens1m/train_single_gpu.py \
  --model_name_or_path "${MODEL_PATH}" \
  --tokenizer_name_or_path "${MODEL_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --tensorboard_log_dir "${TENSORBOARD_LOG_DIR}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --max_steps "${MAX_STEPS}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --torch_dtype bfloat16 \
  --train_mode mocle \
  --num_experts "${NUM_EXPERTS}"
