#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

BASE_MODEL="/home/liushaokun/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TOKENIZER_PATH="${BASE_MODEL}"

DATASET_DIR="/vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/hydralora_ml1m"
VALID_FILE="/vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/hydralora_ml1m/valid.json"

OUTPUT_DIR="/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-main/test_use_lora/output"
EXP_NAME="hydralora_ml1m_qwen"

mkdir -p "${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}/${EXP_NAME}"

python /vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-main/HydraLoRA/fine-tuning.py \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --dataset_dir "${DATASET_DIR}" \
  --validation_file "${VALID_FILE}" \
  --output_dir "${OUTPUT_DIR}/${EXP_NAME}" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 2 \
  --trainable "q_proj,k_proj,v_proj,o_proj" \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --report_to tensorboard \
  --logging_dir "${OUTPUT_DIR}/${EXP_NAME}/runs" \
  # --max_seq_length 512 \
