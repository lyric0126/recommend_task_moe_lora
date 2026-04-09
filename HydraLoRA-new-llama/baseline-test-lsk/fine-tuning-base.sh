#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

BASE_MODEL="/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B"
TOKENIZER_PATH="${BASE_MODEL}"

DATASET_DIR="/vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/hydralora_ml1m"
VALID_FILE="/vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/hydralora_ml1m/valid.json"

OUTPUT_DIR="/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/test_use_lora/output"
EXP_NAME="baseline_lora1_ml1m_llama32_1b"

mkdir -p "${OUTPUT_DIR}"
rm -rf "${OUTPUT_DIR}/${EXP_NAME}"

python /vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/HydraLoRA/fine-tuning.py \
  --model_name_or_path "${BASE_MODEL}" \
  --tokenizer_name_or_path "${TOKENIZER_PATH}" \
  --dataset_dir "${DATASET_DIR}" \
  --validation_file "${VALID_FILE}" \
  --output_dir "${OUTPUT_DIR}/${EXP_NAME}" \
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
  --num_train_epochs 1 \
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
  --lora_nums 1 \
  --load_in_kbits 16 \
  --report_to tensorboard \
  --logging_dir "${OUTPUT_DIR}/${EXP_NAME}/runs" \
  --overwrite_output_dir
