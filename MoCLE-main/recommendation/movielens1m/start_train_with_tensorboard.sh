#!/usr/bin/env bash
set -euo pipefail

cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main

PYTHON_BIN="${PYTHON_BIN:-/home/liushaokun/miniconda3/envs/lavispy310/bin/python}"
MODEL_PATH="${MODEL_PATH:-/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B}"
TRAIN_FILE="${TRAIN_FILE:-data/movielens1m_train_debug/train.json}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6006}"

MAX_STEPS="${MAX_STEPS:-500}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-512}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"

RUN_NAME="${RUN_NAME:-movielens1m_mocle_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${OUTPUT_DIR}/tb_logs}"

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_LOG_DIR}"

export PYTHON_BIN
export MODEL_PATH
export TRAIN_FILE
export OUTPUT_DIR
export TENSORBOARD_LOG_DIR
export MAX_STEPS
export MAX_TRAIN_SAMPLES
export LOGGING_STEPS
export SAVE_STEPS
export NUM_EXPERTS
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cat > "${OUTPUT_DIR}/run_env.txt" <<EOF
PYTHON_BIN=${PYTHON_BIN}
MODEL_PATH=${MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE}
GPU_ID=${GPU_ID}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
MAX_STEPS=${MAX_STEPS}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}
LOGGING_STEPS=${LOGGING_STEPS}
SAVE_STEPS=${SAVE_STEPS}
NUM_EXPERTS=${NUM_EXPERTS}
OUTPUT_DIR=${OUTPUT_DIR}
TENSORBOARD_LOG_DIR=${TENSORBOARD_LOG_DIR}
PORT=${PORT}
EOF

echo "Output dir          : ${OUTPUT_DIR}"
echo "TensorBoard log dir : ${TENSORBOARD_LOG_DIR}"
echo "GPU                 : ${GPU_ID}"
echo "Max steps           : ${MAX_STEPS}"

echo "Starting TensorBoard on port ${PORT}..."
nohup "${PYTHON_BIN%/python}/tensorboard" \
  --logdir "${TENSORBOARD_LOG_DIR}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  > "${OUTPUT_DIR}/tensorboard.log" 2>&1 &
TB_PID=$!
echo "${TB_PID}" > "${OUTPUT_DIR}/tensorboard.pid"
echo "TensorBoard PID     : ${TB_PID}"
echo "TensorBoard URL     : http://localhost:${PORT}"

echo "Starting training..."
echo "Training log is also written to ${OUTPUT_DIR}/train.log"

bash recommendation/movielens1m/run_train_single_gpu.sh 2>&1 | tee "${OUTPUT_DIR}/train.log"
