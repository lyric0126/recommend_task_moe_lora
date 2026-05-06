#!/bin/bash
set -euo pipefail

REPO="/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama"
PYTHON_BIN="/home/liushaokun/miniconda3/envs/hydralora/bin/python"
CURRENT_PID="$1"
CURRENT_DELIVERY="$2"
CHAIN_LOG="${REPO}/deliveries/chain_balanced_after_current.log"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "${CHAIN_LOG}"
}

log "watching current pilot pid=${CURRENT_PID}, delivery=${CURRENT_DELIVERY}"
while kill -0 "${CURRENT_PID}" 2>/dev/null; do
  sleep 60
done
log "current pilot pid=${CURRENT_PID} ended"

RUN_ID="hydralora_pseudo_v2fix_all_balanced_97k_$(date +%Y%m%d_%H%M%S)"
DELIVERY_DIR="${REPO}/deliveries/${RUN_ID}"
mkdir -p "${DELIVERY_DIR}/logs" "${DELIVERY_DIR}/checkpoints"
ln -sfn "${DELIVERY_DIR}" "${REPO}/deliveries/latest_balanced"
ln -sfn "${DELIVERY_DIR}" "${REPO}/deliveries/latest"

cat > "${DELIVERY_DIR}/README.txt" <<EOT
HydraLoRA pseudo-user balanced training run
run_id=${RUN_ID}
created_at=$(date '+%F %T %Z')
repo=${REPO}
source_data=/vepfs-cnbja62d5d769987/liushaokun/sys_work/dataset_human_like/final_data_use/data/data/final_versions/v2fix_all
prepared_data=${REPO}/data/hydralora_pseudo_v2fix_all
base_model=/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B
sampling=balanced per pseudo_user/domain, MAX_SAMPLES=0, MAX_SAMPLES_PER_USER_DOMAIN=20
expected_sft_examples=97472 total, 95523 train, 1949 valid
checkpoints=${DELIVERY_DIR}/checkpoints/${RUN_ID}
train_log=${DELIVERY_DIR}/logs/train.log
tensorboard_log=${DELIVERY_DIR}/logs/tensorboard.log
EOT

cat > "${DELIVERY_DIR}/run_train.sh" <<EOT
#!/bin/bash
set -euo pipefail
cd "${REPO}"
export CUDA_VISIBLE_DEVICES=1
export OUTPUT_ROOT="${DELIVERY_DIR}/checkpoints"
export EXP_NAME="${RUN_ID}"
export MAX_SAMPLES=0
export MAX_SAMPLES_PER_USER_DOMAIN=20
export NUM_TRAIN_EPOCHS=1
export MAX_STEPS=-1
echo "[run_train] start \$(date '+%F %T %Z')"
echo "[run_train] pid=\$\$"
echo "[run_train] output=\${OUTPUT_ROOT}/\${EXP_NAME}"
bash HydraLoRA/fine-tuning_pseudo_user.sh
echo "[run_train] done \$(date '+%F %T %Z')"
EOT
chmod +x "${DELIVERY_DIR}/run_train.sh"

PORT=$("${PYTHON_BIN}" - <<'PY'
import socket
for port in range(6007, 6030):
    s = socket.socket()
    try:
        s.bind(('0.0.0.0', port))
        print(port)
        s.close()
        break
    except OSError:
        s.close()
else:
    raise SystemExit('no free tensorboard port in 6007-6029')
PY
)
cat > "${DELIVERY_DIR}/run_tensorboard.sh" <<EOT
#!/bin/bash
set -euo pipefail
exec /home/liushaokun/miniconda3/envs/hydralora/bin/tensorboard \
  --logdir "${DELIVERY_DIR}/checkpoints" \
  --host 0.0.0.0 \
  --port "${PORT}"
EOT
chmod +x "${DELIVERY_DIR}/run_tensorboard.sh"

setsid bash -c 'echo $$ > "$1"; exec "$2"' _ "${DELIVERY_DIR}/tensorboard.pid" "${DELIVERY_DIR}/run_tensorboard.sh" \
  </dev/null > "${DELIVERY_DIR}/logs/tensorboard.log" 2>&1 &
echo "${PORT}" > "${DELIVERY_DIR}/tensorboard.port"

setsid bash -c 'echo $$ > "$1"; exec "$2"' _ "${DELIVERY_DIR}/train.pid" "${DELIVERY_DIR}/run_train.sh" \
  </dev/null > "${DELIVERY_DIR}/logs/train.log" 2>&1 &
TRAIN_PID="$(cat "${DELIVERY_DIR}/train.pid")"
log "started balanced run ${RUN_ID}, pid=${TRAIN_PID}, tb_port=${PORT}, delivery=${DELIVERY_DIR}"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 60
done
log "balanced train pid=${TRAIN_PID} ended, collecting results"

"${PYTHON_BIN}" - <<PY > "${DELIVERY_DIR}/RESULTS.txt"
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p = Path('${DELIVERY_DIR}')
print('run_id=${RUN_ID}')
print('delivery=${DELIVERY_DIR}')
print('tensorboard_port=${PORT}')
print('status=finished')
for f in sorted((p/'checkpoints').rglob('events.out.tfevents*')):
    ea = EventAccumulator(str(f)); ea.Reload()
    tags = ea.Tags().get('scalars', [])
    for tag in ['train/loss', 'eval/loss', 'train/learning_rate', 'train/grad_norm', 'train/epoch']:
        if tag in tags:
            vals = ea.Scalars(tag)
            if vals:
                print(f'{tag}: points={len(vals)} last_step={vals[-1].step} last_value={vals[-1].value}')
print('latest_checkpoints:')
for ckpt in sorted((p/'checkpoints'/'${RUN_ID}').glob('checkpoint-*'))[-5:]:
    print(ckpt)
final_model = p/'checkpoints'/'${RUN_ID}'/'sft_lora_model'
print(f'final_sft_lora_model={final_model} exists={final_model.exists()}')
PY
log "wrote ${DELIVERY_DIR}/RESULTS.txt"
