#!/bin/bash
set -euo pipefail
exec /home/liushaokun/miniconda3/envs/hydralora/bin/tensorboard   --logdir "/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints"   --host 0.0.0.0   --port "6008"
