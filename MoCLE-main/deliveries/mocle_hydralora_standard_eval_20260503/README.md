# MoCLE Same-Standard Retrain and Evaluation

本交付目录汇总了 MoCLE 在 pseudo-user 数据上的重新训练、问题定位、同标准评估结果，以及与 HydraLoRA 既有结果的对比。

## 结论

最初 MoCLE eval accuracy 只有 `26.17%`，主要不是模型完全无效，而是训练/eval 标准不一致以及训练超参不一致导致的。修正后，MoCLE 在同一套 HydraLoRA pseudo-user valid split 上达到：

```text
accuracy_full_target = 69.93%
accuracy_letter_only = 69.06%
```

推荐使用的 MoCLE checkpoint：

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main/outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940
```

与 HydraLoRA 结果相比：

| Model / checkpoint | Valid examples | Full-target acc | Letter-only acc |
|---|---:|---:|---:|
| MovieLens HydraLoRA ckpt-53000 | 6028 | 87.29% | 87.01% |
| Pseudo-user HydraLoRA balanced ckpt-11000 | 1949 | 73.52% | 73.63% |
| Pseudo-user MoCLE corrected ckpt-11940 | 1949 | 69.93% | 69.06% |

## 问题定位

前一次结果低的原因有三点：

1. **训练和评估不是同一标准。**
   之前 MoCLE 训练用的是从 interaction 重新构造的 prompt，但 eval 用的是 HydraLoRA 的 `valid.json` prompt。prompt 格式、候选构造和样本分布都不一致。

2. **优化超参不一致。**
   之前 MoCLE 使用 `lr=1e-5`、无 scheduler、`gradient_accumulation_steps=1`。HydraLoRA 使用 `lr=2e-4`、`cosine` scheduler、`warmup_ratio=0.03`、`gradient_accumulation_steps=8`。

3. **训练顺序不一致。**
   之前 MoCLE DataLoader 是 `shuffle=False`。HydraLoRA Trainer 默认 shuffle。对于 cosine LR 来说，固定顺序会让不同 domain 在不同学习率阶段被训练，影响公平性。

## 最终同标准设置

数据标准：

```text
source train: /vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/data/hydralora_pseudo_v2fix_all/train.json
source valid: /vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/data/hydralora_pseudo_v2fix_all/valid.json
MoCLE data : /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main/data/hydralora_pseudo_v2fix_all_mocle_standard
```

MoCLE 版数据没有改 prompt，只增加了 `cluster_id`：

```text
movielens -> cluster_id 0 -> expert_0
goodreads -> cluster_id 1 -> expert_1
mind      -> cluster_id 2 -> expert_2
kuairec   -> cluster_id 3 -> expert_3
```

最终训练设置：

```text
base model: /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B
train file: data/hydralora_pseudo_v2fix_all_mocle_standard/train.json
valid file: data/hydralora_pseudo_v2fix_all_mocle_standard/valid.json
train samples: 95523
optimizer: AdamW
learning_rate: 2e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
warmup_steps: 358
gradient_accumulation_steps: 8
batch_size: 1
max_steps: 11940
shuffle_train: true
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.05
target modules: q_proj,k_proj,v_proj,o_proj
num experts: 4
```

## MoCLE Ablation Trail

| Run | Key change | Full-target acc | Letter-only acc |
|---|---|---:|---:|
| `mocle_lr1e5_ckpt95523` | Same prompt data, but old `lr=1e-5`, no scheduler, no shuffle | 38.38% | 38.53% |
| `mocle_hydraopt_ckpt11940` | HydraLoRA optimizer schedule, no shuffle | 66.65% | 66.50% |
| `mocle_hydraopt_ckpt11000` | Same as above, ckpt 11000 | 66.19% | 65.93% |
| `mocle_hydraopt_shuffle_ckpt11940` | HydraLoRA optimizer schedule + shuffle | 69.93% | 69.06% |
| `mocle_hydraopt_shuffle_ckpt11000` | Same as above, ckpt 11000 | 69.47% | 69.16% |

## Domain Breakdown

Final MoCLE corrected checkpoint:

| Domain | Examples | Full-target acc | Letter-only acc |
|---|---:|---:|---:|
| goodreads | 621 | 85.83% | 85.51% |
| kuairec | 141 | 29.08% | 19.15% |
| mind | 561 | 49.20% | 48.48% |
| movielens | 626 | 81.95% | 82.43% |

HydraLoRA pseudo-user balanced checkpoint:

| Domain | Examples | Full-target acc | Letter-only acc |
|---|---:|---:|---:|
| goodreads | 621 | 86.47% | 86.31% |
| kuairec | 141 | 60.28% | 60.28% |
| mind | 561 | 53.30% | 53.30% |
| movielens | 626 | 81.79% | 82.27% |

MoCLE 已经在 `goodreads` 和 `movielens` 接近 HydraLoRA；主要差距集中在 `kuairec` 和 `mind`。当前 MoCLE 是 hard domain routing，每个 domain 固定到一个 expert；HydraLoRA 使用其自己的多 LoRA routing/mixing 实现，这可能是剩余差距的主要来源。

## 复现命令

### 重新训练最终 MoCLE

```bash
cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main

CUDA_VISIBLE_DEVICES=1 /home/liushaokun/miniconda3/envs/lavispy310/bin/python \
  recommendation/movielens1m/train_single_gpu.py \
  --model_name_or_path /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --tokenizer_name_or_path /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --train_file data/hydralora_pseudo_v2fix_all_mocle_standard/train.json \
  --output_dir outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729 \
  --tensorboard_log_dir outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/tb_logs \
  --max_train_samples 95523 \
  --max_seq_length 512 \
  --batch_size 1 \
  --max_steps 11940 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --logging_steps 10 \
  --save_steps 1000 \
  --torch_dtype bfloat16 \
  --train_mode mocle \
  --num_experts 4 \
  --shuffle_train
```

### 重新评估最终 MoCLE

```bash
cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main

CUDA_VISIBLE_DEVICES=1 /home/liushaokun/miniconda3/envs/lavispy310/bin/python \
  recommendation/movielens1m/eval_mocle_choice_accuracy.py \
  --base-model /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --checkpoint outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940 \
  --data data/hydralora_pseudo_v2fix_all_mocle_standard/valid.json \
  --output deliveries/mocle_hydralora_standard_eval_20260503/mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.json \
  --group-field domain \
  --route-field domain \
  --batch-size 16
```

## 文件说明

| File | Description |
|---|---|
| `SUMMARY.txt` | 简版结果汇总和结论 |
| `HYDRALORA_EVAL_SUMMARY.txt` | 原始 HydraLoRA eval summary 备份 |
| `mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.json` | 最终推荐 MoCLE checkpoint 的完整 eval 结果 |
| `mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.log` | 最终推荐 MoCLE checkpoint 的 eval 日志 |
| `mocle_hydraopt_shuffle_train_summary.json` | 最终推荐 MoCLE 训练 summary |
| `standard_data_build_stats.json` | HydraLoRA 标准数据转 MoCLE 数据的统计 |
| `train_single_gpu.py` | 本次修正后的训练脚本快照 |
| `eval_mocle_choice_accuracy.py` | MoCLE 4-choice accuracy 评估脚本快照 |
| `mocle_standard_ckpt95523_valid_accuracy.json` | 同 prompt 但旧训练设置的 ablation 结果 |
| `mocle_hydraopt_ckpt11940_valid_accuracy.json` | 同优化设置但不 shuffle 的 ablation 结果 |
| `mocle_hydraopt_shuffle_ckpt11000_valid_accuracy.json` | shuffle 版 checkpoint 11000 的对照结果 |

## Deliverable Path

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main/deliveries/mocle_hydralora_standard_eval_20260503
```
