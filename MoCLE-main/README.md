# MoCLE Recommendation Deliverable

这是清理后的 MoCLE recommendation 交付目录。它保留了最终同标准训练数据、最终 checkpoint、评估报告、复现脚本、训练日志和必要源码，可以用于复现训练、复现评估、加载最终 MoCLE adapter。

原始 MoCLE 项目 README 已归档到：

```text
archive_uncertain/README_original_mocle.md
```

## 一句话结论

最终推荐模型：

```text
outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940
```

最终评估结果：

| Model | Valid examples | Full-target acc | Letter-only acc |
|---|---:|---:|---:|
| MoCLE corrected checkpoint-step-11940 | 1949 | 69.93% | 69.06% |
| HydraLoRA pseudo-user balanced ckpt-11000 | 1949 | 73.52% | 73.63% |

最终结果和对比报告在：

```text
deliveries/mocle_hydralora_standard_eval_20260503/
```

## 文件夹结构

```text
MoCLE-main/
├── README.md
├── cleanup_keep_list.md
├── cleanup_delete_list.md
├── cleanup_size_report.md
├── archive_uncertain/
├── data/
│   └── hydralora_pseudo_v2fix_all_mocle_standard/
├── deliveries/
│   └── mocle_hydralora_standard_eval_20260503/
├── outputs/
│   └── mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/
├── recommendation/
│   └── movielens1m/
├── peft-main/
├── images/
├── mocle.py
└── mocle.yaml
```

## 每个目录是什么

| Path | 作用 | 是否关键 |
|---|---|---|
| `README.md` | 当前交付说明，解释怎么复现和怎么使用 | 是 |
| `data/hydralora_pseudo_v2fix_all_mocle_standard/` | 最终训练/评估数据，已对齐 HydraLoRA 标准 | 是 |
| `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/` | 最终训练 run，包含最终 checkpoint 和训练日志 | 是 |
| `deliveries/mocle_hydralora_standard_eval_20260503/` | 最终评估交付包，包含结果、报告、脚本快照 | 是 |
| `recommendation/movielens1m/` | 训练、评估、数据处理源码 | 是 |
| `peft-main/` | 本项目使用的本地 PEFT 版本，用于加载 LoRA experts | 是 |
| `mocle.py`, `mocle.yaml` | 原 MoCLE 模型和配置文件 | 建议保留 |
| `cleanup_*.md` | 清理前后的保留、删除、空间统计记录 | 建议保留 |
| `archive_uncertain/` | 旧 README 和早期不一致评估结果，只用于追溯 | 可选 |
| `images/` | 原项目图片资源 | 可选 |

## 整个流程每一步是什么

### Step 1. 标准数据

最终使用的是 HydraLoRA 的 pseudo-user train/valid split，并转换为 MoCLE 可训练格式。转换后数据在：

```text
data/hydralora_pseudo_v2fix_all_mocle_standard/
├── train.json
├── valid.json
└── build_stats.json
```

数据字段保留了原始 prompt 和 answer，只额外增加 `cluster_id` 供 MoCLE routing 使用。

domain 到 expert 的映射：

| Domain | cluster_id | Expert |
|---|---:|---|
| `movielens` | 0 | `expert_0` |
| `goodreads` | 1 | `expert_1` |
| `mind` | 2 | `expert_2` |
| `kuairec` | 3 | `expert_3` |

### Step 2. MoCLE 训练

训练脚本：

```text
recommendation/movielens1m/train_single_gpu.py
```

最终训练 run：

```text
outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/
├── checkpoint-step-11940/
├── run_env.txt
├── train.log
├── train_metrics.jsonl
└── train_summary.json
```

关键训练设置：

| 参数 | 值 |
|---|---|
| base model | `/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B` |
| train samples | 95523 |
| max steps | 11940 |
| batch size | 1 |
| gradient accumulation | 8 |
| learning rate | `2e-4` |
| scheduler | cosine |
| warmup ratio | `0.03` |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| experts | 4 |
| shuffle | true |

### Step 3. 评估

评估脚本：

```text
recommendation/movielens1m/eval_mocle_choice_accuracy.py
```

评估方式：

- 对每条样本分别计算 `A/B/C/D` 四个候选答案的 log-prob score
- 选择 score 最高的字母作为预测
- 分别统计 `full-target accuracy` 和 `letter-only accuracy`
- 按 `domain` 分组统计结果
- 按 domain 路由到对应 expert

最终评估结果：

```text
deliveries/mocle_hydralora_standard_eval_20260503/mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.json
```

### Step 4. 结果汇总

最终交付报告：

```text
deliveries/mocle_hydralora_standard_eval_20260503/
├── README.md
├── SUMMARY.txt
├── HYDRALORA_EVAL_SUMMARY.txt
├── mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.json
├── mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.log
├── mocle_hydraopt_shuffle_train_summary.json
├── train_single_gpu.py
└── eval_mocle_choice_accuracy.py
```

其中：

- `README.md`: 详细问题定位、训练设置、ablation 和 HydraLoRA 对比
- `SUMMARY.txt`: 简版结果汇总
- `mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.json`: 最终 MoCLE 完整评估结果
- `HYDRALORA_EVAL_SUMMARY.txt`: HydraLoRA 原始结果备份
- `train_single_gpu.py`, `eval_mocle_choice_accuracy.py`: 本次交付使用的脚本快照

## 怎么复现训练

先进入项目根目录：

```bash
cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main
```

运行最终训练命令：

```bash
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

训练完成后应生成：

```text
outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940/
```

如果只想保留交付版本，可以删除中间 checkpoint，只保留最终 `checkpoint-step-11940` 和 run 级别的 `train_summary.json`、`train.log`、`train_metrics.jsonl`、`run_env.txt`。

## 怎么复现评估

进入项目根目录：

```bash
cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main
```

运行最终评估命令：

```bash
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

预期主结果：

```text
total = 1949
accuracy_full_target = 0.6993329912775782
accuracy_letter_only = 0.6906105695228322
```

domain 维度结果：

| Domain | Examples | Full-target acc | Letter-only acc |
|---|---:|---:|---:|
| goodreads | 621 | 85.83% | 85.51% |
| kuairec | 141 | 29.08% | 19.15% |
| mind | 561 | 49.20% | 48.48% |
| movielens | 626 | 81.95% | 82.43% |

## 怎么使用最终 checkpoint

最终 checkpoint 是 4 个 LoRA adapter expert，不是完整 base model。因此使用时需要同时提供：

- base model: `/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B`
- adapter checkpoint: `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940`
- 本地 PEFT 源码: `peft-main/src`

最小加载示例：

```python
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_root = "/vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main"
sys.path.insert(0, os.path.join(repo_root, "peft-main", "src"))

from peft import PeftModel

base_model = "/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B"
checkpoint = os.path.join(
    repo_root,
    "outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940",
)

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(
    model,
    os.path.join(checkpoint, "expert_0"),
    adapter_name="expert_0",
)

for idx in [1, 2, 3]:
    model.load_adapter(
        os.path.join(checkpoint, f"expert_{idx}"),
        adapter_name=f"expert_{idx}",
    )

model.to("cuda")
model.eval()
```

根据 domain 选择 expert：

```python
domain_to_expert = {
    "movielens": "expert_0",
    "goodreads": "expert_1",
    "mind": "expert_2",
    "kuairec": "expert_3",
}

domain = "movielens"
model.set_adapter(domain_to_expert[domain])
```

如果你的输入格式和 valid 数据一致，可以参考评估脚本里的 prompt 构造：

```python
source = example["instruction"]
if example.get("input"):
    source = source + "\n" + example["input"]
prompt = f"{source}</s>"
```

对于 A/B/C/D 选择题，推荐沿用 `eval_mocle_choice_accuracy.py` 的 scoring 逻辑，而不是直接 `generate()`，这样和交付 accuracy 完全一致。

## 关键文件说明

| 文件 | 说明 |
|---|---|
| `data/hydralora_pseudo_v2fix_all_mocle_standard/train.json` | 最终训练数据 |
| `data/hydralora_pseudo_v2fix_all_mocle_standard/valid.json` | 最终评估数据 |
| `data/hydralora_pseudo_v2fix_all_mocle_standard/build_stats.json` | 数据构建统计 |
| `outputs/.../checkpoint-step-11940/` | 最终 checkpoint |
| `outputs/.../train_summary.json` | 最终训练 summary |
| `outputs/.../train.log` | 最终训练日志 |
| `outputs/.../train_metrics.jsonl` | step 级训练指标 |
| `recommendation/movielens1m/train_single_gpu.py` | 训练入口 |
| `recommendation/movielens1m/eval_mocle_choice_accuracy.py` | 评估入口 |
| `deliveries/.../README.md` | 最终交付报告 |
| `deliveries/.../SUMMARY.txt` | 简版结果总结 |
| `deliveries/.../mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.json` | 最终完整 eval JSON |
| `cleanup_keep_list.md` | 清理时保留了什么 |
| `cleanup_delete_list.md` | 清理时删除了什么 |
| `cleanup_size_report.md` | 清理前后空间统计 |

## 清理后状态

清理前：

- 项目总大小约 `29G`
- `outputs/` 约 `26G`
- `data/` 约 `3.3G`

清理后：

- 项目总大小约 `164M`
- `data/` 约 `124M`
- `outputs/` 约 `26M`
- `deliveries/` 约 `4.8M`
- `archive_uncertain/` 约 `988K`

释放空间：

```text
30,332,671,712 bytes
约 28.25 GiB / 30.33 GB
```

已删除：

- 失败或被取代的旧训练输出
- debug/smoke/launch/tensorboard 验证输出
- 最终 run 内的中间 checkpoint
- 与最终标准不一致的生成数据
- 重复 MovieLens 大数据
- `__pycache__` 和 TensorBoard 临时日志
- 空目录

不确定但有追溯价值的内容移动到：

```text
archive_uncertain/
```

## 注意事项

- 这个 checkpoint 不是完整模型，只包含 LoRA expert adapter；加载时必须同时有 base model。
- 当前评估标准是 A/B/C/D choice scoring，不是自由生成文本评测。
- `kuairec` 和 `mind` 仍是主要低分 domain；`goodreads` 和 `movielens` 已接近 HydraLoRA。
- 如果后续继续训练，建议新建独立 output run 目录，最终只保留 final checkpoint、训练 summary、训练日志、eval JSON/log 和脚本快照。
