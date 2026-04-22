# MovieLens MoCLE-v0 改动说明

这份文档说明当前仓库里如何把 MovieLens-1M 推荐任务和 MoCLE-style 多 LoRA expert 训练放到一起。

当前实现不是原始 VLM/LAVIS 版 MoCLE 的完整复现，而是一个用于纯文本推荐任务的最小可运行 MoCLE-v0：

- 输入任务是 MovieLens-1M next-item recommendation。
- 基座模型是真实 Llama-3.2-1B。
- 参数高效训练使用本仓库 `peft-main/src` 里的 LoRA。
- MoCLE 路由使用 `cluster_id` 选择不同 LoRA expert。
- v0 只支持 `batch_size=1` 下单样本单 expert 路由。
- v0 暂不实现 universal expert 叠加。

## 主要改动

### 1. 数据预处理增加 cluster_id

文件：

```text
recommendation/movielens1m/preprocess_ml1m.py
```

主要改动：

- 从 MovieLens-1M 原始数据构造推荐 instruction 样本。
- 每条样本保留原有字段：
  - `instruction`
  - `input`
  - `output`
  - `task_type`
  - `user_id`
  - `history_movie_ids`
  - `target_movie_id`
  - `candidate_movie_ids`
- 新增字段：
  - `cluster_id`

当前 v0 的 cluster 规则很简单：

```text
cluster_id = task_type
```

也就是说，先把已有的 `task_type` 直接当作 MoCLE expert 路由依据。后续如果要做更正式的聚类，可以只替换 `cluster_id` 的生成逻辑，dataset 和训练脚本不用大改。

### 2. Dataset 和 collator 透传 cluster_id

文件：

```text
recommendation/movielens1m/dataset.py
```

主要改动：

- 读取 JSON 样本时支持 `cluster_id`。
- 为旧数据保留兼容逻辑：

```text
如果样本没有 cluster_id，则使用 task_type 作为 cluster_id。
```

- `InstructionDataset.__getitem__()` 返回 `cluster_id`。
- `collate_features()` 把 `cluster_id` 放入 batch：
  - `return_tensors=False` 时返回 list。
  - `return_tensors=True` 时返回 `torch.long` tensor。

训练时 batch 中会包含：

```text
input_ids
attention_mask
labels
cluster_id
```

其中 `cluster_id` 不会传给 Llama forward，而是只用于 MoCLE expert 路由。

### 3. 数据检查脚本显示 cluster_id

文件：

```text
recommendation/movielens1m/check_dataset.py
```

主要改动：

- 打印首条样本的 `cluster_id`。
- 打印 collated batch 中的 `cluster_id`。
- 保留 `DATASET_CHECK_OK`，方便快速判断数据链路是否正常。

示例输出里应该能看到：

```text
first cluster_id  : 2
batch keys        : ['attention_mask', 'cluster_id', 'input_ids', 'labels']
batch cluster_id  : [2, 7]
DATASET_CHECK_OK
```

### 4. 训练脚本增加 train_mode=mocle

文件：

```text
recommendation/movielens1m/train_single_gpu.py
```

当前训练脚本是自定义 PyTorch training loop，不是 Hugging Face Trainer。

主要改动：

- 新增 `--train_mode mocle`。
- 新增 `--num_experts`。
- 保留旧模式：
  - `full`
  - `lm_head`
  - `lora`
- `mocle` 模式下使用本仓库 `peft-main/src` 创建多个 LoRA adapter：

```text
expert_0
expert_1
expert_2
expert_3
...
```

构造方式：

```text
先用 get_peft_model(..., adapter_name="expert_0") 创建 expert_0。
再用 model.add_adapter("expert_i", lora_config) 追加其他 expert。
```

训练时不再直接 `model(**batch)`，因为 batch 里有 `cluster_id`。实际 forward 只传：

```text
input_ids
attention_mask
labels
```

MoCLE 路由逻辑：

```text
cluster_id = batch["cluster_id"][0]
expert_id = cluster_id % num_experts
model.set_adapter(f"expert_{expert_id}")
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
```

因为 v0 只做最小可跑版本，所以要求：

```text
batch_size = 1
```

### 5. 增加小规模正式训练和 TensorBoard

文件：

```text
recommendation/movielens1m/train_single_gpu.py
recommendation/movielens1m/run_train_single_gpu.sh
```

训练脚本新增参数：

```text
--logging_steps
--save_steps
--tensorboard_log_dir
--disable_tensorboard
```

默认小规模训练配置：

```text
max_train_samples = 512
max_steps = 100
batch_size = 1
logging_steps = 5
save_steps = 50
train_mode = mocle
num_experts = 4
```

TensorBoard 记录指标：

```text
train/loss
train/lr
train/expert_id
train/cluster_id
train/step_time
train/seq_len
train/grad_norm
train/tokens_per_step
```

同时会写：

```text
outputs/.../train_metrics.jsonl
outputs/.../train_summary.json
outputs/.../tb_logs/events.out.tfevents...
```

### 6. 单卡运行脚本默认走真实主线

文件：

```text
recommendation/movielens1m/run_train_single_gpu.sh
```

默认配置：

```text
PYTHON_BIN=/home/liushaokun/miniconda3/envs/lavispy310/bin/python
MODEL_PATH=/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B
CUDA_VISIBLE_DEVICES=0
TRAIN_FILE=data/movielens1m_train_debug/train.json
OUTPUT_DIR=outputs/movielens1m_mocle_small
TENSORBOARD_LOG_DIR=outputs/movielens1m_mocle_small/tb_logs
MAX_STEPS=100
MAX_TRAIN_SAMPLES=512
NUM_EXPERTS=4
```

脚本可以通过环境变量覆盖，例如：

```bash
MAX_STEPS=500 OUTPUT_DIR=outputs/movielens1m_mocle_500steps bash recommendation/movielens1m/run_train_single_gpu.sh
```

## 推荐任务如何接到 MoCLE

整体链路如下：

```text
MovieLens 原始数据
  -> preprocess_ml1m.py
  -> instruction 样本 JSON
  -> 每条样本带 task_type 和 cluster_id
  -> dataset.py tokenize instruction/input/output
  -> collator 生成 batch，并保留 cluster_id
  -> train_single_gpu.py 读取 cluster_id
  -> cluster_id % num_experts 得到 expert_id
  -> model.set_adapter("expert_{expert_id}")
  -> Llama forward/backward
  -> 只更新当前激活的 LoRA expert 参数
```

推荐任务本身被转成纯文本 instruction tuning：

```text
instruction: 要求模型根据用户历史，从候选电影里选择最可能的下一部电影
input      : 用户历史电影 + 候选电影 A/B/C/D
output     : 正确答案字母，例如 B
```

MoCLE 部分不改变推荐样本的 instruction 格式，只额外使用样本元信息 `cluster_id` 做 expert 路由。

也就是说：

```text
推荐任务负责构造监督学习样本。
MoCLE 负责决定当前样本训练哪个 LoRA expert。
Llama 负责根据 instruction/input 生成 output。
```

## 从零运行

进入仓库：

```bash
cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main
```

生成 MovieLens debug 训练数据：

```bash
/home/liushaokun/miniconda3/envs/lavispy310/bin/python recommendation/movielens1m/preprocess_ml1m.py \
  --raw_dir /vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/ml-1m \
  --out_dir data/movielens1m_train_debug \
  --format hydralora \
  --max_train_samples 512 \
  --max_valid_samples 32 \
  --max_test_samples 32
```

检查数据和 batch：

```bash
PYTHONPATH="${PWD}/peft-main/src:${PWD}:${PYTHONPATH:-}" \
/home/liushaokun/miniconda3/envs/lavispy310/bin/python recommendation/movielens1m/check_dataset.py \
  --data_path data/movielens1m_train_debug/train.json \
  --tokenizer_name_or_path /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --max_samples 8 \
  --batch_size 2 \
  --max_seq_length 512
```

启动 100 step 小规模单卡训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash recommendation/movielens1m/run_train_single_gpu.sh
```

启动 500 step 训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_STEPS=500 \
LOGGING_STEPS=10 \
SAVE_STEPS=100 \
OUTPUT_DIR=outputs/movielens1m_mocle_500steps \
bash recommendation/movielens1m/run_train_single_gpu.sh
```

查看 TensorBoard：

```bash
/home/liushaokun/miniconda3/envs/lavispy310/bin/tensorboard \
  --logdir outputs/movielens1m_mocle_small/tb_logs \
  --host 0.0.0.0 \
  --port 6006
```

## 已验证内容

已经验证过的真实主线：

```text
真实模型：/vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B
真实环境：/home/liushaokun/miniconda3/envs/lavispy310/bin/python
真实设备：CUDA 单卡，NVIDIA L20
真实 PEFT：本仓库 peft-main/src
训练模式：train_mode=mocle
batch_size：1
num_experts：4
```

验证结果：

```text
数据预处理成功
dataset 能读取 MovieLens JSON
batch 中包含 cluster_id
模型初始化成功
PEFT 多 adapter 初始化成功
按 cluster_id 选择 expert 成功
至少 5 个 training step 连续成功
TensorBoard event 文件生成成功
train/loss 写入成功
```

## 当前限制

当前版本是 MoCLE-v0，限制如下：

- 只支持 `batch_size=1` 的单样本单 expert 路由。
- `cluster_id` 暂时直接等于 `task_type`，还不是离线聚类结果。
- 没有实现 universal expert 叠加。
- 没有实现同一个 batch 内不同样本路由到不同 expert 的并行逻辑。
- 当前重点是把 MovieLens 推荐任务和 MoCLE-style LoRA expert 训练跑通，尚未做推荐指标评估。

## 后续建议

下一步可以按这个顺序推进：

```text
1. 用更合理的用户/物品行为聚类替换 cluster_id = task_type。
2. 增加验证集 evaluation，计算 accuracy 或 hit rate。
3. 支持 batch 内按 cluster 分组，多 expert 分组 forward。
4. 加入 universal expert + routed expert 的组合。
5. 保存和加载各 expert adapter，做推理评估。
```
