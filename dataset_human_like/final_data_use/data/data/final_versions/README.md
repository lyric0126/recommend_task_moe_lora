# 最终 Pseudo-User 数据版本说明

本目录包含最终发布的 pseudo-user 数据集版本，基于四个域构建：

- MovieLens
- Goodreads
- MIND
- KuaiRec

默认推荐版本：

- **`v2fix_all/`**

---

## 版本总览

| 版本 | 角色 | 用户数 | 交互数 | 置信度分布 | Global | Semantic | Temporal | Behavior | Full | 用途 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `random/` (`v0`) | 随机基线 | 1542 | 290161 | random: 1542 | 0.395468 | N/A | N/A | N/A | N/A | 最弱 baseline |
| `v1/` | 结构化基线 | 1751 | 481974 | loose: 1746, medium: 5 | 0.453148 | 0.248588 | 0.767702 | 0.820482 | N/A | 保守 baseline |
| `v2/` | 宽覆盖退化版本 | 5253 | 4841259 | loose: 5253 | 0.316953 | 0.101169 | 0.583215 | 0.832109 | 0.304942 | 仅用于退化对照 |
| `v2fix_strict/` | **核心集** | 1211 | 215057 | strict: 1211 | 0.704164 | 0.475797 | 0.811773 | 0.877793 | 0.701151 | 最高质量核心集 |
| `v2fix_all/` | **主发布版本** | 1542 | 298777 | strict: 1211, medium: 331 | 0.680195 | 0.470436 | 0.774082 | 0.871076 | 0.674013 | 默认使用 |

说明：

- `v2fix_strict`：只保留 `strict`，适合最高质量、最保守分析。
- `v2fix_all`：保留 `strict + medium`，是默认主发布版本。
- `v2`：保留为负面对照，用来说明覆盖扩张但质量下降的情况。
- `random/v0`：随机拼接版本，仅用于最弱 baseline。

---

## 评估说明

我们做了两类评估：

1. **Pseudo-user 构造质量评估**
2. **下游推荐评估**

### 1. Pseudo-user 构造质量评估

目标是评估：一个 pseudo-user 中来自不同域的 source users，是否像“同一个潜在真实用户”的跨域行为切片。

当前采用 **anchor-based** 评估协议：

- 以 MovieLens 用户为 anchor
- 分别比较：
  - MovieLens vs Goodreads
  - MovieLens vs MIND
  - MovieLens vs KuaiRec

每个跨域 pair 计算以下分数：

- `semantic`：语义偏好相似度
- `temporal`：时间行为相似度
- `behavior`：行为风格相似度
- `activity`：活跃度相似度
- `full`：加权综合分

V2-fix 的 `full` 权重为：

```yaml
matching_weights:
  semantic: 0.12
  recency_semantic: 0.06
  activity: 0.28
  temporal: 0.34
  behavior: 0.20
```

因此：

```text
full =
  0.12 * semantic
+ 0.06 * recent_semantic
+ 0.28 * activity
+ 0.34 * temporal
+ 0.20 * behavior
```

一个 pseudo-user 的 `global_consistency_score` 是多个 anchor-to-domain pair score 的平均值。  
整体版本分数则是对所有 pseudo-users 的平均：

- `semantic_consistency`
- `temporal_consistency`
- `behavior_consistency`
- `global_consistency`

### 2. 置信度分层

V2-fix 根据 `global_consistency_score` 分为：

- `strict`
- `medium`
- `loose`

最终发布版本中：

- `v2fix_strict`：只保留 `strict`
- `v2fix_all`：保留 `strict + medium`

`v2` 全部落入 `loose`，因此不推荐作为主版本。

### 3. Random Mix 评估

`random/v0` 不使用匹配方法，而是随机拼接 source users。  
但它使用和其他版本相同的 consistency 评估协议，因此可作为公平的最弱 baseline。

整体 global consistency：

```text
random/v0      global = 0.395468
v1             global = 0.453148
v2             global = 0.316953
v2fix_strict   global = 0.704164
v2fix_all      global = 0.680195
```

这说明：

- `v1` 优于 random
- `v2` 因覆盖扩张和 loose 阈值而退化
- `v2fix_strict` 质量最高
- `v2fix_all` 在高质量和规模之间平衡最好

---

## 下游推荐检查

V3 额外测试了 pseudo-user augmentation 是否能提升轻量推荐 baseline。

设置：

- target domain：MovieLens、Goodreads
- split：leave-last-2-out
- baseline：
  - popularity
  - item-item co-occurrence
- 指标：
  - Recall@10
  - NDCG@10
  - HitRate@10
  - MRR@10

主结果（Recall@10）：

| Target | Single-domain | Random-mix | V2-fix |
| --- | ---: | ---: | ---: |
| Goodreads | 0.054505 | 0.050056 | 0.044494 |
| MovieLens | 0.083952 | 0.082239 | 0.079383 |

解释：

- `V2-fix` 明显提升了 pseudo-user 构造质量
- 但在当前轻量 popularity/item-item baseline 下，没有带来推荐收益
- 原因很可能是这些 baseline 只使用目标域共现，没有真正利用跨域 pseudo-user 结构

因此当前应理解为：

- **最佳数据构造版本**：`v2fix_all`
- **最佳高质量核心集**：`v2fix_strict`
- **是否能在更强多域模型中带来收益**：仍是未来工作

---

## 参数搜索检查

在 V2-fix 之后又做了两轮参数搜索，搜索对象包括：

- matching weights
- strict / medium 阈值
- top-k
- fallback candidate size

这些搜索结果只是实验产物，**没有替代最终发布版本**。

| 版本 | Global | 用户数 | 交互数 | 置信度分布 | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| 默认 `v2fix_all` | 0.680195 | 1542 | 298777 | strict: 1211, medium: 331, loose: 0 | 保持默认 |
| 第一轮 tuned | 0.682476 | 1629 | 359630 | strict: 1026, medium: 575, loose: 28 | global 略高，但引入 loose 和规模膨胀，拒绝 |
| 第二轮 round2 | 0.674966 | 1550 | 308491 | strict: 1089, medium: 453, loose: 8 | 比第一轮稳，但仍有 loose 且 global 低于默认，拒绝 |

结论：

- 第一轮 tuned 追求更高 global，但牺牲了发布质量。
- 第二轮 round2 更保守，但全量后仍出现 `8` 个 loose，且 global 低于默认。
- 当前默认 `v2fix_all` 仍是最稳的主发布版本。

搜索产物：

```text
data/processed/search/
data/processed/search_round2/
configs/pseudo_user_pipeline_v2fix_tuned.yaml
configs/pseudo_user_pipeline_v2fix_round2_best.yaml
reports/search_comparison.md
reports/search_round2_comparison.md
```

---

## 文件内容

每个版本目录包含：

```text
pseudo_user_metadata.parquet
pseudo_user_interactions.parquet
summary.json
README.md
```

索引文件：

```text
version_index.json
```

---

## 文件格式说明

这些文件保留 `.parquet` 后缀以保持 pipeline 兼容。  
但在当前环境下，由于没有 parquet backend，实际写出的是：

> **带 metadata header 的 JSONL fallback 文件**

建议通过仓库工具读取：

```python
from src.io_utils import read_table

metadata = read_table("data/final_versions/v2fix_all/pseudo_user_metadata.parquet")
interactions = read_table("data/final_versions/v2fix_all/pseudo_user_interactions.parquet")
```

---

## 推荐使用方式

- 默认使用：`v2fix_all/`
- 最高质量核心集：`v2fix_strict/`
- 最弱 baseline：`random/`
- 保守结构化 baseline：`v1/`
- 退化对照版本：`v2/`

---

## 关键字段

`pseudo_user_metadata.parquet` 包含：

- `pseudo_user_id`
- `source_members`
- `domains_present`
- `global_consistency_score`
- `confidence_level`

`pseudo_user_interactions.parquet` 包含：

- `pseudo_user_id`
- `dataset`
- `source_user_id`
- `user_id`
- `item_id`
- `timestamp`
- `raw_event`
- `event_value`
- `item_text`
- `item_category`
