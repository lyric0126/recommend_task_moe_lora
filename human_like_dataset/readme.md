# 📦 多域推荐数据集构建与基准（MDRB）

## 🚀 项目简介（Overview）

本项目旨在构建一个**科学、可控、可解释的多域推荐数据集基准（Multi-Domain Recommendation Benchmark, MDRB）**，用于研究：

- 跨域推荐（Cross-Domain Recommendation）
- 多行为推荐（Multi-Behavior Recommendation）
- 专家模型（MoE / LoRA / Router）
- 隐私推荐与数据替代（Synthetic Data for Privacy）

现实中，用户行为分散在多个系统中，例如：

- 视频平台（短视频 / 长视频）
- 新闻平台
- 电商平台
- 本地生活（POI）
- 内容消费（书籍 / 电影）

但现有研究通常只基于**单一数据集**（如 MovieLens、Amazon、MIND），无法真实反映用户行为的复杂性。

👉 本项目的核心目标是：

> **构建一个接近真实世界的、多域统一推荐数据环境，而非简单拼接数据集。**

---

## 🎯 核心问题（Research Questions）

### Q1：语义相近域是否可迁移？


电影（MovieLens） → 书籍（Book-Crossing / Goodreads）


研究内容：
- 用户兴趣是否跨内容类型迁移
- 长内容消费的统一建模能力

---

### Q2：行为机制是否可迁移？


新闻点击（MIND） → 短视频观看（KuaiRec）


研究内容：
- click vs watch 的差异
- sequential recommendation 泛化能力

---

### Q3：异构多域是否可统一建模？


Movie + Book + News + Video


研究内容：
- 单模型 vs 多专家模型
- MoE / LoRA 是否能自动学习域间差异

---

### Q4：数据混合方式是否影响模型结论？

对比：

- ❌ 随机混合
- ⚠️ 按比例混合
- ✅ 结构化混合（本工作）

---

### Q5：是否可替代真实用户数据？

研究：

- Synthetic → Real 泛化能力
- 是否可用于隐私推荐（端侧）

---

## 📊 数据集选择（Datasets）

### 1️⃣ 语义相近域（Semantic-close）

| 数据集 | 说明 |
|--|--|
| MovieLens | 电影评分（显式反馈） |
| Book-Crossing / Goodreads | 书籍交互 |

👉 用于研究：内容兴趣迁移

---

### 2️⃣ 行为机制相近域（Mechanism-close）

| 数据集 | 说明 |
|--|--|
| MIND | 新闻点击（impression + click） |
| KuaiRec | 短视频行为（观看/互动） |

👉 用于研究：序列行为迁移

---

### 3️⃣ 异构多域（Heterogeneous）

| 数据集 | 说明 |
|--|--|
| MovieLens | 长内容 |
| Book-Crossing / Goodreads | 长内容 |
| MIND | 信息流 |
| KuaiRec | 短视频 |

👉 用于模拟真实用户多平台行为

---

## 🧠 核心设计原则（Design Principles）

### 原则 1：不随机混合 ❌

随机混合数据集会导致：
- 无用户语义
- 无行为结构
- 无法解释模型行为

---

### 原则 2：差异可控（Controlled Heterogeneity）

| 类型 | 示例 |
|--|--|
| 内容语义 | 电影 vs 新闻 |
| 反馈机制 | rating vs click |
| 时序结构 | 静态 vs session |
| 侧信息 | ID-only vs text-rich |

---

### 原则 3：统一交互语义

统一为：

- `explicit_positive`
- `explicit_negative`
- `implicit_positive`
- `exposure_only`
- `consumption_depth`

---

### 原则 4：统一时间结构

支持：

- 静态（User-level）
- 序列（Session-level）

---

### 原则 5：构造伪用户（Pseudo-User）✅

跨域对齐依据：

- 活跃度
- 偏好分布
- 时间模式
- 内容 embedding

👉 构造“同一个用户”的多域行为

---

## 🏗️ 数据构建流程（Pipeline）

### Step 1：统一数据格式


user_id
item_id
domain
event_type
value
timestamp
session_id


---

### Step 2：用户特征提取

- 域分布
- 活跃度
- 时间模式
- 行为风格
- 内容 embedding

---

### Step 3：跨域用户对齐（核心）


Movie ↔ Book ↔ News ↔ Video


---

### Step 4：构造多域用户


User A = Movie + Book + News + Video


---

### Step 5：Session 构建

- 切分 session
- 限制跨域跳转
- 模拟真实行为

---

### Step 6：数据平衡

- 控制各域比例
- 控制交互规模
- 控制用户分布

---

## 🧪 实验设计（Experiments）

### Baseline

1. 单域训练  
2. 随机混合  
3. 按比例混合  

---

### 本方法

- 结构化多域构建（Pseudo-User）

---

## 📏 评价指标（Evaluation）

### 1️⃣ 数据合理性

- 域分布
- 活跃度分布
- session 长度
- 转移矩阵

---

### 2️⃣ 跨域一致性

- 用户 embedding 相似度
- 偏好一致性

---

### 3️⃣ 推荐效果

Train on Synthetic → Test on Real：

- Recall@K
- NDCG@K
- MRR

---

### 4️⃣ 模型行为（MoE）

- Router entropy
- Expert specialization
- Load balance

---

## 🎯 项目贡献（Contributions）

- ✅ 多域推荐 benchmark
- ✅ 结构化混合方法
- ✅ 跨域实验框架
- ✅ 支持 MoE / LoRA 研究

---

## 🔮 后续方向（Future Work）

- 引入 Amazon / Yelp
- 加入隐私机制（DP）
- LLM 用户模拟
- 端侧推荐系统（federated）

---

# ⭐ 一句话总结

> 我们不是在“混数据”，而是在构建一个**可控、可解释的多域推荐实验世界**。