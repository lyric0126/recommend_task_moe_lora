# MovieLens-1M + TRL LoRA 训练说明

这个 README 记录当前的最小运行流程：

1. 先做数据预处理  
2. 再进行 TRL LoRA 训练  
3. 最后做评测  

---

## 1. 预处理

先运行：

```bash
python preprocess_ml1m.py
```

这一步会先把 MovieLens-1M 数据处理成后续训练和评测可直接读取的格式。

---

## 2. 训练

然后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python train_trl_lora_ml1m.py
```

如果你的机器只有一张卡，这样写就可以。  
如果你有多张卡，也可以把 `0` 改成你想用的 GPU 编号。

---

## 3. 评测

训练完成后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python eval_trl_lora_ml1m.py
```

这一步用于验证训练后的 LoRA 效果。

---

## 4. 如果显存不够

先改小训练脚本中的这几项参数：

```python
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 8
max_seq_length = 384
```

这通常是最直接的降显存方式。

如果这样还是爆显存，再把 LoRA 注入模块缩小为：

```python
target_modules = ["q_proj", "v_proj"]
```

这样会进一步减少训练开销。

---

## 5. 后面如果换成 Llama 3.2 1B

只需要改两个脚本里的模型路径：

```python
MODEL_NAME = "/你的本地llama3.2-1b路径"
```

也就是把原来的模型路径替换成你本地的 Llama 3.2 1B 路径即可。

如果换成 Llama，LoRA 注入模块通常仍然可以先用这一组：

```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

一般先这样就够用了。

---

## 6. 你现在最先要做的事

先跑下面这条命令：

```bash
python preprocess_ml1m.py
```

等预处理完成后，再继续训练和评测。

---

## 7. 推荐的完整执行顺序

```bash
python preprocess_ml1m.py

CUDA_VISIBLE_DEVICES=0 python train_trl_lora_ml1m.py

CUDA_VISIBLE_DEVICES=0 python eval_trl_lora_ml1m.py
```

---

## 8. 备注

如果你后面要迁移到 Llama 3.2 1B，优先检查两件事：

1. `MODEL_NAME` 是否已经改成正确的本地模型路径  
2. `target_modules` 是否和当前模型结构匹配  

如果训练时报显存不足，优先先缩 batch size 和 sequence length，再考虑缩 LoRA 注入模块。
