# [NeurIPS'24 Oral] HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning
This repo contains the source code of [HydraLoRA](https://arxiv.org/abs/2404.19245). 

Discover the power of Asymmetric LoRA—achieving superb performance with significantly fewer parameters! 🌟 HydraLoRA features an asymmetric fine-tuning architecture capable of effectively identifying and adapting to the "intrinsic components" within the data—sub-domains or diverse tasks that may be challenging for domain experts to explicitly delineate.

<figure style="text-align:center">
  <img src="./figures/lora.png"  height="150">
</figure>

**Figure 1**: llustration of LoRA architecture changes in HydraLoRA. Only the tunable parameters
are shown in this Figure. (a) LoRA architecture with matrix A to achieve low rank and matrix B to recover. (b) under the same parameter count, a monolithic LoRA is splitted into multiple smaller A and B matrices to avoid training interference. (c) based on (b), HydraLoRA has an asymmetric structure that has a shared A matrix and multiple B matrices.

## 🚀 Updates
- 2024/12/01 ---  Added [MLLM_HydraLoRA](MLLM-HydraLoRA/README.md) version for Multimodal Large Language Model (Llava).

## 🔥 Motivation and Observation

### LoRA’s Practical Dilemma

Fine-tuning a small subset of parameters offers a streamlined approach for domain adaptation, it’s well-recognized that model performance is closely tied to the number of parameters involved. This intrinsic characteristic of methods like LoRA often results in them falling short of the FFT baseline, which updates all parameters, thereby creating a trade-off between efficiency and model quality. 

This issue of compromised quality in a low-parameter setting becomes even more pronounced in target domains characterized by complex sub-domains and diverse tasks. This situation presents a compelling research question:

**What is the optimal architecture that can deliver superior model performance while still capitalizing on the efficiency benefits of a reduced parameter footprint?**

<figure style="text-align:center">
  <img src="./figures/Heterogeneity.png"  height="150">
</figure>

**Figure 2**: The figure demostrates erformance impact of corpus heterogeneity on full fine-tuning vs. parameter-efficient fine-tuning. Heterogeneity signifies the diversity within the dataset, often leading to intereference due to its varied content and style. Parameter-efficient approaches are particularly sensitive, suffering greater performance losses in heterogeneous cases.

###  LoRA's Asymmetry

When multiple LoRA heads are trained individually on different data, the parameters of matrix A from different heads tend to converge, while those of matrix B are distinguishable.

<figure style="text-align:center">
  <img src="./figures/LoRA_breakdown.png" height="200">
</figure>

**Figure 3**: Breakdown analysis of LoRA modules. Consider LLaMA2-7B (random seed=42), which contains 32 decoder layers, corresponding to 32 adaptive modules. Each module consists of 0: q_proj_A, 1: q_proj_B, 2: v_proj_A, 3: v_proj_B submodules. This makes a total of 32 X 4 submodules. (a,b) left displays all submodules. (a,b) center shows all even submodules, i.e. the A matrix. (a,b) right represents all odd submodules, i.e. the B matrix. It can be seen that the differences in the fine-tuned LoRA modules for different tasks arise mainly from the B matrix.

## Workflow of HydraLoRA
<figure style="text-align:center">
  <img src="./figures/HydraLoRA.png"  height="250">
</figure>

**Figure 4**: Architecture and workflow of HydraLoRA. During the fine-tuning stage, HydraLoRA first adaptively identifies and initializes N of intrinsic components without specific domain knowledge. It then employs a trainable MoE router that treats each intrinsic component as an expert to automatically segregate training samples
into intrinsic components for fine-tuning. During the inference stage, HydraLoRA merges multiple B matrices flexibly and dynamically through a trained router.

**⚠️ Note:** 

1. MoE is not the core contribution of this paper; it is used here merely as a module fusion tool, and other fusion methods could also be considered.
2. K-means is used merely as a tool to determine the number N of B modules. Alternatively, N can be manually specified or derived using other methods, such as DBSCAN discussed in this paper.

**For more details please check out our paper.**

## 🛠️ Install

**Implementation Environment**: The model is implemented by using Pytorch. Using this command to implement your environment.

```
conda create -n hydralora python=3.10
conda activate hydralora
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```

**Dataset**: [Link](https://github.com/Clin0212/HydraLoRA/issues/1) Please note that the asymmetric structure of HydraLoRA is not limited to the datasets listed. You're welcome to use other datasets to explore its robustness.

## 🛠️ Project Structure
The source code is organized as below:

``` shell
|-- Motivation
    -- tesn_lora.py # analyzing the Lora modules
|-- HydraLoRA
    -- peft
    -- fine-tuning.py # main code for hydralora learning
```

## 🕹️ Quickstart
### **1. LoRA analysis**: 

```
bash motivation/tesn_lora.sh
```

### **2. HydraLoRA training**: 

**Single-GPU**

```
bash HydraLoRA/fine-tuning.sh
```

**DeepSpeed**

```
bash HydraLoRA/fine-tuning_dp.sh
```

[Extend to other models.](https://github.com/Clin0212/HydraLoRA/issues/11)

### **3. Evaluate:**

Use [opencompass](https://github.com/open-compass/opencompass/tree/main) for evaluation. 

In `opencompass/opencompass/models/huggingface.py`, add:

```
import sys
sys.path.insert(0, 'path_to_your_current_dir_containing_changed_peft&transformers')
```
In the config file `opencompass/configs/models/hf_llama/hf_llama2_7b.py`:

```
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='llama-2-7b-hf',
        path="path_to_base_model",
        tokenizer_path='path_to_tokenizer',
        peft_path='path_to_hydralora',
        ...
    )
]
```


For zero-shot, `opencompass/configs/datasets/mmlu/mmlu_ppl_ac766d.py` (line 89)

 ```
 retriever=dict(type=ZeroRetriever)
 ```


## ⭐ Citation

If you find our work helpful, please consider citing our paper:
```
@inproceedings{tian2024hydralora,
  title={HydraLoRA: An Asymmetric LoRA Architecture for Efficient Fine-Tuning},
  author={Tian, Chunlin and Shi, Zhan and Guo, Zhijiang and Li, Li and Xu, Chengzhong},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

## ❤️ References

The code refers to the repo [LoRAMoE](https://github.com/Ablustrund/LoRAMoE), [parameter-efficient-moe
](https://github.com/for-ai/parameter-efficient-moe), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [opencompass](https://github.com/open-compass/opencompass/tree/main).

## Local Experiment Delivery

This workspace also contains a cleaned local delivery for pseudo-user HydraLoRA experiments with Llama-3.2-1B.

### Retained Structure

```text
HydraLoRA/
  fine-tuning.py
  fine-tuning_pseudo_user.sh
  prepare_pseudo_user_sft.py
  eval_choice_accuracy.py
data/
  hydralora_pseudo_v2fix_all/
    train.json
    valid.json
deliveries/
  hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/
  eval_accuracy/
test_use_lora/output/
  hydralora_ml1m_llama32_1b/
  baseline_lora1_ml1m_llama32_1b/
```

Cleanup reports:

```text
cleanup_keep_list.md
cleanup_delete_list.md
cleanup_size_report.md
```

### Pseudo-user Training Result

Final balanced pseudo-user run:

```text
deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827
```

Source data:

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/dataset_human_like/final_data_use/data/data/final_versions/v2fix_all
```

Prepared data:

```text
data/hydralora_pseudo_v2fix_all/train.json
data/hydralora_pseudo_v2fix_all/valid.json
```

Training summary:

| Metric | Value |
|---|---:|
| Train examples | 95523 |
| Valid examples | 1949 |
| Total steps | 11940 |
| Final train loss | 0.3163 |
| Best eval loss | 0.3299 |
| Best checkpoint step | 11000 |

Best checkpoint:

```text
deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoint-11000
```

Latest checkpoint:

```text
deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoint-11940
```

Exposed best LoRA path:

```text
deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/sft_lora_model
```

Full run summary:

```text
deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/RESULTS.txt
```

### Accuracy Evaluation

Accuracy here means 4-choice next-item accuracy. For each prompt, the evaluator scores `A`, `B`, `C`, and `D`; the highest-scoring letter is the prediction.

Evaluation summary:

```text
deliveries/eval_accuracy/README.md
deliveries/eval_accuracy/SUMMARY.txt
```

Results:

| Dataset | Checkpoint | Examples | Accuracy |
|---|---|---:|---:|
| MovieLens | `test_use_lora/output/hydralora_ml1m_llama32_1b/checkpoint-53000` | 6028 | 87.29% |
| Pseudo-user balanced | `checkpoint-11000` | 1949 | 73.52% |

Pseudo-user accuracy by domain:

| Domain | Examples | Accuracy |
|---|---:|---:|
| goodreads | 621 | 86.47% |
| movielens | 626 | 81.79% |
| kuairec | 141 | 60.28% |
| mind | 561 | 53.30% |

### Reproduction Commands

Train pseudo-user balanced HydraLoRA:

```bash
cd /vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama
export CUDA_VISIBLE_DEVICES=1
export OUTPUT_ROOT=deliveries/reproduce_checkpoints
export EXP_NAME=hydralora_pseudo_v2fix_all_balanced_repro
export MAX_SAMPLES=0
export MAX_SAMPLES_PER_USER_DOMAIN=20
export NUM_TRAIN_EPOCHS=1
export MAX_STEPS=-1
bash HydraLoRA/fine-tuning_pseudo_user.sh
```

Evaluate pseudo-user accuracy:

```bash
CUDA_VISIBLE_DEVICES=1 /home/liushaokun/miniconda3/envs/hydralora/bin/python HydraLoRA/eval_choice_accuracy.py \
  --base-model /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --adapter deliveries/latest_balanced/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoint-11000 \
  --data data/hydralora_pseudo_v2fix_all/valid.json \
  --output deliveries/eval_accuracy/pseudo_balanced_ckpt11000_valid_accuracy.json \
  --group-field domain \
  --batch-size 4
```

Evaluate MovieLens comparison accuracy:

```bash
CUDA_VISIBLE_DEVICES=1 /home/liushaokun/miniconda3/envs/hydralora/bin/python HydraLoRA/eval_choice_accuracy.py \
  --base-model /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --adapter test_use_lora/output/hydralora_ml1m_llama32_1b/checkpoint-53000 \
  --data /vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/hydralora_ml1m/valid.json \
  --output deliveries/eval_accuracy/movielens_hydralora_ckpt53000_valid_accuracy.json \
  --group-field task_type \
  --batch-size 4
```

### Cleanup Summary

The workspace was cleaned to retain final deliverables and reproducibility artifacts.

Space usage:

```text
Before cleanup: 1.71 GiB
After cleanup:  469.46 MiB
Freed:          1.25 GiB
```

Deleted:

- HuggingFace tokenization caches: `train_512/`, `valid_512/`
- pseudo-user smoke data and smoke model outputs
- superseded 50k pilot delivery
- transient smoke evaluation JSON files
- stale pid/out files
- Python bytecode and empty directories

Kept as uncertain:

- `test_use_lora/output/baseline_lora1_ml1m_llama32_1b/`

It is not part of the final pseudo-user delivery, but it may still be useful as a historical MovieLens baseline.

### Follow-up Suggestions

- Run a held-out `test.json` evaluation if a separate test split is produced later; current reported accuracy uses `valid.json`.
- Investigate low pseudo-user domains first: `mind` and `kuairec`.
- If storage becomes tight again, confirm whether the baseline MovieLens directory is still needed, then archive or delete it.
