# HydraLoRA Pseudo-user and MovieLens Evaluation

This folder contains the 4-choice accuracy evaluation for the HydraLoRA Llama-3.2-1B runs.

## Files

- `SUMMARY.txt`: compact text summary of the evaluation results.
- `movielens_hydralora_ckpt53000_valid_accuracy.json`: full MovieLens validation predictions and scores.
- `movielens_hydralora_ckpt53000_valid_accuracy.log`: MovieLens evaluation log.
- `pseudo_balanced_ckpt11000_valid_accuracy.json`: full pseudo-user validation predictions and scores.
- `pseudo_balanced_ckpt11000_valid_accuracy.log`: pseudo-user evaluation log.

## Training Run

Pseudo-user balanced training delivery:

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827
```

Source data:

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/dataset_human_like/final_data_use/data/data/final_versions/v2fix_all
```

Prepared SFT data:

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/data/hydralora_pseudo_v2fix_all
```

Training size:

| Split | Examples |
|---|---:|
| Train | 95523 |
| Valid | 1949 |

Training metrics:

| Metric | Value |
|---|---:|
| Total steps | 11940 |
| Final train loss | 0.3163 |
| Best eval loss | 0.3299 |
| Best eval step | 11000 |
| Epoch | 1.0 |

Best pseudo-user checkpoint:

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoint-11000
```

Exposed best LoRA model path:

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/sft_lora_model
```

TensorBoard:

```text
http://172.31.39.33:6008/
```

## Evaluation Method

The evaluator scores each candidate answer `A`, `B`, `C`, and `D` using the same validation prompt format used during SFT:

```text
{instruction}</s>{answer}<eos>
```

For each example, the prediction is the candidate with the highest log probability.

Two accuracy variants are saved:

- `accuracy_full_target`: scores `letter + EOS`.
- `accuracy_letter_only`: scores only the first answer letter.

The main reported number below uses `accuracy_full_target`.

Evaluation script:

```text
HydraLoRA/eval_choice_accuracy.py
```

## Results

| Dataset | Model / checkpoint | Validation examples | Accuracy |
|---|---|---:|---:|
| MovieLens | `test_use_lora/output/hydralora_ml1m_llama32_1b/checkpoint-53000` | 6028 | 87.29% |
| Pseudo-user balanced | `checkpoint-11000` | 1949 | 73.52% |

MovieLens is still clearly easier than the pseudo-user mixture. This is consistent with the loss gap: the pseudo-user data is multi-domain and noisier, while MovieLens is a single-domain recommendation task with stronger regularity.

## Pseudo-user Accuracy by Domain

| Domain | Examples | Accuracy |
|---|---:|---:|
| goodreads | 621 | 86.47% |
| movielens | 626 | 81.79% |
| kuairec | 141 | 60.28% |
| mind | 561 | 53.30% |

Main observation: pseudo-user performance is not uniformly low. `goodreads` and `movielens` are relatively strong, while `mind` and `kuairec` pull down the overall average.

## MovieLens Accuracy by Task Type

| Task type | Examples | Accuracy |
|---:|---:|---:|
| 0 | 1457 | 91.08% |
| 1 | 275 | 88.73% |
| 2 | 263 | 92.02% |
| 3 | 108 | 84.26% |
| 4 | 1608 | 85.45% |
| 5 | 273 | 90.48% |
| 6 | 63 | 66.67% |
| 7 | 1317 | 84.66% |
| 8 | 9 | 88.89% |
| 9 | 63 | 84.13% |
| 10 | 193 | 87.05% |
| 11 | 51 | 84.31% |
| 12 | 61 | 90.16% |
| 13 | 22 | 72.73% |
| 14 | 58 | 89.66% |
| 15 | 147 | 88.44% |
| 16 | 9 | 88.89% |
| 17 | 51 | 92.16% |

## Reproduction Commands

MovieLens:

```bash
CUDA_VISIBLE_DEVICES=1 /home/liushaokun/miniconda3/envs/hydralora/bin/python HydraLoRA/eval_choice_accuracy.py \
  --base-model /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --adapter test_use_lora/output/hydralora_ml1m_llama32_1b/checkpoint-53000 \
  --data /vepfs-cnbja62d5d769987/liushaokun/sys_work/test_use_lsk/lora_trl/data/hydralora_ml1m/valid.json \
  --output deliveries/eval_accuracy/movielens_hydralora_ckpt53000_valid_accuracy.json \
  --group-field task_type \
  --batch-size 4
```

Pseudo-user:

```bash
CUDA_VISIBLE_DEVICES=2 /home/liushaokun/miniconda3/envs/hydralora/bin/python HydraLoRA/eval_choice_accuracy.py \
  --base-model /vepfs-cnbja62d5d769987/liushaokun/models/Llama-3.2-1B \
  --adapter deliveries/latest_balanced/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoint-11000 \
  --data data/hydralora_pseudo_v2fix_all/valid.json \
  --output deliveries/eval_accuracy/pseudo_balanced_ckpt11000_valid_accuracy.json \
  --group-field domain \
  --batch-size 4
```

## Notes

- No separate `test.json` was found for these two runs, so this evaluation uses `valid.json`.
- The pseudo-user training completed all steps. The final Trainer cleanup hit a local PEFT compatibility issue at `load_best_model_at_end`; the best checkpoint was recovered and exposed as `sft_lora_model`.
- The HydraLoRA structure used `lora_nums=4`, `lora_rank=8`, `lora_alpha=16`, and target modules `q_proj,k_proj,v_proj,o_proj`.
