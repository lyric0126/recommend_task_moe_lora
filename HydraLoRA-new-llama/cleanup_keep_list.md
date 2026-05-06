# Cleanup Keep List

The following files and directories are retained.

## Source Code and Config

- `HydraLoRA/`
- `MLLM-HydraLoRA/`
- `Motivation/`
- `baseline-test-lsk/`
- `figures/`
- `environment.yml`
- `requirements.txt`
- `README.md`

Key scripts added or used for this experiment:

- `HydraLoRA/prepare_pseudo_user_sft.py`
- `HydraLoRA/fine-tuning_pseudo_user.sh`
- `HydraLoRA/eval_choice_accuracy.py`
- `HydraLoRA/fine-tuning.py`
- `HydraLoRA/build_dataset.py`

## Final Pseudo-user Data

- `data/hydralora_pseudo_v2fix_all/train.json`
- `data/hydralora_pseudo_v2fix_all/valid.json`

Rebuildable cache directories under this dataset are not kept:

- `data/hydralora_pseudo_v2fix_all/train_512/`
- `data/hydralora_pseudo_v2fix_all/valid_512/`

## Final Pseudo-user Delivery

Final balanced run:

```text
deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/
```

Important retained paths:

- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/README.txt`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/RESULTS.txt`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/run_train.sh`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/run_tensorboard.sh`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/logs/train.log`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/logs/tensorboard.log`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoint-11000/`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoint-11940/`
- `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/checkpoints/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/sft_lora_model`

Symlinks retained:

- `deliveries/latest`
- `deliveries/latest_balanced`

## Accuracy Evaluation Deliverables

- `deliveries/eval_accuracy/README.md`
- `deliveries/eval_accuracy/SUMMARY.txt`
- `deliveries/eval_accuracy/movielens_hydralora_ckpt53000_valid_accuracy.json`
- `deliveries/eval_accuracy/movielens_hydralora_ckpt53000_valid_accuracy.log`
- `deliveries/eval_accuracy/pseudo_balanced_ckpt11000_valid_accuracy.json`
- `deliveries/eval_accuracy/pseudo_balanced_ckpt11000_valid_accuracy.log`

## MovieLens Comparison Artifacts

Kept because they were used to compare pseudo-user accuracy and loss:

- `test_use_lora/output/hydralora_ml1m_llama32_1b/`

## Uncertain Artifacts Kept

Kept because their ownership or future use is uncertain:

- `test_use_lora/output/baseline_lora1_ml1m_llama32_1b/`

If more aggressive cleanup is desired later, this can be archived or deleted after confirming it is not needed.
