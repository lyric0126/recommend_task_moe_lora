# Cleanup Keep List

Generated: 2026-05-06

Cleanup scope:

`/vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main`

## Final Deliverables

Keep these as the final handoff artifacts:

- `deliveries/mocle_hydralora_standard_eval_20260503/`
  - `README.md`
  - `SUMMARY.txt`
  - `HYDRALORA_EVAL_SUMMARY.txt`
  - `mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.json`
  - `mocle_hydraopt_shuffle_ckpt11940_valid_accuracy.log`
  - `mocle_hydraopt_shuffle_train_summary.json`
  - ablation eval JSON/log files used by the final comparison
  - `train_single_gpu.py`
  - `eval_mocle_choice_accuracy.py`
  - `standard_data_build_stats.json`

## Final Checkpoint

Keep only the final selected MoCLE checkpoint:

- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940/`
  - `expert_0/adapter_model.bin`
  - `expert_1/adapter_model.bin`
  - `expert_2/adapter_model.bin`
  - `expert_3/adapter_model.bin`
  - adapter configs
  - tokenizer files

Keep final run metadata needed for reproducibility:

- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/train_summary.json`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/train.log`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/train_metrics.jsonl`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/run_env.txt`

## Final Data

Keep the standardized data used for the same-standard train/eval:

- `data/hydralora_pseudo_v2fix_all_mocle_standard/train.json`
- `data/hydralora_pseudo_v2fix_all_mocle_standard/valid.json`
- `data/hydralora_pseudo_v2fix_all_mocle_standard/build_stats.json`

Original upstream HydraLoRA source data is outside this cleanup scope:

- `/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/data/hydralora_pseudo_v2fix_all/train.json`
- `/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama/data/hydralora_pseudo_v2fix_all/valid.json`

## Source Code And Config

Keep repository source and key scripts:

- `recommendation/`
  - `movielens1m/train_single_gpu.py`
  - `movielens1m/eval_mocle_choice_accuracy.py`
  - dataset/preprocess/check scripts
  - local READMEs
- `peft-main/`
- `images/`
- root project metadata and README

## Archived Uncertain Items

Move these instead of deleting because they are small and may be useful for traceability:

- original root `README.md` before cleanup rewrite
- `deliveries/mocle_eval_accuracy_20260502/` old mismatch evaluation delivery

## Not Present In This Project

The scan did not find final engine, ONNX, benchmark, correctness report, or mesh result files under this project tree.
