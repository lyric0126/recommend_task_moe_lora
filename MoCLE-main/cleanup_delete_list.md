# Cleanup Delete List

Generated: 2026-05-06

Cleanup scope:

`/vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main`

## Delete Large Failed Or Superseded Output Directories

These are intermediate, failed, debug, or superseded training runs. Final comparison files are already preserved in `deliveries/mocle_hydralora_standard_eval_20260503/`.

- `outputs/movielens1m_mocle_fullrun/` (~22G)
- `outputs/movielens1m_mocle_full_1epoch/` (~1.6G)
- `outputs/human_like_v2fix_all_mocle_full_20260501_230156/` (~837M)
- `outputs/mocle_hydralora_standard_full_20260503_001218/` (~503M)
- `outputs/mocle_hydralora_standard_hydraopt_20260503_014224/` (~272M)
- `outputs/movielens1m_mocle_full_5k/` (~223M)
- `outputs/movielens1m_mocle_20260422_010412/` (~110M)
- `outputs/movielens1m_mocle_100steps/` (~44M)
- `outputs/movielens1m_mocle_debug_5steps_20260501/` (~22M)
- `outputs/movielens1m_mocle_debug_5steps_20260501_train.log`
- `outputs/movielens1m_mocle_tb_verify/`
- `outputs/movielens1m_mocle_launch_check/`
- `outputs/movielens1m_mocle_live_20260422_005211/`
- `outputs/movielens1m_single_gpu_smoke/`
- `outputs/movielens1m_single_gpu_debug/`

## Delete Intermediate Checkpoints Inside The Final Run

Keep `checkpoint-step-11940` only. Delete these repeated intermediate checkpoints:

- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-1000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-2000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-3000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-4000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-5000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-6000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-7000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-8000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-9000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-10000/`
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11000/`

## Delete Cache And TensorBoard Temp Logs

- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/tb_logs/`
- `recommendation/movielens1m/__pycache__/`
- `peft-main/src/peft/__pycache__/`

## Delete Superseded Or Debug Data Generated Inside This Repo

Keep `data/hydralora_pseudo_v2fix_all_mocle_standard/` only.

- `data/human_like_v2fix_all_instruction/` (~1.8G; mismatch prompt/data experiment)
- `data/movielens1m_train_full/` (~684M; superseded MovieLens full data)
- `data/movielens1m_full/` (~684M; duplicate/superseded MovieLens full data)
- `data/movielens1m_train_debug/` (~737K; debug data)
- `data/movielens1m_smoke/` (~233K; smoke data)

## Move To Archive Instead Of Deleting

These are not deleted because they are small and useful for provenance:

- `README.md` original MoCLE README -> `archive_uncertain/README_original_mocle.md`
- `deliveries/mocle_eval_accuracy_20260502/` -> `archive_uncertain/mocle_eval_accuracy_20260502/`

## Empty Directories

After deletion, remove empty directories with:

`find . -type d -empty -delete`
