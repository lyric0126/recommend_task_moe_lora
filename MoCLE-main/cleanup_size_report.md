# Cleanup Size Report

Generated: 2026-05-06

Cleanup scope:

`/vepfs-cnbja62d5d769987/liushaokun/sys_work/MoCLE-main`

## Before Cleanup

Project size before cleanup:

- `.`: ~29G (`du -sb`: 30,503,819,098 bytes)

Top-level directory sizes:

- `outputs/`: ~26G
- `data/`: ~3.3G
- `peft-main/`: ~8.2M
- `deliveries/`: ~5.7M
- `images/`: ~913K
- `recommendation/`: ~173K

Largest directories/files found:

- `outputs/movielens1m_mocle_fullrun/`: ~22G
- `data/human_like_v2fix_all_instruction/train.jsonl`: ~1.76G
- `outputs/movielens1m_mocle_full_1epoch/`: ~1.6G
- `outputs/human_like_v2fix_all_mocle_full_20260501_230156/`: ~837M
- `data/movielens1m_train_full/train.json`: ~669M
- `data/movielens1m_full/train.json`: ~669M
- `outputs/mocle_hydralora_standard_full_20260503_001218/`: ~503M
- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/`: ~272M
- `outputs/mocle_hydralora_standard_hydraopt_20260503_014224/`: ~272M
- `data/hydralora_pseudo_v2fix_all_mocle_standard/train.json`: ~120.7M

## Planned Keep Size

Expected retained large artifacts:

- final checkpoint `checkpoint-step-11940/`: ~22M
- final run logs and metadata: ~3.8M plus `train.log`
- standardized data: ~124M
- final delivery reports and eval JSON/logs: ~4.8M
- source tree, PEFT fork, images, READMEs: ~10M

Expected post-cleanup project size:

- roughly 170M to 220M, depending on filesystem block accounting and archived small files

## Planned Space Release

Estimated space released:

- roughly 28.8G to 29.0G

Most of the release comes from:

- deleting old `outputs/` experiment directories
- deleting repeated intermediate checkpoints in the final run
- deleting superseded generated data
- deleting cache and TensorBoard temp logs

## Safety Notes

- Final checkpoint is explicitly excluded from deletion.
- Final delivery directory is explicitly excluded from deletion.
- Source code and key scripts are explicitly excluded from deletion.
- Uncertain small provenance items are moved to `archive_uncertain/` instead of deleted.
- No engine, ONNX, benchmark, correctness report, or mesh result files were found in this project tree during scan.

## After Cleanup

Project size after cleanup:

- `.`: ~164M (`du -sb`: 171,147,386 bytes)

Top retained directory sizes:

- `data/`: ~124M
- `outputs/`: ~26M
- `peft-main/`: ~8.1M
- `deliveries/`: ~4.8M
- `archive_uncertain/`: ~988K
- `images/`: ~913K
- `recommendation/`: ~123K

Retained final artifacts:

- `outputs/mocle_hydralora_standard_hydraopt_shuffle_20260503_031729/checkpoint-step-11940/`: ~22M
- `data/hydralora_pseudo_v2fix_all_mocle_standard/`: ~124M
- `deliveries/mocle_hydralora_standard_eval_20260503/`: ~4.8M

Actual space released:

- 30,332,671,712 bytes
- approximately 28.25 GiB
- approximately 30.33 GB
