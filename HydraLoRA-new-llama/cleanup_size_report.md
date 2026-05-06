# Cleanup Size Report

Generated for:

```text
/vepfs-cnbja62d5d769987/liushaokun/sys_work/HydraLoRA-new-llama
```

## Pre-cleanup Size

Total project size before cleanup:

```text
1.8G .
```

## Largest Directories

| Path | Size | Notes |
|---|---:|---|
| `data/` | 938M | Prepared pseudo-user data and HF tokenization caches |
| `data/hydralora_pseudo_v2fix_all/` | 906M | Final pseudo-user JSON plus generated `*_512` cache |
| `data/hydralora_pseudo_v2fix_all/train_512/` | 769M | Rebuildable HuggingFace tokenization cache |
| `test_use_lora/output/` | 560M | Prior MovieLens checkpoints plus pseudo-user smoke runs |
| `deliveries/` | 222M | Final balanced run, pilot run, evaluation reports |
| `test_use_lora/output/hydralora_ml1m_llama32_1b/` | 116M | MovieLens HydraLoRA comparison checkpoint |
| `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/` | 110M | Final balanced pseudo-user delivery |
| `deliveries/hydralora_pseudo_v2fix_all_llama32_1b_20260501_134211/` | 109M | Superseded 50k pilot delivery |
| `test_use_lora/output/hydralora_pseudo_v2fix_all_*smoke*` | 364M total | Smoke/debug runs, not final |
| `test_use_lora/output/baseline_lora1_ml1m_llama32_1b/` | 84M | Prior baseline comparison, kept as uncertain |

## Largest Files

| Path | Size | Decision |
|---|---:|---|
| `data/hydralora_pseudo_v2fix_all/train_512/.../cache-*.arrow` | 347M | Delete, rebuildable cache |
| `data/hydralora_pseudo_v2fix_all/train_512/train/data-00000-of-00001.arrow` | 347M | Delete, rebuildable cache |
| `data/hydralora_pseudo_v2fix_all/train.json` | 119M | Keep, prepared final SFT train data |
| `data/hydralora_pseudo_v2fix_all/train_512/.../json-train.arrow` | 111M | Delete, rebuildable cache |
| `data/hydralora_pseudo_v2fix_all/valid.json` | 2.5M | Keep, final validation/evaluation data |
| `deliveries/eval_accuracy/movielens_hydralora_ckpt53000_valid_accuracy.json` | 2.7M | Keep, final accuracy report |
| `deliveries/eval_accuracy/pseudo_balanced_ckpt11000_valid_accuracy.json` | 888K | Keep, final accuracy report |

## Expected Space Recovery

Expected recoverable space from planned safe deletions:

```text
~1.3G
```

Main contributors:

- `data/hydralora_pseudo_v2fix_all/train_512/`: ~769M
- `data/hydralora_pseudo_v2fix_all/valid_512/`: ~16M
- `data/hydralora_pseudo_v2fix_all_smoke/`: ~32M
- pseudo-user smoke runs under `test_use_lora/output/`: ~364M
- superseded 50k pilot delivery: ~109M
- Python bytecode and stale pid/out files: small

## Post-cleanup Size

Cleanup completed.

```text
Before: 1.71 GiB
After:  469.46 MiB
Freed:  1.25 GiB
```

Largest directories after cleanup:

| Path | Size | Notes |
|---|---:|---|
| `test_use_lora/output/` | 200M | MovieLens comparison and baseline uncertain artifact |
| `data/hydralora_pseudo_v2fix_all/` | 122M | Final prepared train/valid JSON only |
| `deliveries/` | 114M | Final balanced run and accuracy deliverables |
| `deliveries/hydralora_pseudo_v2fix_all_balanced_97k_20260501_154827/` | 110M | Final balanced pseudo-user delivery |
| `HydraLoRA/` | 38M | Source code and vendored local modules |

Removed successfully:

- `data/hydralora_pseudo_v2fix_all/train_512/`
- `data/hydralora_pseudo_v2fix_all/valid_512/`
- `data/hydralora_pseudo_v2fix_all_smoke/`
- pseudo-user smoke output directories under `test_use_lora/output/`
- superseded 50k pilot delivery
- stale pid/out files
- Python `__pycache__/` directories
- empty directories left by cleanup

## Uncertain Large Items Kept

These are not deleted because they may still be useful for comparison or reproduction:

- `test_use_lora/output/hydralora_ml1m_llama32_1b/`: MovieLens comparison checkpoint used in accuracy evaluation.
- `test_use_lora/output/baseline_lora1_ml1m_llama32_1b/`: baseline comparison run; not final for pseudo-user, but may be useful for historical comparison.
- `HydraLoRA/transformers_bak/`: vendored source backup, kept to avoid breaking local code assumptions.
