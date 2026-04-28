# V2 Pseudo-User Pipeline Plan

## Current State

V1 is runnable and has produced:

- Cleaned domain interactions/items under `data/interim_clean/`.
- Item embeddings under `data/processed/item_embeddings.parquet`.
- User profiles under `data/processed/user_profiles.parquet`.
- Pairwise matches under `data/processed/matches_*.parquet`.
- Pseudo-user metadata/interactions under `data/processed/`.
- Evaluation under `data/processed/eval_summary.json` and `reports/pseudo_user_eval.md`.

The runtime currently lacks `pandas`, `pyarrow`, `fastparquet`, `numpy`, `sklearn`, and `sentence_transformers`; `yaml` is available.

## Modules To Keep

- `src/run_pipeline.py`: keep the main staged runner and add V2 stages.
- `src/io_utils.py`: keep public `write_table/read_table` API and upgrade internals.
- `src/loaders/datasets.py`: keep V1 loader outputs as upstream inputs for V2.
- `src/normalize/cleaning.py`: keep cleaned V1 canonical tables as V2 base.
- `src/features/embeddings.py` and `src/features/profiles.py`: keep V1 functions and add V2 variants.
- `src/matching/matcher.py`: keep V1 pair scoring and add configurable V2 matching.
- `src/synthesis/pseudo_users.py`: keep V1 synthesis and add V2 synthesis.
- `src/evaluation/evaluate.py`: keep V1 evaluation and add V2 evaluation.

## Modules To Enhance

- Storage: auto-detect true parquet support; use parquet when possible and JSONL fallback otherwise with clearer metadata.
- Item representation: richer item text, configurable embedding backend, stronger fallback using TF-IDF-style hashed weighting.
- User profiles: add recency-aware semantic vectors, entropy/diversity, structured activity/temporal/behavior fields.
- Matching: add configurable weights, thresholds, optional semantic coarse block, ablation scores, and retained candidates.
- Synthesis: support 2/3/4-domain pseudo users and improved global consistency instead of forcing 4-domain coverage.
- Evaluation: add V1 vs V2 comparison, ablation, confidence-level metrics, and richer coverage/reuse reporting.

## V2 Outputs

- `configs/pseudo_user_pipeline_v2.yaml`
- `data/processed/item_embeddings_v2.parquet`
- `data/processed/user_profiles_v2.parquet`
- `data/processed/matches_v2_*.parquet`
- `data/processed/pseudo_user_metadata_v2.parquet`
- `data/processed/pseudo_user_interactions_v2.parquet`
- `data/processed/eval_summary_v2.json`
- `reports/v2_*.md`
- `logs/stage_v2_N.log`
- `reports/checkpoint_v2_stage_N.md`

## Execution Policy

Each V2 stage will write outputs, run a smoke test, and write both log and checkpoint files. If any stage fails, execution stops at the failure point.
