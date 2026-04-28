# Pseudo-User Synthesis Pipeline Plan

## Repository Scan

- Raw datasets are present under `raw/`:
  - `raw/movielens/ml-25m`
  - `raw/goodreads`
  - `raw/mind/MINDlarge_train`
  - `raw/kuairec/KuaiRec 2.0/data`
- No existing `src/`, `configs/`, `data/interim`, `data/interim_clean`, or `data/processed` pipeline implementation was found.
- The available Python runtime for this pipeline is `python3`.
- Heavy data libraries (`pandas`, `numpy`, `pyarrow`, `sklearn`) are not installed. The pipeline will use Python standard library implementations.

## Format Strategy

The requested outputs use `.parquet` paths. Because `pyarrow`/`fastparquet` are unavailable, the pipeline will write deterministic JSONL records to those paths using `src/io_utils.py`. Each file begins with a metadata record marking `storage_format=jsonl_fallback`. This keeps the requested file layout stable and runnable in the current environment.

## Directory Plan

- `configs/pseudo_user_pipeline.yaml`: pipeline parameters and raw paths.
- `src/schema.py`: shared canonical schema.
- `src/io_utils.py`: logging, checkpoint, JSONL fallback, config loading.
- `src/run_pipeline.py`: staged command runner.
- `src/loaders/`: dataset-specific adapters.
- `src/normalize/`: cleaning and standardization.
- `src/features/`: deterministic embeddings and user profiles.
- `src/matching/`: blocking, candidate generation, scoring.
- `src/synthesis/`: pseudo-user synthesis.
- `src/evaluation/`: minimal evaluation and baseline.
- `data/interim/`: loader outputs.
- `data/interim_clean/`: cleaned domain outputs.
- `data/processed/`: embeddings, profiles, matches, pseudo users, evaluation.
- `reports/`: summaries and checkpoints.
- `logs/`: per-stage logs.

## Stage Plan

1. Stage 0: write this plan plus log/checkpoint and verify they exist.
2. Stage 1: create code skeleton, config, schema, IO helpers, CLI runner.
3. Stage 2: implement four loaders and write standardized interactions/items.
4. Stage 3: clean/filter domains and normalize time/text/category.
5. Stage 4: generate deterministic item embeddings from item text.
6. Stage 5: create user semantic/activity/temporal/behavior profiles.
7. Stage 6: match MovieLens-Goodreads and MIND-KuaiRec users.
8. Stage 7: synthesize pseudo users anchored on MovieLens.
9. Stage 8: run minimal evaluation against Random Mix baseline.

## Smoke Test Policy

Every stage writes:

- `logs/stage_N.log`
- `reports/checkpoint_stage_N.md`

Every stage smoke test checks the expected stage artifacts and writes a concise result into both files.
