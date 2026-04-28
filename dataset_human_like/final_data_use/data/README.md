# Pseudo-User Synthesis Pipeline

This repository contains a runnable pseudo-user synthesis pipeline and downstream recommendation experiment suite for four domains:

- MovieLens
- Goodreads
- MIND
- KuaiRec

The current best data construction version is **V2-fix**. V3 adds downstream recommendation experiments to test whether the pseudo-user data helps lightweight recommenders.

## Current Status

The project has completed four major phases:

| Version | Purpose | Status | Main Result |
| --- | --- | --- | --- |
| V1 | Initial pseudo-user pipeline | Complete | Working end-to-end baseline |
| V2 | Improve storage, representation, profiles, matching, synthesis, evaluation | Complete | More coverage, but lower quality |
| V2-fix | Quality repair of V2 | Complete | Best pseudo-user construction quality |
| V3 | Downstream recommendation validation | Complete | Lightweight baselines did not benefit reliably |

## Key Results

### Pseudo-User Quality

V2-fix is the best pseudo-user construction version by consistency metrics.

| Method | Global Consistency |
| --- | ---: |
| V2 | 0.316953 |
| Random Mix | 0.395468 |
| V1 | 0.453148 |
| V2-fix | 0.680195 |

V2-fix ablation:

| Score Component | Value |
| --- | ---: |
| semantic-only | 0.470436 |
| activity+temporal | 0.663218 |
| full method | 0.674013 |

V2-fix confidence distribution:

| Confidence | Count |
| --- | ---: |
| strict | 1211 |
| medium | 331 |
| loose | 0 |

V2-fix output scale:

| Output | Count |
| --- | ---: |
| pseudo users | 1542 |
| pseudo interactions | 298777 |

### Downstream Recommendation Results

V3 evaluates whether V2-fix pseudo-user data improves lightweight target-domain recommendation.

The answer is currently **no** for the implemented lightweight baselines.

Main item-item Recall@10:

| Target Domain | Single-domain | Random-mix | V2-fix pseudo-user |
| --- | ---: | ---: | ---: |
| Goodreads | 0.054505 | 0.050056 | 0.044494 |
| MovieLens | 0.083952 | 0.082239 | 0.079383 |

Interpretation:

- V2-fix clearly improves pseudo-user construction quality.
- However, the current V3 lightweight baselines only use target-domain interactions.
- They do not fully exploit the cross-domain structure inside pseudo users.
- As a result, pseudo-user augmentation behaves mostly like extra target-domain co-occurrence data and can dilute the original target-domain signal.

## Methods

## V1 Pipeline

V1 implements the initial full pseudo-user synthesis pipeline:

1. Dataset loaders for MovieLens, Goodreads, MIND, and KuaiRec.
2. Canonical interaction/item schema.
3. Cleaning and standardization.
4. Deterministic fallback item embeddings.
5. User profiles.
6. Two-domain matching.
7. MovieLens-anchored pseudo-user synthesis.
8. Minimal evaluation.

Canonical interaction fields include:

- `dataset`
- `user_id`
- `item_id`
- `timestamp`
- `raw_event`
- `event_value`
- `item_text`
- `item_category`

## V2 Pipeline

V2 tried to improve:

- storage layer
- item representation
- user profile quality
- matching quality
- pseudo-user synthesis
- evaluation

But V2 expanded coverage too aggressively:

- V1 pseudo users: 1751, all 4-domain
- V2 pseudo users: 5253, split across 2-domain, 3-domain, and 4-domain

V2 also had weak semantic scoring:

- `semantic-only = 0.101169`
- `activity+temporal = 0.533362`
- `full = 0.304942`

All V2 pseudo users fell into `loose`, so V2 was judged lower quality than V1.

## V2-fix Pipeline

V2-fix does not add new product scope. It fixes V2 quality issues.

Main changes:

1. Stronger fallback semantic representation:
   - cleaner item text
   - lowercasing
   - stopword filtering
   - TF-IDF-style weighting
   - feature pruning
   - 128-dimensional hashing fallback
   - broad topic bridge tokens

2. Better user semantic profiles:
   - long-term semantic profile
   - recent semantic profile
   - topic entropy/diversity
   - structured activity, temporal, and behavior fields

3. Stricter matching:
   - activity bucket
   - temporal bucket
   - semantic coarse cluster
   - smaller fallback candidate pool
   - calibrated strict/medium/loose thresholds
   - weights shifted toward robust activity, temporal, and behavior signals

4. Stricter synthesis:
   - medium+ component filtering
   - global consistency filtering
   - no forced 4-domain coverage
   - interaction cap per source user

V2-fix is the current main pseudo-user construction version.

## V3 Downstream Experiments

V3 fixes V2-fix as the data construction version and runs downstream recommendation experiments.

Target domains:

- MovieLens
- Goodreads

Training variants:

1. `single-domain`: target-domain train interactions only.
2. `random-mix`: target-domain train plus random target-domain augmentation.
3. `pseudo-user_v2fix`: target-domain train plus V2-fix pseudo-user target-domain augmentation.

Split:

- leave-last-2-out per user
- train: all but last two interactions
- validation: second-to-last interaction
- test: last interaction

Baselines:

- popularity / frequency
- pure Python item-item co-occurrence

Metrics:

- Recall@10
- NDCG@10
- HitRate@10
- MRR@10

V3 result:

- V2-fix pseudo-user data did not beat single-domain or random-mix under the current lightweight baselines.
- This is likely a model/task limitation rather than direct evidence that V2-fix pseudo users are useless.
- The baselines do not consume full cross-domain pseudo-user histories.

## Important Files

### Configs

- `configs/pseudo_user_pipeline.yaml`: V1 config
- `configs/pseudo_user_pipeline_v2.yaml`: V2 config
- `configs/pseudo_user_pipeline_v2fix.yaml`: V2-fix config
- `configs/pseudo_user_experiments_v3.yaml`: V3 experiment config

### Source Code

- `src/run_pipeline.py`: main staged runner
- `src/io_utils.py`: table IO, logs, checkpoints
- `src/loaders/datasets.py`: four-domain loaders
- `src/normalize/cleaning.py`: cleaning and standardization
- `src/features/embeddings_v2fix.py`: V2-fix item embeddings
- `src/features/profiles_v2fix.py`: V2-fix user profiles
- `src/matching/matcher_v2fix.py`: V2-fix matching
- `src/synthesis/pseudo_users_v2fix.py`: V2-fix synthesis
- `src/evaluation/evaluate_v2fix.py`: V2-fix evaluation
- `src/experiments/`: V3 downstream experiments

### Main Outputs

- `data/processed/item_embeddings_v2fix.parquet`
- `data/processed/user_profiles_v2fix.parquet`
- `data/processed/matches_v2fix_*.parquet`
- `data/processed/pseudo_user_metadata_v2fix.parquet`
- `data/processed/pseudo_user_interactions_v2fix.parquet`
- `data/processed/eval_summary_v2fix.json`
- `data/processed/exp_results_v3.json`
- `data/processed/exp_ablation_v3.json`

### Reports

- `reports/v2fix_plan.md`
- `reports/pseudo_user_eval_v2fix.md`
- `reports/v3_experiment_plan.md`
- `reports/v3_main_results.md`
- `reports/v3_ablation.md`
- `reports/v3_final_summary.md`

## How To Run

Run V2-fix pseudo-user construction:

```bash
python3 src/run_pipeline.py --v2fix-all --config configs/pseudo_user_pipeline_v2fix.yaml
```

Run V3 downstream experiments:

```bash
python3 src/run_pipeline.py --v3-all --config configs/pseudo_user_experiments_v3.yaml
```

Run a single stage:

```bash
python3 src/run_pipeline.py --v3-stage 4 --config configs/pseudo_user_experiments_v3.yaml
```

## Storage Note

The current environment does not provide `pyarrow`, `pandas`, or `fastparquet`.

Therefore files with `.parquet` suffix are currently written as JSONL fallback files with a metadata header. The IO layer auto-detects parquet support and will write true parquet if the environment later provides a parquet backend.

## Limitations

1. No heavy semantic embedding backend is available in the current environment.
2. Current V3 baselines are intentionally lightweight and do not model full cross-domain pseudo-user structure.
3. V3 only validates MovieLens and Goodreads as target domains.
4. V3 augmentation uses target-domain pseudo interactions, not a true multi-domain recommender.
5. The downstream negative result should be interpreted as: simple target-domain popularity/item-item baselines do not benefit from this augmentation.

## Recommended Next Steps

1. Keep V2-fix as the current best data construction version.
2. Add a recommender that can consume multi-domain pseudo-user histories directly.
3. Try a two-tower retrieval or sequence-based model using all domains in each pseudo user.
4. Add a stronger semantic backend when dependencies allow.
5. Enable true parquet storage for faster analysis and lower disk overhead.

