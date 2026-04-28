# V3 Downstream Recommendation Experiment Plan

## Fixed Data Construction Version

V3 fixes `V2-fix` as the main pseudo-user construction version. V3 does not alter pseudo-user construction logic.

## Goal

Validate whether high-quality pseudo-user data improves lightweight downstream recommendation compared with:

1. `single-domain`: target-domain train interactions only.
2. `random-mix`: target-domain train interactions plus randomly selected pseudo-style target-domain augmentation.
3. `pseudo-user(V2-fix)`: target-domain train interactions plus V2-fix pseudo-user target-domain augmentation.

## Target Domains

Primary targets:

- MovieLens
- Goodreads

Optional targets, not required for V3 completion:

- MIND
- KuaiRec

## Split Design

For each target domain, use a leave-last-2-out split per user:

- train: all but the last two interactions by timestamp.
- val: second-to-last interaction.
- test: last interaction.

The test set is fixed and reused by all training variants to ensure fair comparison.

## Training Set Construction

For each target domain:

- `single-domain`: original target-domain train split.
- `random-mix`: single-domain train plus random target-domain pseudo-style augmentation with the same rough scale as V2-fix augmentation.
- `pseudo-user_v2fix`: single-domain train plus V2-fix pseudo-user interactions in the target domain.

All augmentation filters out held-out validation/test user-item pairs using each source user's original id to reduce leakage.

## Baselines

Use lightweight, interpretable baselines:

- Popularity/frequency baseline.
- Item-item co-occurrence baseline implemented in pure Python.

## Metrics

Compute top-k metrics on fixed test sets:

- Recall@10
- NDCG@10
- HitRate@10
- MRR@10

## Why This Validates Pseudo-User Value

The target test set is fixed across variants. If V2-fix augmentation beats random-mix, the pseudo-user matching quality is useful beyond adding more interactions. If it beats single-domain, pseudo-user augmentation improves downstream recommendation under a lightweight model.
