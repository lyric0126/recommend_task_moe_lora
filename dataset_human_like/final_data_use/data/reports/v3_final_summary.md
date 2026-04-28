# V3 Final Summary

## Summary

## Task Definition

Evaluate whether V2-fix pseudo-user augmentation improves lightweight target-domain recommendation on fixed MovieLens and Goodreads leave-last-2-out splits.

## Data Construction

- `single-domain`: target-domain train only.
- `random-mix`: target-domain train plus random target-domain augmentation.
- `pseudo-user_v2fix`: target-domain train plus V2-fix pseudo-user target-domain augmentation.

## Baselines

- Popularity/frequency.
- Pure Python item-item co-occurrence.

## Main Results

| target | variant | model | recall | ndcg | hitrate | mrr |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| goodreads | pseudo_user_v2fix | item_item | 0.044494 | 0.020981 | 0.044494 | 0.013839 |
| goodreads | pseudo_user_v2fix | popularity | 0.03337 | 0.015398 | 0.03337 | 0.010061 |
| goodreads | random_mix | item_item | 0.050056 | 0.02317 | 0.050056 | 0.015139 |
| goodreads | random_mix | popularity | 0.031702 | 0.014839 | 0.031702 | 0.009822 |
| goodreads | single_domain | item_item | 0.054505 | 0.02439 | 0.054505 | 0.015416 |
| goodreads | single_domain | popularity | 0.034483 | 0.015705 | 0.034483 | 0.010161 |
| movielens | pseudo_user_v2fix | item_item | 0.079383 | 0.039957 | 0.079383 | 0.028109 |
| movielens | pseudo_user_v2fix | popularity | 0.052541 | 0.02769 | 0.052541 | 0.020169 |
| movielens | random_mix | item_item | 0.082239 | 0.04072 | 0.082239 | 0.028386 |
| movielens | random_mix | popularity | 0.051399 | 0.025465 | 0.051399 | 0.017563 |
| movielens | single_domain | item_item | 0.083952 | 0.039971 | 0.083952 | 0.026944 |
| movielens | single_domain | popularity | 0.051399 | 0.026408 | 0.051399 | 0.01881 |

## Ablation

| target | variant | recall | ndcg | mrr | train_rows |
| --- | --- | ---: | ---: | ---: | ---: |
| goodreads | random_mix | 0.050056 | 0.02317 | 0.015139 | 287912 |
| goodreads | single_domain | 0.054505 | 0.02439 | 0.015416 | 201400 |
| goodreads | v1_pseudo_user | 0.052836 | 0.02372 | 0.015044 | 214467 |
| goodreads | v2_pseudo_user | 0.037264 | 0.017756 | 0.01191 | 321400 |
| goodreads | v2fix_pseudo_user | 0.043382 | 0.022028 | 0.015547 | 274480 |
| movielens | random_mix | 0.082239 | 0.04072 | 0.028386 | 322667 |
| movielens | single_domain | 0.083952 | 0.039971 | 0.026944 | 232206 |
| movielens | v1_pseudo_user | 0.07767 | 0.03954 | 0.028193 | 315879 |
| movielens | v2_pseudo_user | 0.073101 | 0.037622 | 0.027054 | 352206 |
| movielens | v2fix_pseudo_user | 0.077099 | 0.039146 | 0.027841 | 307550 |

## Key Questions

1. pseudo-user(V2-fix) vs random-mix: `{'goodreads': False, 'movielens': False}`
2. pseudo-user(V2-fix) vs single-domain: `{'goodreads': False, 'movielens': False}`
3. Quality trend by ablation best variant: `{'goodreads': 'single_domain', 'movielens': 'single_domain'}`
4. Support for stronger models: yes if V2-fix beats random-mix or single-domain on at least one target; otherwise only after revisiting split/model design.

## Limitations

- Baselines are intentionally lightweight and may not exploit cross-domain user structure fully.
- Augmentation uses target-domain pseudo interactions only; no neural cross-domain recommender is trained.
- Current storage remains JSONL fallback in `.parquet` paths due to missing parquet dependencies.
- Results depend on the leave-last-2-out split and sampled V1/V2/V2-fix artifacts.

## Next Steps

- Add a stronger but still reproducible implicit-feedback model when dependencies allow.
- Evaluate true cross-domain transfer models that consume all domains in each pseudo user.
- Repeat with true parquet storage and full data scale.
