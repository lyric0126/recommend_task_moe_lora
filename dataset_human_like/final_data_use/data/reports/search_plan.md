# V2-fix Parameter Search Plan

## Current Default Parameters

Source config: `configs/pseudo_user_pipeline_v2fix.yaml`

Matching weights:

- semantic: `0.12`
- recency_semantic: `0.06`
- activity: `0.28`
- temporal: `0.34`
- behavior: `0.20`

Thresholds:

- strict: `0.62`
- medium: `0.50`

Control parameters:

- top_k: `3`
- fallback_candidates: `180`

## Search Scope

The search optimizes V2-fix matching parameters only:

- matching weights
- strict / medium thresholds
- top_k
- fallback_candidate_size

The search does not change datasets, embeddings, profile construction, matching feature definitions, or synthesis logic.

## Objective

For each trial:

```text
objective =
  global_consistency
  - 0.5 * loose_ratio
  - 0.2 * reuse_rate
  + 0.1 * strict_ratio
```

Constraints:

- pseudo_user_count >= 1000
- strict_count >= 500
- interactions_count must remain capped by the V2-fix synthesis interaction cap

Invalid trials receive a heavy penalty.

## Search Data

Search uses V2-fix user profiles and MovieLens anchor users. Stage Search-2 defines a deterministic dev anchor subset to keep random search affordable and reproducible.

## Reused Code

- V2-fix profile outputs: `data/processed/user_profiles_v2fix.parquet`
- V2-fix scoring primitives: `src/matching/matcher_v2fix.py`
- V2-fix synthesis structure: MovieLens anchor with Goodreads/MIND/KuaiRec matches

## Output

All trial records and summaries are written to:

- `data/processed/search/`
