# V2-fix Plan

## Diagnosis

V2 expanded coverage from 1751 V1 pseudo users to 5253 pseudo users by emitting 2-domain, 3-domain, and 4-domain variants for each anchor. This increased recall but mixed many low-confidence combinations into the mean metric.

Observed V2 metrics:

- `semantic_only = 0.101169`
- `activity_temporal = 0.533362`
- `full = 0.304942`
- `global_consistency = 0.316953`
- confidence distribution: `loose = 5253`, `strict = 0`, `medium = 0`

## Likely Causes

- Semantic-only is weak because V2 uses generic signed hashing over noisy, uneven cross-domain text. Cross-domain vocabularies are sparse and not aligned enough, so semantic similarity has low magnitude.
- Full is worse than activity+temporal because V2 gave high weight to weak semantic vectors, pulling otherwise good temporal/activity candidates down.
- Confidence collapsed to loose because thresholds were calibrated for a stronger score distribution than the implemented fallback representation produced.
- 2/3/4-domain expansion diluted consistency because low-quality partial combinations were retained without a minimum component score and without interaction volume control.

## Fix Strategy

- Rebuild item embeddings with cleaner text, stopword filtering, TF-IDF pruning, unsigned hashing, and broad domain/category bridge tokens.
- Rebuild user semantic profiles from the fixed embeddings while preserving V2 structured activity, temporal, and behavior features.
- Use stricter candidate generation and much smaller fallback candidate pools.
- Reweight full scoring toward robust activity, temporal, and behavior signals while keeping semantic as a secondary positive signal.
- Calibrate confidence thresholds so strict/medium/loose are meaningful.
- Synthesize only medium+ component matches, prefer 3/4-domain users, and cap interactions per source user.

## Expected Outcome

V2-fix should be clearly better than V2 and Random Mix. It may or may not exceed V1 because no stronger embedding backend is available in the current environment, but the target is to approach V1 while preserving some V2 flexibility.
