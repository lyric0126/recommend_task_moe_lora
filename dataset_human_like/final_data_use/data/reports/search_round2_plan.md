# Search Round 2 Plan

## Why Round 1 Tuned Did Not Replace Default

Round 1 found a configuration with slightly higher full global consistency:

- default V2-fix global: `0.680195`
- first-round tuned global: `0.682476`

However, the tuned version is less suitable as a release default:

- introduced `28` loose pseudo users
- reduced strict ratio from default
- increased interactions from `298777` to `359630`
- lower conservative objective: `0.619461` vs default `0.636447`

## Why Round 2 Is More Conservative

Round 2 treats loose samples as high risk and optimizes release stability rather than raw global consistency.

Hard constraints:

- `pseudo_user_count >= 1300`
- `strict_count >= 1000`
- `loose_count == 0` preferred; `loose_count <= 5` maximum tolerated
- estimated interaction count must not exceed `1.10 * 298777`
- strict ratio should stay close to default

## Round 2 Objective

If `loose_count > 5`, the trial is invalid. Otherwise:

```text
objective =
  1.0 * global_consistency
  + 0.15 * strict_ratio
  - 0.15 * reuse_rate
  - 0.20 * interaction_growth_penalty
```

where:

```text
interaction_growth_penalty = max(0, interaction_count / 298777 - 1)
```

## Search Space

Weights:

- semantic: `[0.08, 0.22]`
- recency_semantic: `[0.00, 0.08]`
- activity: `[0.18, 0.35]`
- temporal: `[0.24, 0.40]`
- behavior: `[0.15, 0.30]`

Thresholds, after smoke testing the originally suggested stricter range:

- strict: `[0.62, 0.70]`
- medium: `[0.50, 0.60]`

The initially proposed stricter threshold range produced no valid trials under `strict_count >= 1000`, so the executable round-2 search narrows around the current stable default instead of exploring overly strict thresholds.

Controls:

- top_k: `{2, 3, 4}`
- fallback_candidate_size: `{80, 120, 180, 220}`

## Output

All round-2 artifacts are written under:

- `data/processed/search_round2/`
