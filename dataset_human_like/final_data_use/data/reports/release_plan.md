# Final Versions Release Plan

## Existing Artifacts

| Version | Metadata | Interactions | Status |
| --- | --- | --- | --- |
| V1 | `data/processed/pseudo_user_metadata.parquet` | `data/processed/pseudo_user_interactions.parquet` | reusable directly |
| V2 | `data/processed/pseudo_user_metadata_v2.parquet` | `data/processed/pseudo_user_interactions_v2.parquet` | reusable directly |
| V2-fix | `data/processed/pseudo_user_metadata_v2fix.parquet` | `data/processed/pseudo_user_interactions_v2fix.parquet` | reusable; strict/all require filtering |
| Random / V0 | no standalone metadata/interactions version | original cleaned domain interactions exist | must generate final random baseline from original datasets |

## Confirmed Fields

- `pseudo_user_id` exists in all pseudo-user metadata/interactions.
- `source_members` exists in all metadata versions.
- `domains_present` exists in all metadata versions.
- `confidence_level` exists in V1, V2, and V2-fix.
- V2-fix confidence is distinguishable:
  - strict: 1211
  - medium: 331

## Release Versions

Final release directory:

- `data/final_versions/random/`
- `data/final_versions/v1/`
- `data/final_versions/v2/`
- `data/final_versions/v2fix_strict/`
- `data/final_versions/v2fix_all/`

Each version will contain:

- `pseudo_user_metadata.parquet`
- `pseudo_user_interactions.parquet`
- `summary.json`
- `README.md`

## Export Strategy

- `random`: generate a standalone V0 random pseudo-user baseline by randomly sampling source users from original cleaned domain datasets. It uses the V2-fix coverage profile only as a size/coverage template so it is comparable in scale, but it does not use matching scores or structured synthesis.
- `v1`: copy existing V1 metadata/interactions.
- `v2`: copy existing V2 metadata/interactions.
- `v2fix_strict`: filter V2-fix metadata to `confidence_level == strict`, then filter interactions by retained pseudo ids.
- `v2fix_all`: copy V2-fix strict+medium metadata/interactions as the main release version.

## Storage Format

The environment currently lacks a parquet backend. Files retain the `.parquet` suffix for compatibility with the existing pipeline but are JSONL fallback files with metadata headers.
