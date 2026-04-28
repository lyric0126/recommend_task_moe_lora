# v2

V2 is a wide-coverage comparison set. It is useful for showing that coverage expansion with weak thresholds can degrade quality. It is not recommended as the main version.

## Files

- `pseudo_user_metadata.parquet`
- `pseudo_user_interactions.parquet`
- `summary.json`
- `README.md`

## Summary

- pseudo users: `5253`
- interactions: `4841259`
- construction type: `structured_v2`
- confidence distribution: `{'loose': 5253}`
- domain coverage distribution: `{'2': 1751, '3': 1751, '4': 1751}`
- recommended default: `False`

## Format Note

The files use the repository table API. In this environment they are JSONL fallback files with `.parquet` suffix because no parquet backend is installed.
