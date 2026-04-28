# v1

V1 is the conservative structured pseudo-user baseline. It improves over random in construction quality but is not the current best version.

## Files

- `pseudo_user_metadata.parquet`
- `pseudo_user_interactions.parquet`
- `summary.json`
- `README.md`

## Summary

- pseudo users: `1751`
- interactions: `481974`
- construction type: `structured_v1`
- confidence distribution: `{'loose': 1746, 'medium': 5}`
- domain coverage distribution: `{'4': 1751}`
- recommended default: `False`

## Format Note

The files use the repository table API. In this environment they are JSONL fallback files with `.parquet` suffix because no parquet backend is installed.
