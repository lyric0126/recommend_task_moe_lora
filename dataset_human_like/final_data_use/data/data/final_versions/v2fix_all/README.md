# v2fix_all

V2-fix all is the main release version. It keeps strict and medium pseudo users and balances quality with scale. This is the default recommended dataset.

## Files

- `pseudo_user_metadata.parquet`
- `pseudo_user_interactions.parquet`
- `summary.json`
- `README.md`

## Summary

- pseudo users: `1542`
- interactions: `298777`
- construction type: `structured_v2fix_all`
- confidence distribution: `{'strict': 1211, 'medium': 331}`
- domain coverage distribution: `{'3': 1231, '4': 311}`
- recommended default: `True`

## Format Note

The files use the repository table API. In this environment they are JSONL fallback files with `.parquet` suffix because no parquet backend is installed.
