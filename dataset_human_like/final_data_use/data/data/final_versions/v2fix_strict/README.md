# v2fix_strict

V2-fix strict is the core high-quality set. It only retains strict pseudo users and is suitable for conservative analysis and benchmark core sets.

## Files

- `pseudo_user_metadata.parquet`
- `pseudo_user_interactions.parquet`
- `summary.json`
- `README.md`

## Summary

- pseudo users: `1211`
- interactions: `215057`
- construction type: `structured_v2fix_strict`
- confidence distribution: `{'strict': 1211}`
- domain coverage distribution: `{'3': 1090, '4': 121}`
- recommended default: `False`

## Format Note

The files use the repository table API. In this environment they are JSONL fallback files with `.parquet` suffix because no parquet backend is installed.
