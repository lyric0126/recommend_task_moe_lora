# V2 Storage Upgrade

## Smoke Test

storage_info=`{'path': 'data/processed/v2_storage_smoke.parquet', 'exists': True, 'storage_format': 'jsonl_fallback', 'schema': ['dataset', 'user_id', 'item_id', 'score'], 'sample_count': 1}`
rows_read=1
sample=`[{'dataset': 'smoke', 'item_id': 'i1', 'score': 1.0, 'user_id': 'u1'}]`
