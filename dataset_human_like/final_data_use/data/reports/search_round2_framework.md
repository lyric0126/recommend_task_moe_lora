# Search Round2 Framework

## Smoke Test

trial=`{'trial_id': 1, 'params': {'weights': {'semantic': 0.180564, 'recency_semantic': 0.061343, 'activity': 0.293189, 'temporal': 0.312881, 'behavior': 0.152023}, 'strict_threshold': 0.705868, 'medium_threshold': 0.544247, 'top_k': 2, 'fallback_candidate_size': 80}, 'pseudo_user_count': 96, 'strict_count': 33, 'medium_count': 63, 'loose_count': 0, 'global_consistency': 0.684515, 'semantic_consistency': 0.488214, 'temporal_consistency': 0.833031, 'behavior_consistency': 0.856963, 'reuse_rate': 0.165605, 'loose_ratio': 0.0, 'strict_ratio': 0.34375, 'objective': -0.388763, 'valid': False, 'interaction_count': 18601, 'interaction_growth_penalty': 0.0, 'invalid_reason': ['pseudo_user_count_below_min', 'strict_count_below_min']}`
valid_logic=False invalid_reason=['pseudo_user_count_below_min', 'strict_count_below_min']
candidate_parts=data/processed/search_round2/dev_candidate_parts.parquet
