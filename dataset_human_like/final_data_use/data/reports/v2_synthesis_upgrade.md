# V2 Synthesis Upgrade

## Smoke Test

pseudo_users=5253
pseudo_interactions=4841259
coverage={2: 1751, 3: 1751, 4: 1751}
confidence={'loose': 5253}
high_confidence_samples=`[{'pseudo_user_id': 'pseudo_v2_001273_2d', 'source_members': {'movielens': '568', 'mind': 'U232080'}, 'domains_present': ['mind', 'movielens'], 'global_consistency_score': 0.509801, 'confidence_level': 'loose', 'component_scores': {'mind': 0.569801}}, {'pseudo_user_id': 'pseudo_v2_000395_2d', 'source_members': {'movielens': '1353', 'mind': 'U216022'}, 'domains_present': ['mind', 'movielens'], 'global_consistency_score': 0.497921, 'confidence_level': 'loose', 'component_scores': {'mind': 0.557921}}, {'pseudo_user_id': 'pseudo_v2_000580_2d', 'source_members': {'movielens': '152', 'mind': 'U232080'}, 'domains_present': ['mind', 'movielens'], 'global_consistency_score': 0.470217, 'confidence_level': 'loose', 'component_scores': {'mind': 0.530217}}]`
low_confidence_samples=`[{'pseudo_user_id': 'pseudo_v2_000953_2d', 'source_members': {'movielens': '28', 'goodreads': '90b206d005f68dc36fe0b9f1aa951f11'}, 'domains_present': ['goodreads', 'movielens'], 'global_consistency_score': 0.106427, 'confidence_level': 'loose', 'component_scores': {'goodreads': 0.166427}}, {'pseudo_user_id': 'pseudo_v2_000953_3d', 'source_members': {'movielens': '28', 'goodreads': '90b206d005f68dc36fe0b9f1aa951f11', 'kuairec': '78'}, 'domains_present': ['goodreads', 'kuairec', 'movielens'], 'global_consistency_score': 0.135861, 'confidence_level': 'loose', 'component_scores': {'goodreads': 0.166427, 'kuairec': 0.165295}}, {'pseudo_user_id': 'pseudo_v2_001183_2d', 'source_members': {'movielens': '487', 'kuairec': '9'}, 'domains_present': ['kuairec', 'movielens'], 'global_consistency_score': 0.135961, 'confidence_level': 'loose', 'component_scores': {'kuairec': 0.195961}}]`
