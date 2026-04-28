# Stage 6 Matching Summary

## Results

- movielens_goodreads rows=8755 output=data/processed/matches_movielens_goodreads.parquet
  candidate_count_distribution={300: 326, 230: 281, 177: 248, 321: 213, 133: 192, 233: 184, 136: 130, 151: 92}
  sample=`[{'left_dataset': 'movielens', 'left_user_id': '1', 'right_dataset': 'goodreads', 'right_user_id': 'c08783ca7436f12df531731d1bf04de5', 'rank': 1, 'score': 0.507408, 'score_parts': {'semantic': 0.361204, 'activity': 0.335227, 'temporal': 0.830231, 'behavior': 0.937143}, 'block_key': 'mid|day', 'candidate_count': 177}, {'left_dataset': 'movielens', 'left_user_id': '1', 'right_dataset': 'goodreads', 'right_user_id': '88e38cfdb1f780527beb811ab5c3d0e8', 'rank': 2, 'score': 0.504322, 'score_parts': {'semantic': 0.336638, 'activity': 0.393973, 'temporal': 0.918461, 'behavior': 0.735162}, 'block_key': 'mid|day', 'candidate_count': 177}]`
- mind_kuairec rows=14750 output=data/processed/matches_mind_kuairec.parquet
  candidate_count_distribution={133: 2230, 55: 394, 53: 219, 25: 107}
  sample=`[{'left_dataset': 'mind', 'left_user_id': 'U87243', 'right_dataset': 'kuairec', 'right_user_id': '32', 'rank': 1, 'score': 0.234124, 'score_parts': {'semantic': 0.0, 'activity': 0.081652, 'temporal': 0.62927, 'behavior': 0.919394}, 'block_key': 'mid|night', 'candidate_count': 133}, {'left_dataset': 'mind', 'left_user_id': 'U87243', 'right_dataset': 'kuairec', 'right_user_id': '126', 'rank': 2, 'score': 0.223091, 'score_parts': {'semantic': 0.0, 'activity': 0.168109, 'temporal': 0.452554, 'behavior': 0.989584}, 'block_key': 'mid|night', 'candidate_count': 133}]`
