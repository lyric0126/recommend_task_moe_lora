# V2 Item Representation

## Smoke Test

backend=hash_tfidf_fallback
old_embedding_rows=26156
new_embedding_rows=26156
new_embedding_dim=96
coverage={'movielens': 5575, 'goodreads': 7593, 'mind': 7605, 'kuairec': 5383}
storage={'path': 'data/processed/item_embeddings_v2.parquet', 'exists': True, 'storage_format': 'jsonl_fallback', 'schema': ['dataset', 'item_id', 'item_text_v2', 'embedding'], 'sample_count': 26156}
- movielens/1 text=`Toy Story (1995) Adventure;Animation;Children;Comedy;Fantasy adventure;animation;children;comedy;fantasy` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/10 text=`GoldenEye (1995) Action;Adventure;Thriller action;adventure;thriller` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100 text=`City Hall (1996) Drama;Thriller drama;thriller` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100044 text=`Human Planet (2011) Documentary documentary` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100106 text=`Pervert's Guide to Ideology, The (2012) Documentary documentary` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100163 text=`Hansel & Gretel: Witch Hunters (2013) Action;Fantasy;Horror;IMAX action;fantasy;horror;imax` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/1003 text=`Extreme Measures (1996) Drama;Thriller drama;thriller` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100302 text=`Upside Down (2012) Drama;Romance;Sci-Fi drama;romance;sci-fi` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
