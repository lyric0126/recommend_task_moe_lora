# V2fix Embeddings

## Smoke Test

v2fix_rows=26156
v2fix_dim=128
coverage={'movielens': 5575, 'goodreads': 7593, 'mind': 7605, 'kuairec': 5383}
v2_sample_dim=96
similarity_checks=['movielens first_pair_cosine=0.394725', 'goodreads first_pair_cosine=0.445092', 'mind first_pair_cosine=0.17989', 'kuairec first_pair_cosine=0.071776']
- movielens/1 text=`title toy story (1995) genres adventure animation children comedy fantasy` emb_head=[0.0, 0.0, 0.0, 0.361082, 0.0, 0.0]
- movielens/10 text=`title goldeneye (1995) genres action adventure thriller` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100 text=`title city hall (1996) genres drama thriller` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100044 text=`title human planet (2011) genres documentary` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100106 text=`title pervert's guide to ideology the (2012) genres documentary` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100163 text=`title hansel & gretel: witch hunters (2013) genres action fantasy horror imax` emb_head=[0.0, 0.206886, 0.0, 0.0, 0.0, 0.0]
- movielens/1003 text=`title extreme measures (1996) genres drama thriller` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- movielens/100302 text=`title upside down (2012) genres drama romance sci-fi` emb_head=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
