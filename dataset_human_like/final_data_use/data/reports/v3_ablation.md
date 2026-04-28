# V3 Ablation

## Smoke Test

- movielens/single_domain item_item Recall@10=0.083952 NDCG@10=0.039971 MRR@10=0.026944 train_rows=232206
- movielens/random_mix item_item Recall@10=0.082239 NDCG@10=0.04072 MRR@10=0.028386 train_rows=322667
- movielens/v1_pseudo_user item_item Recall@10=0.07767 NDCG@10=0.03954 MRR@10=0.028193 train_rows=315879
- movielens/v2_pseudo_user item_item Recall@10=0.073101 NDCG@10=0.037622 MRR@10=0.027054 train_rows=352206
- movielens/v2fix_pseudo_user item_item Recall@10=0.077099 NDCG@10=0.039146 MRR@10=0.027841 train_rows=307550
  best_by_recall=single_domain
- goodreads/single_domain item_item Recall@10=0.054505 NDCG@10=0.02439 MRR@10=0.015416 train_rows=201400
- goodreads/random_mix item_item Recall@10=0.050056 NDCG@10=0.02317 MRR@10=0.015139 train_rows=287912
- goodreads/v1_pseudo_user item_item Recall@10=0.052836 NDCG@10=0.02372 MRR@10=0.015044 train_rows=214467
- goodreads/v2_pseudo_user item_item Recall@10=0.037264 NDCG@10=0.017756 MRR@10=0.01191 train_rows=321400
- goodreads/v2fix_pseudo_user item_item Recall@10=0.043382 NDCG@10=0.022028 MRR@10=0.015547 train_rows=274480
  best_by_recall=single_domain
