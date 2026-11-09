# Cold-start test: Model 0 (SASRec), seeds 42, 43, 44

Model trained on the cold variant (new products removed from training). 1,142 new-product test cases, 54,875 warm control cases scored by the same model. Mean ± sample std over seeds. Relative rank 0.5 = chance, 1.0 = last.

| Scorer, cases | Recall@10 | NDCG@10 | MRR | median_rank | mean_relative_rank |
|---|---|---|---|---|---|
| Model 0 (SASRec), new products | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00004 ± 0.00000 | 24947 ± 53 | 0.940 ± 0.006 |
| Popularity, new products | 0.0000 | 0.0000 | 0.00004 | 25228 | 0.974 |
| Chance level, new products | 0.0004 | 0.0002 | 0.00041 | 12948 | 0.500 |
| Model 0 (SASRec), warm control | 0.0664 ± 0.0006 | 0.0387 ± 0.0002 | 0.03527 ± 0.00030 | 1124 ± 11 | 0.134 ± 0.001 |
| Popularity, warm control | 0.0256 | 0.0134 | 0.01326 | 2222 | 0.238 |
