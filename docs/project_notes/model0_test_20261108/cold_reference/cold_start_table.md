# Cold-start references (no model)

1,142 new-product test cases, 54,875 warm control cases. Relative rank 0.5 = chance, 1.0 = last. Popularity uses cold-variant training counts.

| Scorer, cases | Recall@10 | NDCG@10 | MRR | median_rank | mean_relative_rank |
|---|---|---|---|---|---|
| Popularity, new products | 0.0000 | 0.0000 | 0.00004 | 25228 | 0.974 |
| Chance level, new products | 0.0004 | 0.0002 | 0.00041 | 12948 | 0.500 |
| Popularity, warm control | 0.0256 | 0.0134 | 0.01326 | 2222 | 0.238 |
