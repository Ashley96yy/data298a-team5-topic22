# Model 0 (SASRec): test results over seeds 42, 43, 44

Mean ± sample standard deviation over 3 seeds, 56,017 test users. Selection used validation NDCG@10 only.

| Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | MRR |
|---|---|---|---|---|---|
| Model 0 (SASRec), full catalog | 0.0648 ± 0.0006 | 0.0381 ± 0.0003 | 0.0899 ± 0.0006 | 0.0444 ± 0.0003 | 0.0350 ± 0.0002 |
| Model 0 (SASRec), 100 sampled negatives | 0.6315 ± 0.0033 | 0.4226 ± 0.0020 | 0.7613 ± 0.0035 | 0.4554 ± 0.0020 | 0.3732 ± 0.0015 |

| Model | Coverage@10 | Coverage@20 | ILD@10 |
|---|---|---|---|
| Model 0 (SASRec), full catalog | 0.6037 ± 0.0065 | 0.7023 ± 0.0068 | 0.2896 ± 0.0052 |

| Grouping | Group | Cases | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|
| user_group | long | 6,608 | 0.0389 ± 0.0005 | 0.0223 ± 0.0003 | 0.0219 ± 0.0003 |
| user_group | medium | 17,276 | 0.0571 ± 0.0006 | 0.0328 ± 0.0006 | 0.0303 ± 0.0005 |
| user_group | short | 32,133 | 0.0744 ± 0.0008 | 0.0442 ± 0.0004 | 0.0402 ± 0.0004 |
| target_item_group | head | 35,176 | 0.0939 ± 0.0007 | 0.0550 ± 0.0004 | 0.0502 ± 0.0003 |
| target_item_group | mid | 8,622 | 0.0199 ± 0.0005 | 0.0124 ± 0.0002 | 0.0122 ± 0.0002 |
| target_item_group | tail | 12,219 | 0.0130 ± 0.0007 | 0.0077 ± 0.0006 | 0.0073 ± 0.0005 |
