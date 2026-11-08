# Evaluation metrics commitment (Workbook 2, section 4.4)

Metrics are committed before any Model 1 result exists and are not revised after results are seen.

## 1. Metrics already frozen (all models)
Source: `docs/data_run/records/eval_protocol.json` (frozen with the dataset; the dataset manifest was created 2026-10-18 UTC; the file is in
the repository at commit `ddc1662`, 2026-11-06). The Model 0 full grid ran on 2026-11-07, after this commitment.

| Metric | Why this metric | Benchmark or split it is measured on | Committed on |
|---|---|---|---|
| Recall@K and HitRate@K, K = 5, 10, 20 | One next item per user, so both equal the share of users whose true next item is in the top K; easy to explain and robust | Full-catalog ranking, history masked; validation for tuning, test once at the end | 2026-11-06 (commit `ddc1662`) |
| NDCG@K, K = 5, 10, 20 | Rewards a hit more when it is ranked higher; NDCG@10 on validation is the model-selection and early-stopping metric | Same | 2026-11-06 (commit `ddc1662`) |
| MRR | Position of the first hit; sensitive to the very top of the list | Same | 2026-11-06 (commit `ddc1662`) |
| Coverage@K | Share of the catalog that appears in at least one top-K list; exposes popularity collapse | Same | 2026-11-06 (commit `ddc1662`) |
| ILD@K (category) | Diversity of categories inside one list | Same | 2026-11-06 (commit `ddc1662`) |

Protocol details: seeds 42, 43, 44 (mean and standard deviation); secondary check with 100 uniformly sampled negatives; slices by user
history group, head/mid/tail target item, and the cold-start variant (1,295 cold items, 1,142 test cases); test evaluated once per run.

## 2. Metrics planned for 298B (committed on the date this file is committed)
| Metric | Why this metric | Benchmark or split it is measured on |
|---|---|---|
| Privacy budget epsilon with NDCG@10 | Shows the utility cost of differential privacy | Model 4, epsilon in {0.5, 1, 2, 5, 10}, validation and test as above |
| Model size (MB), CPU latency (single thread), peak memory | The on-device claim is about footprint, not accuracy alone | Model 3, against the 5, 10 and 25 MB budgets |
| Attribute-inference accuracy of the LLM attacker versus chance | Measures what each model reveals about a user | Models 0 to 4, attributes defined from data that exists (categories, brands, price tier, held-out purchases) |
| Federated rounds and communication per round | Cost of training without centralising data | Model 2 |
