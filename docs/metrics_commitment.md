# Evaluation metrics commitment (Workbook 2, section 4.4)

This file records Table 4.4.1 of Workbook 2 section 4.4 in the repository, so that the metrics have a dated, reviewable record
in Git. The metrics are committed before any Model 1 result exists and are not revised after results are seen; any later change is
listed in the amendment log at the end.

How to read the dates: **Committed on** is the date given in Workbook 1 or Workbook 2 for that row. **Evidence in Git** is what the
repository can show: rows marked `eval_protocol.json` are defined in `docs/data_run/records/eval_protocol.json` (frozen with the
dataset, manifest created 2026-10-18 UTC, in the repository since commit `ddc1662`, 2026-11-06; the Model 0 full grid ran on 2026-11-07).
Every row is also recorded by the commit that adds this file.

| Metric | Why this metric | Benchmark or split it is measured on | Committed on | Evidence in Git |
|---|---|---|---|---|
| NDCG@10 (primary, also the model selection metric) | Named on the topic slide. Rewards placing the true item near the top of the list, not only inside it. | Full-catalog test split, all 56,017 users. Validation split for selection only. | October 7, 2026 (Workbook 1) | `eval_protocol.json` (selection metric) and this file |
| Recall@10 (equal to HitRate@10, since each user has one true item) | Named on the topic slide. Shows how often the true item appears in a 10-item list. | Full-catalog test split | October 7, 2026 (Workbook 1) | `eval_protocol.json` and this file |
| NDCG@5, NDCG@20, Recall@5, Recall@20, MRR | Check that conclusions hold for shorter and longer lists. MRR uses the full ranking, not only the top K. | Full-catalog test split | October 18, 2026 (eval/eval_protocol.json) | `eval_protocol.json` |
| Coverage@10, Coverage@20, category diversity@10 | Detect a model that looks accurate only because it recommends the same popular items to everyone. | Full-catalog test split, top-K lists | October 18, 2026 (eval/eval_protocol.json) | `eval_protocol.json` |
| NDCG@10, Recall@10 and MRR by user group and item group | Users with short histories and unpopular items are where privacy constraints are expected to hurt most. | Test split by user history length (short, medium, long) and by popularity of the true item (head, mid, tail) | October 7, 2026 (Workbook 1), groups fixed October 18, 2026 | Groups in the frozen dataset (`groups/`, checksums in `manifest.json`) and this file |
| Cold-start Recall@10, NDCG@10, MRR, median rank and mean relative rank | Named on the topic slide. Tests products the model has never seen, compared with the control group, popularity and random ranking. | 1,142 new-product test cases in the cold-start version of the data. The other 54,875 users as the control group. | November 6, 2026 | Cold-start variant in `records/cold_start_info.json` (commit `ddc1662`) and this file |
| Training loss and validation metrics per epoch | Learning curves show whether training converged, overfitted or diverged. | Training and validation splits, every epoch of every run | November 6, 2026 | This file |
| Mean and sample standard deviation over seeds | A single run can be lucky. The spread shows whether a difference between models is larger than run-to-run noise. | Seeds 42, 43 and 44 for each final configuration | October 18, 2026 (eval/eval_protocol.json) | `eval_protocol.json` |
| NDCG@10 and Recall@10 at each privacy budget epsilon | Named on the topic slide. Measures how much accuracy a formal privacy guarantee costs. | Full-catalog test split, Model 4 at epsilon = 0.5, 1, 2, 5 and 10 | October 7, 2026 (Workbook 1) | This file |
| Membership-inference attack AUC (primary attack metric) | Named on the topic slide. Measures how much each model leaks about the data it was trained on. Membership is what differential privacy and federated training are designed to protect. | Interactions used in training against held-out interactions, every model | October 7, 2026 (Workbook 1) | This file |
| Model size, peak memory, time per recommendation on one CPU thread | On-device compute cost from the topic slide. Targets: at most 5 MB, under 100 ms per recommendation, and at most 10% lower NDCG@10 than Model 1. The 10 MB and 25 MB budgets are size ablations, not targets. | Model 3 on the test split | October 7, 2026 (Workbook 1) | This file |
| Data sent per training round and in total | Communication cost from the topic slide. Federated training sends model updates instead of user data. | Model 2 federated training | November 6, 2026 | This file |
| Failure taxonomy: count and share of failures per category | Shows why a model fails, not only how often, and points to fixes. | At least 50 test cases ranked outside the top 10, per final model | November 6, 2026 | This file |
| Training compute (GPU-hours) and recommendation latency | Every benchmark result states its compute. Latency is what a user of the prototype would experience. | Each final run. Latency at the prototype's serving configuration. | November 6, 2026 | This file |
| NDCG@10 and Recall@10 with 100 sampled negatives (secondary) | Comparison with published papers that report sampled metrics. Never used to choose a model. | Test split, fixed negatives in eval/neg_test.npy | October 18, 2026 (eval/eval_protocol.json) | `eval_protocol.json` (sampled negatives in `neg_test.npy`, checksum in `manifest.json`) |

Optional additional metric for 298B, only if the team decides to include it: attribute-inference accuracy of the LLM attacker against chance,
with attributes defined from data that exists (categories, brands, price tier, held-out purchases). It does not replace the membership-inference AUC.

## Amendment log
- 2026-11-07: aligned with Table 4.4.1 after review by Sneha (PR #13), before this file was merged. Changes against the first version: membership-inference AUC is the
  primary attack metric (attribute inference moved to an optional addition); Model 3 targets are at most 5 MB, under 100 ms per recommendation and at most 10% lower NDCG@10
  than Model 1, with 10 MB and 25 MB as size ablations; added the cold-start, failure-taxonomy, GPU-hours, communication, learning-curve, group and sampled-negative rows.
