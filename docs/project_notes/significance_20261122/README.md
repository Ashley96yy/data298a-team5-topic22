# Paired significance tests: Model 0 against the baselines (2026-11-22)

Produced by `src/evaluation/significance.py` as committed in `docs/metrics_commitment.md`
(amendment of 2026-11-22), before any Model 1 test-split result exists.

| Comparison (A minus B) | NDCG@10 A | NDCG@10 B | Difference | Relative | 95% CI | p | Holm p | Significant |
|---|---|---|---|---|---|---|---|---|
| Model 0 vs popularity | 0.0381 | 0.0131 | +0.0250 | +190.7% | [+0.0237, +0.0263] | < 0.0001 | 0.0003 | yes |
| Model 0 vs markov_last_item | 0.0381 | 0.0274 | +0.0107 | +38.9% | [+0.0098, +0.0116] | < 0.0001 | 0.0003 | yes |
| Model 0 vs itemknn_cosine | 0.0381 | 0.0204 | +0.0177 | +86.8% | [+0.0166, +0.0189] | < 0.0001 | 0.0003 | yes |

All 56,017 test users; Model 0 per-user NDCG@10 is the mean over seeds 42, 43 and 44.
A p-value below 0.0001 is the smallest possible with 10,000 permutations (1/10,001).
Holm covers only these three comparisons. The final table adds the four Model 1
comparisons and is corrected again across all seven.

Inputs and checks:
- Model 0: `test_ranks.npz` of seeds 42, 43 and 44 (shared Drive, `part_e_f_results/model0_real_results/runs/`).
  Each reproduces the test NDCG@10 in its `test_evaluation.json`; SHA-256 hashes are in `significance.json`.
- Baselines: rescored once through the frozen evaluator; they reproduce `docs/baselines/baselines.json` exactly.
  Per-user ranks (`baseline_test_ranks.npz`, SHA-256 in `baseline_ranks.json`) are kept in the shared Drive, not in Git.

Files: `significance.md`, `significance.json` (all values, secondary Recall@10 and MRR intervals, input hashes),
`baseline_ranks.json` (baseline metrics, verified input hashes, code hashes).
