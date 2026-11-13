# Model 0 final baseline comparison

Written by: Abhinita Sanabada

SASRec Model 0 achieves full-catalog test NDCG@10 of **0.038115 ± 0.000311** over seeds 42, 43 and 44. This exceeds the strongest overall simple reference, last-item Markov (0.027433), by **38.94%**. These are descriptive results on the frozen Skin Care dataset, not a statistical-significance claim.

## Protocol and attribution

All comparisons below use 56,017 test cases, a catalog of 25,903 items, history masking, and the same frozen evaluator. Ties follow its average-rank-then-metrics convention. The eight-run search used only full-catalog validation NDCG@10 with seed 42; the winner (hidden size 128, dropout 0.5, learning rate 0.0005) was repeated with seeds 43 and 44. Test scores did not select the configuration.

Abhinita Sanabada implemented the configuration/grid and checkpoint workflow and wrote this comparison. Ruchika Chaurasia produced the simple baselines. Sneha Tumkur Narendra produced the three-seed test and cold-start evaluation. Training uses Yuyao Ding’s framework and Disha Rawat’s sequence loader. This report checks their committed evidence; it does not claim to rerun their experiments.

## Full-catalog test comparison

| Model | NDCG@5 | NDCG@10 | NDCG@20 | Recall@5 | Recall@10 | Recall@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.009846 | 0.013112 | 0.016261 | 0.015049 | 0.025100 | 0.037703 | 0.012988 |
| Last-item Markov | 0.023505 | 0.027433 | 0.031000 | 0.032651 | 0.044879 | 0.058982 | 0.025465 |
| Cosine item-kNN | 0.016977 | 0.020408 | 0.023307 | 0.024992 | 0.035614 | 0.047129 | 0.018254 |
| SASRec, mean ± sample SD | 0.032223 ± 0.000234 | 0.038115 ± 0.000311 | 0.044421 ± 0.000341 | 0.046569 ± 0.000334 | 0.064849 ± 0.000553 | 0.089925 ± 0.000622 | 0.035006 ± 0.000248 |

SASRec uncertainty is the sample standard deviation (ddof=1) across three training seeds. The deterministic reference scores have no seed standard deviation. All three SASRec seeds exceed Markov on the seven overall metrics shown. No full-catalog random baseline is supplied in this reference artifact; a cold-start chance value must not substitute for one.

The NDCG@10 difference is 0.010682 (38.94% relative). Recall@10 improves by 0.019970 (44.50% relative). Three seeds quantify run-to-run variation; they do not establish significance against a baseline. A paired analysis would require per-case predictions or ranks.

## Additional metrics and limitations

| Metric | SASRec mean ± sample SD |
|---|---:|
| Coverage@10 | 0.603675 ± 0.006530 |
| Coverage@20 | 0.702300 ± 0.006828 |
| ILD@10 | 0.289563 ± 0.005197 |

The reference JSON does not provide coverage or diversity, so no baseline improvement is claimed for these metrics. The secondary evaluation with 100 sampled negatives gives NDCG@10 of 0.422557 ± 0.001972. That candidate set is much easier than full-catalog ranking and is not used in the table above.

Overall gains do not hold uniformly across item groups. The comparison with cosine item-kNN below uses the same group case counts.

| Group | Test cases | SASRec NDCG@10 mean ± sample SD | Cosine item-kNN NDCG@10 |
|---|---:|---:|---:|
| long | 6,608 | 0.022311 ± 0.000281 | 0.008213 |
| medium | 17,276 | 0.032796 ± 0.000571 | 0.018158 |
| short | 32,133 | 0.044225 ± 0.000424 | 0.024125 |
| head | 35,176 | 0.054975 ± 0.000395 | 0.023837 |
| mid | 8,622 | 0.012379 ± 0.000225 | 0.014077 |
| tail | 12,219 | 0.007740 ± 0.000553 | 0.015002 |

Cosine item-kNN is stronger on mid and tail items, despite SASRec’s better overall score. Sneha’s [separate cold-start evaluation](../model0_test_20261108/cold_start_3seeds/cold_start_table.md) reports zero Recall@10 and NDCG@10 on 1,142 cold cases after retraining with the cold items removed from training. That altered training set is a separate experiment, not a slice of the original three checkpoints. These limitations motivate evaluating item text and profiles in Model 1; they do not establish that Model 1 will improve.

The reference cosine scorer uses an unpruned similarity matrix (no neighbour k), summed over the evaluator’s last-50-item input window. Both co-occurrence scorers include a fixed additive popularity term of 1e-6 times normalized popularity. It can change near-ties and is not an exact-ties-only rule.

## Validation and literature context

The selected seed-42 validation NDCG@10 was 0.045582; the three-seed validation mean was 0.045405 ± 0.000163. Those validation scores are not test estimates. The same-data test comparison above provides stronger evidence of a useful sequential baseline.

The [four-paper literature benchmark](../../literature/amazon2023_ndcg10_benchmarks.md) documents results and limitations for HSTU-BLaIR, How Well Does Generative Recommendation Generalize?, DISR and MEMOIR. Their categories, preprocessing, splits and candidate pools differ. In particular, DISR uses 99 negatives and MEMOIR uses 999; neither provides a direct target for our Skin Care full-catalog result. The result is plausible in light of the verified local baselines, but it does not reproduce any paper’s numerical score or establish superiority over its model.

## Evidence verification

The verification recomputed all 24 aggregate full-catalog and sampled metrics from the three per-seed JSON records, including mean, sample SD, minimum and maximum. Each checkpoint hash and best epoch matches the Part D handoff, and every recorded validation reload gap is zero. All nine frozen input hashes and the manifest hash agree between the reference baseline provenance and all three training records. This is an independent record/arithmetic check, not a new inference run or fresh checkpoint load.

The experiments retain their actual environments: SASRec records Python 3.13.16 / PyTorch 2.14.1+cu130, while the baseline provenance records Python 3.11.18 / NumPy 2.2.6 / pandas 2.3.2 / SciPy 1.17.1. No cross-environment bitwise reproducibility is claimed.

- [Baseline source at 2cd58b0](https://github.com/Ashley96yy/data298a-team5-topic22/blob/2cd58b099bc34ae1c8730b3801a1d63c3fdbf0d5/docs/baselines/baselines.json) and [local evidence snapshot](reference_baselines.json). PR #6 remained open at this review; the versioned results are cited explicitly.
- [Sneha’s committed test results](../model0_test_20261108/README.md), added in commit c203d42.
- [Original validation handoff](../model0_full_grid_20261107/issue3_1_validation_summary.json).
- [Machine-readable verification and source hashes](verification.json).
