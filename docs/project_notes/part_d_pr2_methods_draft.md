# Workbook 2 contribution — Part D methodology and Model 0 results

Written by: Abhinita Sanabada

## Experimental configuration

The configuration system separates frozen dataset properties, fixed training
settings, the hyperparameter search space, and evaluation settings. The dataset
is the Skin Care subset of Amazon Reviews 2023 Beauty and Personal Care. Sequence
length is fixed at 50. Model selection uses history-masked full-catalog validation
NDCG@10 through the shared frozen evaluator. Dataset artifacts are checked against
the supplied manifest before the evaluator is imported.

The assigned search contains eight combinations: hidden size 64 or 128, dropout
0.2 or 0.5, and learning rate 0.001 or 0.0005. The fixed settings agreed on
2026-11-06 are batch size 128, at most 200 epochs, two transformer layers, one
attention head, Adam, zero weight decay, patience 10, and no learning-rate
scheduler. Item and position embeddings use normal initialization with standard
deviation 0.02; the initial item-padding row is zero. This implementation is the
merged PR #3 model, with PR #4 padding/consistency tests.

The eight configurations are tuned with seed 42. The selected configuration is
retrained with seeds 43 and 44, and all three seeds contribute to the final
report. Development smoke checks use two epochs and limited users, keeping the
other agreed choices; they are not reported as final performance.

## Search and selection

Each configuration runs in a fresh process with its own seeded model, optimizer,
and dataloader. The training framework selects the highest validation NDCG@10
checkpoint within a run. The grid runner compares those selected validation
scores only after all eight runs succeed. Equal scores select the first
configuration in the documented JSON order. This deterministic tie rule does
not imply statistically superior performance. Failed grids do not select a
winner from the remaining configurations.

The current implementation uses one explicitly selected tuning seed. Repeated
training of the selected configuration and final test reporting belong to the
multi-seed evaluation workflow after the selected configuration is frozen. No test metrics are used for hyperparameter selection.

## Evaluation convention

The team retains the frozen evaluator's average-rank convention for score ties:
rank = 1 + number of higher scores + half the number of other equal scores.
Metrics are computed from that rank. This is not the expectation of NDCG or MRR
over random permutations of tied items. The same convention should be reported
and retained across model comparisons.

## Saved artifacts and reproducibility

The grid saves per-run logs and histories, a results CSV, the selected JSON
configuration, the selected checkpoint, and a selection record with a checkpoint
hash. Run metadata includes model and data provenance and runtime versions.
Checkpoint verification reconstructs the selected model and checks consistency
with the configuration, epoch, validation score, and evaluation policy.

The training framework's last checkpoint additionally preserves optimizer,
scheduler, early-stopping, training-history and random-number state for resume
at epoch boundaries. The optional storage-saving mode removes completed runs'
last checkpoints while keeping their best checkpoints; those runs can no longer
resume from last.pt.

## Literature context

The separate [Amazon 2023 literature benchmark](../literature/amazon2023_ndcg10_benchmarks.md)
records SASRec and comparison-model results from four papers, with table citations
and protocol limitations. Sampled-negative scores and scores from different
Amazon subsets are contextual references rather than direct full-catalog targets.
The final comparison should use the team's same-data baseline table and explain
these protocol differences.

## Full-grid results — contribution to Section 4.6

On November 7, 2026, the eight-configuration search completed on all training and
validation users with seed 42. Every run passed. The selected model had hidden
size 128, dropout 0.5 and learning rate 0.0005. Its best full-catalog validation
NDCG@10 was 0.0455815861 at epoch 7. The 200-epoch setting was an upper limit,
with patience 10; the best epoch is not the total number of epochs executed.
The sum of the eight recorded run durations was 2,812.77 seconds (46.88 minutes),
excluding setup and repeat training. It is not a measured GPU utilization total.


| Hidden size | Dropout | Learning rate | Validation NDCG@10 | Best epoch | Status |
|---:|---:|---:|---:|---:|---|
| 64 | 0.2 | 0.001 | 0.041763 | 5 | PASS |
| 64 | 0.2 | 0.0005 | 0.042151 | 8 | PASS |
| 64 | 0.5 | 0.001 | 0.043840 | 7 | PASS |
| 64 | 0.5 | 0.0005 | 0.044091 | 12 | PASS |
| 128 | 0.2 | 0.001 | 0.043023 | 4 | PASS |
| 128 | 0.2 | 0.0005 | 0.044194 | 6 | PASS |
| 128 | 0.5 | 0.001 | 0.045177 | 6 | PASS |
| 128 | 0.5 | 0.0005 | 0.045582 | 7 | PASS |

Nine data artifact hashes matched the frozen manifest. Pre-training losses were
10.1817 and 10.1980 for hidden sizes 64 and 128, respectively, close to the uniform
reference of 10.1622. These checks validate initialization and data integration;
they are not performance results. The selected-checkpoint integrity check passed.

The winning configuration was also trained with seeds 43 and 44. The actual
saved three-seed validation summary is now attached under
`model0_full_grid_20261107/issue3_1_validation_summary.json`. Seeds 42, 43 and 44
achieved validation NDCG@10 of 0.045582, 0.045259 and 0.045375 at best epochs 7,
9 and 8, respectively. The original Colab reload checks reproduced each saved
value. Mean validation NDCG@10 was 0.045405 with sample standard deviation
0.000163. All three downloaded best-checkpoint hashes matched the summary.
Seed 42 selected the winning configuration, so this validation aggregate is not
an unbiased test estimate. Part E has now completed test evaluation: full-catalog
NDCG@10 is 0.038115 ± 0.000311, Recall@10 is 0.064849 ± 0.000553, and MRR is
0.035006 ± 0.000248 (mean ± sample standard deviation, seeds 42/43/44).
See the [completed comparison](model0_final_comparison/README.md).

The full Colab run recorded Python 3.13.16, PyTorch 2.14.1+cu130, NumPy 2.2.6 and
pandas 2.3.2. This differs from the repository's pinned local environment. Code
hashes, frozen data hashes and runtime versions accompany the results; the
notebook ran bundled source files and therefore recorded a null Git commit.
All twelve bundled Python modules matched the local project at output review.

## Comparison and interpretation — contribution to Section 4.3

SASRec Model 0 provides the ID-based sequential reference for the subsequent
models. Its validation NDCG@10 of 0.045582 exceeds the author-produced same-dataset
validation scores for popularity (0.013995), last-item Markov (0.031781) and cosine
item-kNN (0.024794). These values were checked against the baseline JSON in
PR #6 at commit 2cd58b0, whose provenance matches all nine frozen input hashes. Markov is the strongest of these simple references.
Using the unrounded values, the absolute validation difference is
0.013801, or 43.42% relative improvement. This is a descriptive comparison of the selected
seed-42 validation checkpoint against author-reported baselines, not a test-set
comparison or a significance test. The completed full-catalog test comparison
shows a 38.94% NDCG@10 improvement over Markov, with weaker mid- and tail-item
performance than cosine item-kNN. See the linked final comparison for all metrics
and the distinction between seed variation and statistical significance.

Published Amazon Reviews 2023 results provide context rather than a directly
comparable target. The four-paper review covers HSTU-BLaIR, How Well Does Generative
Recommendation Generalize?, DISR and MEMOIR. Their categories, preprocessing,
holdouts and candidate sets differ from this project's Skin Care subset. In
particular, the reviewed DISR experiment uses 99 negative candidates and MEMOIR
uses 999; neither establishes an expected score for our full-catalog ranking.
Candidate-pool details were unresolved in the inspected descriptions of the other
two studies. The current result supports comparison against the team's local
reference models, but does not establish reproduction of a published paper's
numerical performance. Full citations, reported scores and limitations are in the
[research table](../literature/amazon2023_ndcg10_benchmarks.md).

## Final evidence

The [completed comparison](model0_final_comparison/README.md) includes the full test
table, per-group limitations, author attribution and exact source versions.
Test evaluation is credited to Sneha Tumkur Narendra and reference baselines to
Ruchika Chaurasia. This contribution summarizes their evidence alongside Part D
tuning; it does not replace their evaluation or architecture sections.
