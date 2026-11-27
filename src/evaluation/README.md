# Parts E and F: multi-seed evaluation and cold-start analysis

Part E repeats ONE selected Model 0 configuration over several seeds and scores
the test split once per seed. Part F asks whether Model 0 can recommend products
it never saw in training. Neither part chooses hyperparameters, changes the
frozen data or edits `rec_eval.py`.

## Files

- `multi_seed.py`: `train` calls `src.training.train` once per seed (Part C,
  unchanged). `evaluate` reloads each `best.pt`, checks it reproduces its saved
  validation NDCG@10, then runs the frozen evaluator on test.
- `cold_start.py`: `reference` (popularity and chance, no PyTorch), `train`
  (Model 0 on the cold variant) and `evaluate` (cold and warm cases, over seeds).
- `cold_start_data.py`: loads and checks the notebook 05 cold variant. Pure NumPy/pandas.
- `seed_stats.py`: mean and sample standard deviation (ddof=1) across seeds.
- `common.py`: manifest checks, configuration and partial-run checks, category
  labels, run loading, write-once files.
- `tests/test_evaluation.py`: synthetic fixtures. Model tests need PyTorch and
  `data/processed/rec_eval.py`; without them they are skipped, not passed.

## Part E: multi-seed test evaluation

1. Pick the configuration on validation NDCG@10 only (Part D grid, seed 42).
2. Train the remaining seeds with exactly the same arguments. Everything after
   `--seeds` is forwarded to `src.training.train` unchanged:

   ```sh
   python -B -m src.evaluation.multi_seed train --runs-dir results/model0_seeds \
     --seeds 42 43 44 --config configs/team_confirmed.json --grid-index <best> \
     --optimizer <Adam|AdamW> --device cuda
   ```

   Seeds with a `summary.json` are skipped. A folder without one (failed or
   interrupted) stops the runner; resume it with `src.training.train --resume`
   into a new folder and pass that folder to `evaluate`.
   The seeds must be listed in the config's `training.seeds`. Forwarded
   `--limit-train-users` / `--limit-val-users` are refused unless
   `--allow-partial-runs` is given (development only).

3. Score test and aggregate:

   ```sh
   python -B -m src.evaluation.multi_seed evaluate --device cuda \
     --config configs/team_confirmed.json --grid-index <best> \
     --run-dirs results/model0_seeds/seed_42 results/model0_seeds/seed_43 \
                results/model0_seeds/seed_44 \
     --output-dir results/model0_test_3seeds
   ```

   An existing grid-search run can replace `seed_42` if it was trained by
   `src.training.train` with the same settings and source files.

Before scoring test, `evaluate` refuses:

- Configuration consistency: any run whose recorded settings differ from the team
  config and selected grid configuration (hyperparameters, batch size, epochs,
  layers, heads, weight decay, patience, sequence length), or whose seed is not
  in the config's seeds. A config with empty settings is refused outright.
  Runs must also match each other in everything except seed and device (Git
  commit excepted, source hashes still compared).
- Partial validation: any run trained or validated on fewer than all 56,017
  users, unless `--allow-partial-runs` (development only; never for results).
- Runs on other frozen data (manifest and file hashes).

Each run folder gets `test_evaluation.json` and `test_ranks.npz` (per-user rank
and top-20, for later significance tests and the failure study). They are
written once. Running `evaluate` again reuses them, so test is never re-scored
and a changed `best.pt` is detected. The output folder gets `aggregate.json`,
`evaluation_table.md` (full catalog, 100 sampled negatives, coverage, category
diversity, user and item groups) and `group_breakdown.csv`.

## Part F: cold-start analysis

Notebook 05 removed 1,295 new products from every training sequence in a
separate cold variant. The main split is unchanged: those products are in main
training data for 3,323 users. So the main Model 0 checkpoint cannot be used
for cold-start; `train` fits a Model 0 copy on the cold variant with the
settings of a main run, without retuning.

```sh
python -B -m src.evaluation.cold_start reference --output-dir results/cold_reference
python -B -m src.evaluation.cold_start train --device cuda \
  --config configs/team_confirmed.json --grid-index <best> \
  --reference-run results/model0_seeds/seed_42 --output-dir results/cold_model0_seed42
python -B -m src.evaluation.cold_start evaluate --device cuda \
  --config configs/team_confirmed.json --grid-index <best> \
  --run-dirs results/cold_model0_seed42 results/cold_model0_seed43 results/cold_model0_seed44 \
  --output-dir results/cold_start_3seeds
```

Use `--seed 43` and `--seed 44` with `train` for the other seeds. `train` refuses
a reference run that does not match the team configuration, a seed not in the
config, a partial reference run, and `--limit-*` flags (the last two unless
`--allow-partial-runs`). `evaluate` applies the same configuration check and
refuses runs trained on fewer than all users or validated on fewer than the
55,174 cold-variant validation users.

- Cold cases: the 1,142 users whose test item is a new product. Warm cases:
  the other 54,875 users, scored by the same cold-variant model, as the control.
- Early stopping uses the 55,174 users whose validation item was not removed.
- Test history: notebook 05 marks a removed validation item with -1.
  `test_view` gives `rec_eval` the cold-variant history without passing -1 to
  the model or the mask.
- Removed items the user already had are masked as well, so a model is not
  credited for recommending them again.
- Reported per case group: Recall@10, NDCG@10, MRR, median rank and mean
  relative rank (rank / items left to rank; 0.5 = chance, 1.0 = last), next to
  popularity and the exact chance level.

Reference run on the frozen data (no model, 2026-11-05):

| Scorer, cases | Recall@10 | NDCG@10 | MRR | Median rank | Relative rank |
|---|---|---|---|---|---|
| Popularity, new products | 0.0000 | 0.0000 | 0.00004 | 25,228 | 0.974 |
| Chance level, new products | 0.0004 | 0.0002 | 0.00041 | 12,948 | 0.500 |
| Popularity, warm control | 0.0256 | 0.0134 | 0.01326 | 2,222 | 0.238 |

Model 0 behaves like popularity on new products, not like chance (relative rank
0.940 over seeds 42, 43 and 44; `docs/project_notes/model0_test_20261108`). Full
softmax training uses every catalog item as a negative, so the embeddings of
products that never appear as a target are pushed down. Profile distillation
does not change this, because those ID embeddings are still never trained. The
item-text embedding variant of Model 1 is the one meant to close the gap.

## Model 1 test evaluation

`multi_seed evaluate` also scores the Model 1 runs. It reads each run's recorded
`model_class`: profile-distillation runs (`src.training.train_model1`) are plain
SASRec at inference, and the item-text arms (`src.training.train_item_text`) are
`TextSASRec`, loaded with `load_text_checkpoint` together with their frozen text
buffer. The checks are the same as for Model 0: every run must match the team
configuration and each other except for seed and device, partial runs are refused,
the reloaded checkpoint must reproduce its validation NDCG@10, and test is scored
once per run (a second call reuses `test_evaluation.json`). Each record states the
model class and what identifies it (text-embedding hash and encoder, or profile file
and target hashes), and `test_ranks.npz` has the format `significance.py` reads.

```sh
python -B -m src.evaluation.multi_seed evaluate --device cuda \
  --config configs/config.json --grid-index 7 --label "Model 1 (<arm>)" \
  --run-dirs <seed_42 dir> <seed_43 dir> <seed_44 dir> \
  --output-dir results/model1_<arm>_test_3seeds
```

Run it once per ablation arm, then pass the arm's run directories to
`significance compare --model "Model 1=..."`.

## Failure study

`failure_study.py` implements the failure-taxonomy row of `docs/metrics_commitment.md`.
A failure is a test case whose true item is outside the top 10 in every seed. A fixed
random sample of failures (100 by default, never fewer than 50, seed 2026) gets one
category each, by the first rule that applies: near miss (rank 11 to 20 in some
seed), unpopular target (item group mid or tail), new brand, category switch from the
most recent item, a gap of more than 180 days, or other. The rules were chosen after
looking at the Model 0 failures and are applied unchanged to later models. The study
also reports how common each condition is among all failures and among cases that
are hits in every seed. `cases_for_review.csv` (titles and recent histories of the
sampled users, with empty `manual_category` and `notes` columns) is for manual review
and stays out of Git.

```sh
python -B -m src.evaluation.failure_study --processed data/processed \
  --model "Model 0=<seed_42 dir>,<seed_43 dir>,<seed_44 dir>" \
  --output results/failure_study/model0
```

Model 0 results: `docs/project_notes/failure_study_model0_20261126`.

## Significance tests

`significance.py` implements the significance row of `docs/metrics_commitment.md`
(amendment of 2026-11-22). The unit is per-user NDCG@10 on the full-catalog test
split, averaged over seeds 42, 43 and 44 for a trained model. The test is a
two-sided paired randomization (sign-flip) test with 10,000 permutations, with a
paired bootstrap 95% interval (10,000 resamples), random seed 2026, and Holm
correction across the comparisons in one table. Recall@10 and MRR are reported
with an interval only. Most users score 0 under both models (91% for Model 0),
which is why the Wilcoxon signed-rank test, which drops tied pairs, is not used.

No model is retrained or rescored. Model ranks come from each seed's saved
`test_ranks.npz`, which must reproduce the NDCG@10 in its `test_evaluation.json`.
The untrained baselines are rescored once and must reproduce
`docs/baselines/baselines.json`.

```sh
python -B -m src.evaluation.significance baselines --processed data/processed \
  --output results/significance/baseline_ranks
python -B -m src.evaluation.significance compare \
  --model "Model 0=<seed_42 dir>,<seed_43 dir>,<seed_44 dir>" \
  --baseline-ranks results/significance/baseline_ranks \
  --output results/significance/model0_vs_baselines
```

Add `--model "Model 1=..."` once its three seeds are evaluated; the command runs
every committed comparison the inputs allow and lists the ones still missing.
Results: `docs/project_notes/significance_20261122`.

## Tests

```sh
python -B -m unittest tests.test_evaluation -v
python -B -m unittest tests.test_significance -v
```

Run in the pinned Python 3.11 / PyTorch 2.8.0 environment with
`data/processed/` present, otherwise the model tests are skipped.
