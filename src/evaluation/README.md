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

Expect Model 0 to look like popularity on new products, not like chance. Full
softmax training uses every catalog item as a negative, so the embeddings of
products that never appear as a target are pushed down. That is the gap Model 1
(item text embeddings) is meant to close.

## Tests

```sh
python -B -m unittest tests.test_evaluation -v
```

Run in the pinned Python 3.11 / PyTorch 2.8.0 environment with
`data/processed/` present, otherwise the model tests are skipped.
