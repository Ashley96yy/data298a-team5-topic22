# Model 0 three-seed test evaluation and cold-start analysis, 2026-11-08

Written by: Sneha Tumkur Narendra

Parts E and F run on the three Model 0 runs from Abhinita's grid search
(`docs/project_notes/model0_full_grid_20261107`): grid index 7, hidden size 128,
dropout 0.5, learning rate 0.0005, seeds 42, 43 and 44. The test split was scored
once per seed. These are the first Model 0 test results.

Checkpoints (`best.pt`, `last.pt`) and per-user ranks (`test_ranks.npz`) are kept in
the shared Drive folder `298A/part_e_f_results/model0_real_results`
([part_e_f_results](https://drive.google.com/drive/folders/1e6zKWNpx6PA8WIMIMvGBSIT7dD5a7Vrc)), not in GitHub.
This folder holds the small result files only.

## Checks

- All 96 repository tests passed in the run environment.
- Each `best.pt` matched the checkpoint hash in Abhinita's `handoff_verification.json`.
- Each reloaded checkpoint reproduced its saved validation NDCG@10 exactly (gap 0.0).
- All runs matched `configs/config.json` with `--grid-index 7` and used all 56,017 users.
- The three cold-start runs trained on all 56,017 users and validated on the 55,174
  users who keep a validation case in the cold variant. Each reused the settings of
  the main run with the same seed (`reference_run_config_sha256` in `cold_runs/`).
- Evaluation code matched main at `a13ad30` (`cold_source_sha256` in `cold_runs/`).

Environment: Colab GPU, Python 3.13.16, PyTorch 2.14.1+cu130, NumPy 2.2.6,
pandas 2.3.2, PyArrow 21.0.0, the same versions recorded by the training runs.

## Part E: test results

Mean ± sample standard deviation over 3 seeds, 56,017 test users, full-catalog ranking.

| Model | Recall@10 | NDCG@10 | MRR |
|---|---|---|---|
| Model 0 (SASRec) | 0.0648 ± 0.0006 | 0.0381 ± 0.0003 | 0.0350 ± 0.0002 |
| Popularity | 0.0251 | 0.0131 | 0.0130 |

Validation NDCG@10 for the same runs was 0.0454 ± 0.0002. Full tables, including
100 sampled negatives, coverage, category diversity and user and item groups, are in
`test_3seeds/evaluation_table.md`.

## Part F: cold-start

1,142 test cases whose true item is a new product removed from training, and the
other 54,875 users as a warm control, scored by Model 0 trained on the cold variant.
Relative rank 0.5 = chance, 1.0 = ranked last.

| Scorer, cases | NDCG@10 | MRR | Median rank | Relative rank |
|---|---|---|---|---|
| Model 0, new products | 0.0000 ± 0.0000 | 0.00004 ± 0.00000 | 24,947 ± 53 | 0.940 ± 0.006 |
| Popularity, new products | 0.0000 | 0.00004 | 25,228 | 0.974 |
| Chance level, new products | 0.0002 | 0.00041 | 12,948 | 0.500 |
| Model 0, warm control | 0.0387 ± 0.0002 | 0.03527 ± 0.00030 | 1,124 ± 11 | 0.134 ± 0.001 |
| Popularity, warm control | 0.0134 | 0.01326 | 2,222 | 0.238 |

Model 0 ranks products it never saw in training far below chance, close to popularity.
The warm control matches the main test result, so the cold-variant models trained normally.

## Files

| Folder | Contents |
|---|---|
| `test_3seeds/` | `evaluation_table.md`, `aggregate.json`, `group_breakdown.csv` |
| `test_runs/` | `test_evaluation.json` of each seed (metrics, groups, validation check, checkpoint hash) |
| `cold_reference/` | Popularity and chance references, no model |
| `cold_start_3seeds/` | Cold-start table and `cold_start.json` over the three seeds |
| `cold_runs/` | `run_config.json`, `summary.json`, `history.csv` and `cold_evaluation.json` of each cold-variant run |

## Commands

Run from the repository root with the frozen data in `data/processed`. The Colab notebook that ran
them is `notebooks/model_experiments/12_model0_test_and_cold_start_colab.ipynb`.

```sh
python -B -m src.evaluation.multi_seed evaluate --device cuda \
  --config configs/config.json --grid-index 7 \
  --run-dirs runs/seed_42 runs/seed_43 runs/seed_44 --output-dir model0_test_3seeds
python -B -m src.evaluation.cold_start reference --output-dir cold_reference
python -B -m src.evaluation.cold_start train --device cuda \
  --config configs/config.json --grid-index 7 \
  --reference-run runs/seed_42 --output-dir cold_model0_seed42   # also 43, 44
python -B -m src.evaluation.cold_start evaluate --device cuda \
  --config configs/config.json --grid-index 7 \
  --run-dirs cold_model0_seed42 cold_model0_seed43 cold_model0_seed44 \
  --output-dir cold_start_3seeds
```
