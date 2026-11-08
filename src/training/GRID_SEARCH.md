# Issue 2.1 — eight-configuration grid search

`src/training/grid_search.py` orchestrates the team's Part C CLI without changing
its model, dataloader, evaluator or checkpoint implementation. Each configuration
gets a fresh process, seeded model, optimizer and dataloader. Jobs run sequentially
to limit memory. Selection uses each run's best epoch validation NDCG@10, never test.

## Preview without training

From the repository root:

```sh
.venv/bin/python -B -m src.training.grid_search --dry-run
```

This enumerates 64/128 hidden size × 0.2/0.5 dropout × 0.001/0.0005 learning rate.
A preview accepts the planning template's nulls; execution rejects them.

## Reproduce the bounded integration check

`configs/part_d_development_smoke.json` contains DEVELOPMENT settings only:
batch 128, two epochs, two layers, one head, weight decay 0, patience 10,
seed 42, Adam and no scheduler. The full config is now agreed and complete;
smoke checks shorten its epoch budget and use a user subset.

```sh
OMP_NUM_THREADS=2 .venv/bin/python -B -m src.training.grid_search \
  --config configs/part_d_development_smoke.json \
  --seed 42 --optimizer Adam --eval-batch-size 128 \
  --limit-train-users 128 --limit-val-users 128 \
  --discard-last-checkpoint \
  --output-dir results/issue2_grid_smoke_new
```

Always choose a new output directory. Relative data/output paths resolve from the
invocation directory; child processes use the repository root. The data path is
made absolute in the saved configuration snapshot.

## Outputs

- `search_config.json`: configuration snapshot used by every job.
- `search_plan.json`: explicit seed, optimizer, sample limits, selection policy,
  accepted tie policy, user-limit indicator and runner source hash.
- `run_00_seed_42/` through `run_07_seed_42/`: Part C's run configuration,
  epoch history, best/last checkpoints, data/code provenance and summary.
- `run_00.log` through `run_07.log`: child-process stdout/stderr.
- `grid_results.csv`: hyperparameters, seed, status, best validation score/epoch,
  elapsed time and run directory; updated after each attempted job.
- `best_config.json`: winning expanded configuration, readable by Part C's
  `resolve_config` without `--grid-index`; its seed list contains the tuning seed.
- `best_checkpoint.pt`: byte-for-byte copy of that run's selected checkpoint.
- `selection.json`: winner, score, checkpoint hash and selection conditions.

The optimizer and execution limits are in `search_plan.json`; the exact trained
model metadata is also in the checkpoint/run metadata. A standalone configuration
file does not override Part C's explicit CLI optimizer requirement.

A failed job stops the search and records `failure.json`. No winner is selected
from an incomplete grid. Equal validation scores choose the first configuration
in the supplied JSON order, with no claim that it is statistically superior.
Existing directories are rejected, including empty ones. Grid-level resume is
not implemented; keep failed-run logs and use a new directory for a retry.

## Seeds and remaining dependencies

The agreed policy tunes all eight configurations on seed 42, then retrains the
winner with seeds 43 and 44. `configs/config.json` contains the fixed settings
and all three eligible seeds. The runner selects one seed explicitly; it does
not repeat the grid across all seeds.

The runner uses current Part C without removed acknowledgment flags and records
the team-accepted average-rank-then-metrics convention. PRs #3/#4 are merged.
Full tuning is ready to execute on suitable compute after the metrics section
4.4 is committed as requested by the team. Do not present smoke metrics as final
Model 0 results.
Checkpoint internals are reused from Part C; Issue 2.2 acceptance remains a
separate review. Large checkpoint files and local results remain Git-ignored.

## Validation

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

New tests cover all eight independent commands, a nonmonotonic winner, ties,
configuration round-trip, missing settings, obsolete-flag exclusion, invalid
scores, existing output directories and failure without partial-grid selection.

## Storage

The optional `--discard-last-checkpoint` deletes ONLY each successfully completed
run's `last.pt` after validating its result. It retains every `best.pt` and logs.
This reduces disk use but removes epoch-resume capability for completed runs.
By default all checkpoints are retained. Allow space for temporary atomic writes
as well as eight runs; even a small user sample has full-catalog model weights.
If storage is exhausted, free space before retrying. Failure-log writes themselves
cannot be guaranteed when no disk space remains.

## Load the selected checkpoint (Issue 2.2)

```sh
.venv/bin/python -B -m src.training.checkpoints --run-dir results/YOUR_GRID
```

`load_selected_model(directory, device='cpu')` returns the restored eval-mode model,
checkpoint metadata, and selection record. It verifies the copied checkpoint hash,
configuration, validation score/epoch and evaluation protocol. Keep
`best_config.json`, `best_checkpoint.pt`, and `selection.json` together when moving
the selected artifacts to Drive. Treat these as trusted project files; the hash
checks consistency, not authenticity. The loader does not access the dataset or test split.

Part C remains responsible for saving `best.pt` and resumable `last.pt` files.
Existing tests verify optimizer/scheduler/RNG restoration against uninterrupted
CPU training for Adam and AdamW. Resume needs `last.pt`, an unchanged model/data
provenance, a larger epoch budget, and a new output directory; it is unavailable
for completed runs if `--discard-last-checkpoint` was used. A selected `best.pt`
is for inference and validation, not a replacement for the resume checkpoint.

To hand the winner to the multi-seed evaluation owner, provide the full winning
run directory, selected configuration, selection record, optimizer, tuning seed,
and search results. `best_config.json` lists only the tuning seed; the agreed
repeat seeds must be added to a separate copy before repeat training. Keep code,
training settings, data and evaluation provenance consistent across those runs.

## Full grid (after smoke verification)

```sh
python -B -m src.training.grid_search --config configs/config.json \
  --seed 42 --optimizer Adam --device cuda --eval-batch-size 128 \
  --output-dir results/model0_full_NEW
```

No user limits are supplied; at most 200 epochs are run per configuration with
patience 10. In the Colab notebook select `RUN_MODE="full"`; full mode writes
artifacts to Drive directly. Use a fresh output directory. Keep the same code
snapshot and runtime versions for seed-43/44 repeats. The full grid completed on 2026-11-07; see
`docs/project_notes/model0_full_grid_20261107/` for evidence extracted from notebook
outputs. The repeat-training and validation-verification commands also succeeded;
the per-seed numeric summary remains in Drive. Final test metrics are still needed
for the performance comparison. The full-run/repeat notebook is
`notebooks/model_experiments/10_sasrec_full_tuning_and_repeats_colab.ipynb`.
