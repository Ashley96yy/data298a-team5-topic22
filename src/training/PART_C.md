# Part C training loop

This is the training component, not a new model or a grid-search runner. It
connects Part B batches, the current provisional model adapter, Part D settings,
and the frozen shared evaluator. It does not select the final Model 0 design,
fill unconfirmed settings, modify frozen data, or evaluate the test split.

## Files and ownership boundaries

- `src/data/sequence_dataset.py` is an importable adapter derived from sections
  2-3 of the teammate's `sequence_dataloader.ipynb`. The notebook is unchanged.
  It preserves the sequence, padding, target-shift and full-history contract;
  the parity test compares its arrays with the actual notebook definitions.
  Histories are copied before conversion to avoid read-only NumPy warnings.
- `src/training/trainer.py` trains one run, validates after each epoch, saves
  checkpoints, stops early, logs diagnostics, and supports epoch-boundary resume.
- `src/training/train.py` selects one Part D configuration and one seed, verifies
  the data through the existing integration checker, and connects the components.
- `tests/test_training.py` uses synthetic fixtures. It is not evidence of Amazon
  recommendation quality, paper reproduction, or final hyperparameter selection.

The model must provide `training_logits(x, y)` and `next_item_scores(x)`.
The current `SASRec.forward()` already returns catalog logits, not hidden states.
Do not multiply those logits by the embedding table a second time. A future
approved Part A implementation can satisfy these two adapter methods without
rewriting the outer training loop.

## Run controlled checks first

From the repository root, use a Python 3.11 environment with `requirements.txt`:

```sh
python -B -m unittest discover -s tests -v
python -B -m src.training.train --synthetic-smoke \
  --output-dir results/part_c_synthetic_smoke
```

Choose a NEW output directory every time, even when resuming. The synthetic smoke
uses 8 invented users, 30 items, a tiny CPU model and at most 3 epochs. Its local
deterministic-tie evaluator is only a test fixture; it does not replace
`rec_eval.py` for Amazon experiments. It also reloads the selected checkpoint and
checks that its validation score is reproduced. None of its metrics are research
results or approved full-training defaults.

## One provisional real-data run

Before running, the team must fill the null settings in a separate approved copy
of `configs/config.json`. Do not replace the data-preparation config in Drive.
The runner accepts either that Part D grid template plus `--grid-index 0..7`, or
one dictionary returned by `expand_grid()` with `model_hyperparameters` present.
It does not launch all configurations or repeat seeds automatically.

After settings and implementation review, a development command is:

```sh
python -B -m src.training.train --config configs/team_confirmed.json \
  --grid-index 0 --seed 42 --optimizer Adam --device cpu \
  --limit-train-users 256 --limit-val-users 128 \
  --allow-unapproved-model --allow-legacy-tie-metrics \
  --output-dir results/part_c_development_seed42
```

`configs/team_confirmed.json` is a path for a future team-approved configuration,
not a file supplied here. Seed 42 must actually be in its configured seed list.
Adam above is an explicit development choice, not an assumption that it has been
approved over AdamW. The user limits are for connectivity only; omitted limits
use all users. Use `--device cuda` only in a compatible GPU environment.

Two explicit acknowledgments prevent accidental claims of final results:

1. The existing model is a TinySASRec adaptation, not a validated reproduction
   of the SASRec paper. Team adoption of Model 0 is still pending.
2. The frozen evaluator uses an unresolved average-rank tie policy. Agreement
   between two copies of that evaluator is not proof of metric correctness.
   Part C rejects NaN/Inf scores, ranks below 1, and metrics outside [0, 1], but
   does NOT silently change this shared protocol. Real-data runs remain labelled
   `provisional_development_not_final_baseline`. Resolve and version the shared
   tie policy before final experiments, then update this guard with its owner.

The source manifest hashes are checked before importing `rec_eval.py`. Only use
reviewed, trusted source files and manifests. This checks consistency with the
supplied manifest, not the authenticity of an arbitrary downloaded manifest.

## What the loop does

- Uses cross entropy at every nonpadding target position, with loss averaged
  over target positions rather than equally over batches. Only valid positions
  are projected to the full item catalog, avoiding padding-logit allocations.
- Supports explicitly selected Adam or AdamW, gradient clipping, and optional
  StepLR via BOTH `--lr-step-size` and `--lr-gamma`. No schedule is used otherwise.
- Calls the frozen `rec_eval.evaluate` with `split='val'`, `sampled=False`, and
  full-history masking; it does not use test scores to train, stop, or select.
  The shared loader reads test artifacts for consistency, not test evaluation.
- Saves the highest validation NDCG@10 checkpoint. Equal scores do not replace
  it. `min_delta` affects early-stopping patience, not best-checkpoint selection.
- Writes per-epoch loss, target count, learning rate, validation metrics and wall
  times for the teammate's training diagnostics.
- Rejects nonfinite logits/loss/gradients/scores and invalid ranks/metrics. A
  failed run writes `failure.json`; it is not marked PASS.

## Outputs and resume

Each run writes `run_config.json`, `history.csv`, `last.pt`, `best.pt`, and
`summary.json`. Checkpoints/large outputs belong in Drive or ignored `results/`,
not GitHub. Metadata records source code hashes, Git commit, environment versions,
data-manifest/consumed-file hashes, selected hyperparameters, seed, and sample size.
Successful completion is a training-workflow check, not evidence of good accuracy.

`best.pt` is for inference or downstream evaluation via `load_best()`.
`last.pt` additionally contains optimizer/scheduler state, best weights, early-stop
state, training history and Python/NumPy/PyTorch/DataLoader RNG state. Use only
trusted project checkpoints; loading uses `weights_only=True`.

Resume into a NEW output directory with the same command plus:

```sh
--resume results/previous_run/last.pt
```

Increase the maximum epoch budget in the confirmed configuration. Other training
settings, model/data/code provenance, seed and batch sizes must match. Already
early-stopped runs are not resumed. Reproducible resume is supported at epoch
boundaries with `num_workers=0`, not midway through an epoch. Seed the model
before constructing it when calling the library directly. Cross-hardware bitwise
identity is not claimed.

References: PyTorch's [checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
and [Adam documentation](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html).
