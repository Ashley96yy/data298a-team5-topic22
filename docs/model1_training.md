# Model 1 training entry point

`src/training/train_model1.py` trains ONE configuration and ONE seed of the LLM-distilled SASRec student. It reuses Model 0's data
loading, verified evaluator, model, trainer and 8-configuration grid (`src.training.train`), and adds the frozen profile targets,
the two-stage schedule (`src/models/distill.py`) and the trainer hook (`epoch_fn`, `selection_start_epoch`, PR #11).

```sh
python -m src.training.train_model1 --config configs/config.json --grid-index 7 --seed 42 --optimizer Adam --device cuda \
    --targets profiles.npz --alpha 0.6 --stage1-epochs 20 --output-dir results/model1/g7_s42_a0.6_e20
```
`--alpha` and `--stage1-epochs` above are placeholders until the team decides them (docs/model1_spec.md, section 7).

## Rules built into the script
- A real run refuses synthetic profiles and profiles not attested as built from **training items only**.
- PCA (to the model's hidden size, standardised) is fit on the profile users only, never on validation or test information.
- Stage 1 (`--stage1-epochs`, at least 1, at most epochs minus 1) distils; checkpoint selection and early stopping start after it.
- The run metadata records the schedule, the profile file's SHA-256, the reduced targets' SHA-256 and the source hashes of
  `distill.py` and `train_model1.py`, next to the usual provenance. Resuming needs the same hook, the same targets and the same schedule.
- `distill_log.csv` in the run folder has the stage, alpha, recommendation loss and distillation loss for every epoch
  (`distill_log_resumed.csv` after a resume, covering the resumed epochs only).
- `--synthetic-smoke` checks the plumbing on a tiny synthetic fixture. Its output is never a result and must not be reported.

## Profile file (what the profile pipeline must produce)
A NumPy `.npz` written with `src.data.profile_file.write_profile_file`, with:

| Array | Meaning |
|---|---|
| `user_idx` | int64, unique user ids (0 to n_users - 1); users without a row get no distillation term |
| `embeddings` | float, shape (n, d): one text embedding per user, finite |
| `source` | `llm_profile`, `item_text_centroid` or `synthetic` |
| `train_items_only` | `True`: the producer attests that every profile used only training items |

Controls worth running next to the LLM profiles: `item_text_centroid` (the mean of a user's training items' text embeddings, no
LLM) shows how much of any gain needs an LLM at all.
