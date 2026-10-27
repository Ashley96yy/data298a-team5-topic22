# sequence_dataloader.ipynb (Part B: data loader)

**Folder:** `data/` · **Runs after:** notebooks 01–07 · **Runtime:** ~2–4 min on a Colab GPU

## What it does
Turns each user's purchase history in `sequences.parquet` into PyTorch batches for SASRec (Model 0) and all later models.

| Mode | Input to the model (50 product IDs, padded with 0 on the left) | Target |
|---|---|---|
| `train` | training history without its last item | the same history shifted one step (next item at every position) |
| `val` | last 50 training items | `val_item` |
| `test` | last 50 of training items + `val_item` | `test_item` |

Example: history `[12, 45, 7, 88, 30]`, `val_item = 51`, `test_item = 9`
- train: `[0,…,12,45,7,88]` → `[0,…,45,7,88,30]`
- val: `[0,…,12,45,7,88,30]` → `51`
- test: `[0,…,45,7,88,30,51]` → `9`

**Main pieces (copy sections 2–4 into the SASRec notebook):**
- `SequenceDataset(sequences, mode, max_len=50, sliding_window=False)`: the PyTorch Dataset
- `make_dataloaders(...)`: train (shuffled, seed 42), val and test loaders
- `evaluate_loader(score_fn, loader)`: GPU NDCG@K, Recall@K and MRR for per-epoch validation and early stopping. It ranks against the full catalog and hides items the user already bought, using the same rules as `rec_eval.py`.

## Input
Folder: `MyDrive/298A/04_Data/NewDataRunFiles/processed/` (shared folder; needs a shortcut in My Drive)

| File | Required? | From |
|---|---|---|
| `sequences.parquet` | yes | notebook 05 |
| `dataset_info.json` | optional (gives `n_items`, `max_seq_len`) | notebook 07 |
| `rec_eval.py`, `eval/`, `groups/` | optional (only for the agreement test 5c) | notebooks 05 and 07 |

## Output
Saved in the same `processed/` folder:

| File | Contents |
|---|---|
| `08_history_length.png` | chart of training history length per user, with the 50-item cut-off |
| `dataloader_summary.json` | users, items, rows per split, % of long histories, `all_tests_passed` |

The datasets and batches themselves are **not saved**; they are rebuilt from `sequences.parquet` in about 1 second.

## Results (run of 2026-10-26)
| Item | Value |
|---|---|
| Users / products | 56,017 / 25,903 (IDs 1–25,903, 0 = padding) |
| Rows: train / val / test | 56,017 / 56,017 / 56,017 (219 / 110 / 110 batches) |
| Median training history | 4 items; 1.17% of users have more than 50 |
| Padding share per training row | ~89% |
| `sliding_window=True` | 57,163 training rows (extra rows for 631 long-history users) |
| Build time / one epoch of batches | 1.3 s / 0.4 s |
| All checks (5a–5e) | PASS, including identical arrays and metrics vs `rec_eval.py` |
| **Popularity baseline, validation** | **NDCG@10 = 0.0142, Recall@10 = 0.0273**: the minimum SASRec must beat |

**Where to find them:** printed in the notebook cells (sections 1, 3 and 5), and in `dataloader_summary.json`.

## Who uses this
- **Part A (SASRec):** copies sections 2–4 and follows the interface table in section 6:
  - input `(batch, 50)` int64, padding 0 on the left, most recent item last
  - item embedding `N_ITEMS + 1` with `padding_idx=0`, and a causal mask
  - loss over the full catalog, only on positions where the target is > 0 (saves ~10× memory)
  - scores `(batch, N_ITEMS + 1)` from the last position
- **Training diagnostics:** `evaluate_loader` gives the per-epoch validation NDCG@10. The training loop must log epoch, train loss and validation metrics to a file.
- **PR2 data section:** the chart and the numbers above.

## Notes
- `TinyStandIn` in test 5d is a smoke-test model, **not** Model 0.
- To change the data folder, edit `DATA_DIR` in the second code cell.
