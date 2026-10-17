# Data preparation: Amazon Reviews 2023, Beauty_and_Personal_Care

Seven notebooks that take the raw dataset to a frozen, leakage-checked dataset for Models 0–4.
Run them **in order, 01 → 07**. Each one reads only files saved by the earlier ones, so you can re-run any single notebook.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_acquisition.ipynb` | Writes `config.json`, downloads the 5-core ratings and metadata, checks the schema and counts, records checksums |
| 02 | `02_eda_and_subset_selection.ipynb` | EDA (activity, long tail, ratings, time, quality, metadata coverage, categories); compares candidate subsets and saves the recommended one |
| 03 | `03_cleaning.ipynb` | Cleans interactions and metadata; logs every step with row counts |
| 04 | `04_subset_kcore_reindex.ipynb` | Applies the subset, iterative 5-core, re-indexes IDs (0 = padding), statistics table |
| 05 | `05_split_eval_groups_clients.ipynb` | Chronological leave-one-out split, evaluation protocol and negatives, user/item groups, cold-start variant, federated clients, leakage checks |
| 06 | `06_item_text_preparation.ipynb` | Clean item text for Model 1 (with training-only review snippets), side info, empty LLM-profile column |
| 07 | `07_freeze_and_sanity_check.ipynb` | Writes the shared `rec_eval.py`, tests it (oracle / random / popularity), writes the manifest and data card |

## Where things are saved
- **Colab:** `MyDrive/298A_recsys/` (Drive is mounted automatically). Large downloads go to `/content/downloads`
  (local disk, faster). If the runtime resets they are downloaded again; the Parquet outputs on Drive are kept.
- **Local Jupyter:** `./298A_recsys/` next to the notebooks.
- To use another folder, set the environment variable `RECSYS_BASE`.

```
298A_recsys/
  config.json            all data decisions (edit only in notebook 01)
  raw/                   ratings + metadata as Parquet, source checksums
  interim/               cleaned data, cleaning log, subset choice
  processed/             FROZEN dataset used by all models (+ rec_eval.py, manifest.json, DATA_CARD.md)
  reports/               figures, statistics tables, EDA summary, sanity baselines
```

## Runtime notes (real data)
- 01: ratings ~1 min; metadata download + parse ~5–10 min.
- 02: subset comparison runs 5-core on ~11 candidates, ~5–10 min.
- 06: the review file is several GB; download + streaming takes ~20–40 min. Set `include_review_text: False` in
  notebook 01's config to skip it (Model 1 then uses metadata text only).
- A standard Colab CPU runtime (12 GB RAM) is enough; no GPU needed for data preparation.

## Changing the subset
Notebook 02 picks the subset with written rules. To override it, set `MANUAL_CHOICE` in section 8 of notebook 02,
e.g. `{'kind': 'category', 'value': 'Skin Care'}`, then re-run 02 → 07.

## Using the frozen data in a model notebook
```python
import sys; sys.path.insert(0, str(PROCESSED)); import rec_eval as R
data = R.load_dataset(PROCESSED)
X_tr, Y_tr = R.train_arrays(data['sequences'], data['max_seq_len'])
res = R.evaluate(score_fn, data, split='val')        # full-catalog ranking, history masked
print(res['metrics']); print(R.group_breakdown(res, data))
```
`score_fn(inputs, users)` returns a `(batch, n_items + 1)` score array; column 0 (padding) is ignored.
