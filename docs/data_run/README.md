# Data run record (frozen dataset, Amazon Reviews 2023 Skin Care)

Evidence for the numbers in Workbook 2 sections 3.3 to 3.6 from one end-to-end execution of the seven data-preparation
notebooks. The data files are too large for Git and stay on the shared project Drive.

- Pipeline author: Disha Rawat (clean source notebooks are in `notebooks/data_prep/`).
- Executed and verified end to end by: Ruchika Chaurasia.
- Frozen dataset manifest created: see `manifest.json` (UTC timestamp and software versions).

| Item | What it is |
|---|---|
| `manifest.json` | SHA-256 and size of every processed file, plus software versions |
| `dataset_info.json`, `DATA_CARD.md` | Dataset description, split rule, leakage safeguards, limitations |
| `leakage_report.json` | The 12 automated leakage checks (the 13th safeguard, review snippets from training only, is in `DATA_CARD.md`) |
| `data_prep_config.json` | Data decisions used by the notebooks |
| `records/` | Cleaning log, subset choice, re-indexing record, evaluation protocol, client and cold-start summaries |
| `reports/` | Statistics tables, k-core iterations, quality gate, sanity baselines, figures |
| `executed_notebooks/` | The notebooks with their outputs, as executed |

Not included: the `raw`, `interim` and `processed` data files, and `sasrec_smoke_test.json` (its NDCG@10 and MRR exceed 1,
which is impossible with one target per user; do not cite it).
