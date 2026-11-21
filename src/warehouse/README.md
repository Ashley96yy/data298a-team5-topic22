# Frozen-data DuckDB warehouse

This loader creates a local analytical copy of the frozen Amazon Reviews
2023 Skin Care dataset. It loads the existing cleaned Parquet files; it does not
repeat cleaning, change train/validation/test splits, or change model inputs.

## Run from the repository root

Python 3.11 is the project environment. DuckDB is a separate dependency so that
training and serving environments do not acquire a warehouse dependency.

```bash
.venv/bin/python -m pip install -r src/warehouse/requirements.txt
.venv/bin/python -m src.warehouse.load \
  --processed data/processed \
  --manifest docs/data_run/manifest.json \
  --output data/warehouse/my_first_run
```

Use a **new output directory for every run**. Existing directories are rejected,
including failed run directories, so a retry cannot overwrite earlier evidence.
All 22 files listed in the trusted manifest must be available under `--processed`.
The committed manifest is the trust reference; do not regenerate it to make an
unexpected dataset pass. A deliberately new dataset needs a separately reviewed
manifest and compatible schema.

## Outputs and publication contract

| File | Purpose |
|---|---|
| `warehouse.duckdb` | Persistent snapshot, published only after checks pass |
| `quality_report.json` | PASS/FAIL, input hashes, row counts, schema, SQL checks, code/environment provenance and runtime |
| `quality_checks.csv` | Aggregate check results, with LF line endings |

The CLI exits with code 0 for PASS and 1 for a failed build. A failed build leaves
its report but removes its unpublished database. An existing output directory is
an error and remains untouched. Consumers must require report status `PASS` and
can verify the database against `database_sha256` before opening it read-only.
The database is a materialized copy and remains queryable without the Parquet
files. Generated snapshots are ignored by Git.

Reports contain schema, hashes and aggregate counts, not user rows or review text.
The database itself includes the frozen source identifiers and reviews and belongs
in controlled data storage, not a GitHub commit or public app artifact.

## Tables and views

Thirteen source tables preserve the Parquet columns:

- `interactions`, `user_map`, `item_map`, `items`, `item_text`, `sequences`
- `train_reviews`, `interaction_side_info`, `eval_cases`
- `item_groups`, `user_groups`, `clients`, `cold_sequences`

`source_files` records all verified input hashes and sizes. `frozen_metadata`
stores the input JSON records. `build_metadata` stores the checked build context.
`sequence_events` expands sequences for comparisons with interaction rows;
`split_summary` exposes aggregate split counts.

```python
import duckdb

with duckdb.connect("data/warehouse/my_first_run/warehouse.duckdb", read_only=True) as db:
    print(db.execute("SELECT * FROM split_summary ORDER BY split").fetchall())
    print(db.execute("SELECT item_group, count(*) FROM item_groups GROUP BY item_group").fetchall())
```

## Verification

Before loading, the loader compares the local manifest with the trusted manifest
and verifies every listed file's SHA-256 and byte count. It rechecks inputs before
publishing. Evaluator code and NumPy artifacts are verified but not executed or
converted into warehouse tables.

The 42 SQL checks cover frozen counts, unique/non-null keys, ID ranges and foreign
keys, user/item k-core counts on the complete dataset, split counts, chronological
positions (including the original row-ID tie-break), sequence/interaction agreement,
evaluation targets, catalog/text alignment, training-review user/item provenance,
and training counts in group/client tables. Training-review provenance checks the
mapped user/item pair against the training split; it does not reconstruct text
snippets or rerun the review-matching pipeline. Cold-start and negative-sampling
artifacts are hash-verified; their full semantic audits remain in the frozen
pipeline/evaluator. This module does not claim to revalidate generated LLM profiles.

```bash
.venv/bin/python -m unittest discover -s tests -p test_warehouse.py -v
```

The tests cover successful persistent loading, missing/corrupt files, a replaced
manifest, unsafe manifest paths, wrong counts, duplicate identities, chronology,
sequence leakage, held-out review provenance, inputs changing during a build, and
preservation of an existing output.

Reference: [DuckDB 1.4 Parquet loading documentation](https://duckdb.org/docs/1.4/data/parquet/overview).
