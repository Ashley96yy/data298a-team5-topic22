# Local warehouse verification

Verified on 2026-11-23T00:18:02.841740+00:00 using Python 3.11.14 and DuckDB 1.4.3.

- Status: **PASS**; all 42 SQL quality checks passed.
- All 22 frozen artifacts matched their manifest hashes and sizes.
- Loaded 13 Parquet tables; 56,017 users, 25,903 items, 501,399 interactions.
- Splits: 389,365 training interactions, 56,017 validation and 56,017 test.
- Build runtime on this local machine: 4.237 seconds (not a cloud performance benchmark).
- Nineteen automated tests passed, including rejection cases and existing-output preservation.
- Tested implementation commit: `68a5116b2f79899fd21dd7ebf5c5b33dcf10398f` (clean working tree).
- Manifest SHA-256: `e4b7801c0d9ea27d0b39a6af253ac154d6f4a7181d8a7c1dc31d651a050af0e5`.

The local snapshot is `data/warehouse/review_fixes_20261122/warehouse.duckdb`.
The database and source data are excluded from Git. This directory contains only
aggregate verification evidence. Source hashes in the report identify the exact
tested implementation.

See [run instructions](../../src/warehouse/README.md),
[full report](quality_report.json), and [check results](quality_checks.csv).

Regression coverage includes zero-interaction catalog items, failures writing either
report, atomic bundle publication, unavailable Git, and concurrent-output protection.
The CSV and JSON checks were compared field by field.

This verifies local warehouse loading and quality checks. AWS deployment and
Airflow integration have not been performed.
