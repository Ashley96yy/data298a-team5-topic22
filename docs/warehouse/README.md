# Local warehouse verification

Verified on 2026-11-21T18:28:12.334761+00:00 using Python 3.11.14 and DuckDB 1.4.3.

- Status: **PASS**; all 42 SQL quality checks passed.
- All 22 frozen artifacts matched their manifest hashes and sizes.
- Loaded 13 Parquet tables; 56,017 users, 25,903 items, 501,399 interactions.
- Splits: 389,365 training interactions, 56,017 validation and 56,017 test.
- Build runtime on this local machine: 4.738 seconds (not a cloud performance benchmark).
- Twelve automated tests passed, including rejection cases and existing-output preservation.
- Tested implementation commit: `a58024314cec57fd173cd6bd8bdc520d3b499377` (clean working tree).
- Manifest SHA-256: `e4b7801c0d9ea27d0b39a6af253ac154d6f4a7181d8a7c1dc31d651a050af0e5`.

The local snapshot is `data/warehouse/pr_verification_20261121/warehouse.duckdb`.
The database and source data are excluded from Git. This directory contains only
aggregate verification evidence. Source hashes in the report identify the exact
tested implementation.

See [run instructions](../../src/warehouse/README.md),
[full report](quality_report.json), and [check results](quality_checks.csv).

This verifies local warehouse loading and quality checks. AWS deployment and
Airflow integration have not been performed.
