# Model 0 handoff to Part E

Written by: Abhinita Sanabada

The full seed-42 grid and selected-config repeats are complete. These are validation
results; no test metrics are claimed. The source summary was retrieved from Drive,
not reconstructed from rounded notebook output.

[Full run folder](https://drive.google.com/drive/folders/1kgzCRnc1RtyCNhQ8g7C72sGV2doe7sJA)
contains the selection, configurations and original run records.
[Original validation summary](https://drive.google.com/file/d/1vK1Us7ppmrjh67B0gUcrVV72GWaZetXe/view)
is also attached here as `issue3_1_validation_summary.json` without changing its content.

| Seed | Best epoch | Saved validation NDCG@10 | Recomputed in original Colab verification | Complete run folder |
|---|---|---|---|---|
| 42 | 7 | 0.045581586096 | 0.045581586096 | [seed 42](https://drive.google.com/drive/folders/1GM5o-YVXlCHgSr5Tzc2aM2DK5eS6jz6T) |
| 43 | 9 | 0.045258997681 | 0.045258997681 | [seed 43](https://drive.google.com/drive/folders/1uJeqpBf9oNW6ATb6sVYKaccRryVS8tV5) |
| 44 | 8 | 0.045375437271 | 0.045375437271 | [seed 44](https://drive.google.com/drive/folders/1fI4JLTjvE5XrMFYG4dDzBIXC-_WeAJmG) |

Validation mean: **0.045405340349347866**; sample standard deviation (ddof=1):
**0.0001633599310770804**. Seed 42 selected the configuration, so this aggregate
is not an unbiased final test estimate.

Each folder lists best.pt, last.pt, run_config.json, history.csv and summary.json.
The three best.pt files were downloaded and their bytes hashed independently;
all hashes match the original summary. Copies of each original run_config.json
are attached, and all pass the strengthened shared configuration checks.
The report also verifies full user counts, matching run provenance and aggregate
arithmetic. See `handoff_verification.json` for hashes and the exact check scope.
Predictions were not independently recomputed in this follow-up. Part E still
needs to reload the checkpoints, reproduce validation scores and evaluate test.

## Environment

All three original run records state Python 3.13.16, torch 2.14.1+cu130,
NumPy 2.2.6 and pandas 2.3.2. PyArrow 21.0.0 was specified by the notebook install
cell; the old per-run metadata did not record it. Keep these historical records
unchanged. Repository pins (Python 3.11.13 / torch 2.8.0) describe a different
reference environment. The follow-up tests ran locally on Python 3.11.14 / torch
2.8.0: 66 tests passed, zero skipped.

Notebook 10 now bundles `src/training/environment_check.py` and an explicit
`configs/colab_environment.json` target. It checks exact Python/library versions
and CUDA availability in a fresh process before data loading, grid training and
repeat training. Drift stops execution. New runs save environment reports with
actual versions, CUDA/GPU details and pip freeze in the run folder. This is a
preflight specification, not an installer or a claim that the historical Colab
runtime can currently be provisioned. If it is unavailable, agree on a new target
for a separate experiment; do not change historical metadata or mix runtime
versions across seeds. No full-grid rerun is required just for these review fixes.

## Receiving the runs

Sneha should confirm access to all three folder links and use the original full
folders, including their configurations and metadata. The connector reports the
folders as shared, but recipient-specific access has not been confirmed. No
sharing permissions were changed by this follow-up.

Use `winner_repeat_config.json` from the full-run folder, adjusting only the local
processed-data path in a working copy. Keep the historical evidence unchanged.
Run Part E against all three run folders; the common checks now enforce the
configured optimizer and require both scheduler fields to be explicitly null
when scheduler is none. Keep test output separate from the validation report.
