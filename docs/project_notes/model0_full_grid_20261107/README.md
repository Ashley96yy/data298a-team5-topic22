# Model 0 full-grid evidence — 2026-11-07

Written by: Abhinita Sanabada

The original grid records were extracted from the saved outputs of the executed Colab notebook
`10_sasrec_full_tuning_and_repeats_colab.ipynb`. Those original extracts are not an independent rerun or
a download of the underlying Drive artifacts. The later handoff attachment below
includes files retrieved directly from Drive. `notebook_review.json` records the
notebook checksum, source-code comparison, output folder and evidence limits.

All eight seed-42 configurations passed, with no training or validation user limits.
Selection used full-catalog validation NDCG@10. The winning configuration was hidden
size 128, dropout 0.5, learning rate 0.0005; validation NDCG@10 was
0.04558158609645662 at best epoch 7. The sum of recorded grid-run durations was
approximately 46.88 minutes, excluding setup and repeat training.

Initial losses were 10.1816587 (hidden size 64) and 10.1980295 (128), against a
uniform reference of 10.1621527. All nine data artifact hashes matched the manifest.
All twelve bundled Python modules matched local project source at review time.
The Colab runtime recorded Python 3.13.16, PyTorch 2.14.1+cu130, NumPy 2.2.6 and
pandas 2.3.2; these differ from the local test environment and are preserved here.

The selected-checkpoint check, seed-43/44 training and final three-checkpoint
validation verification each returned exit code zero. The notebook did not embed
the per-seed repeat scores or their aggregate. The actual Drive summary has now
been attached as `issue3_1_validation_summary.json`: validation mean
0.045405340349347866, sample standard deviation 0.0001633599310770804.
`part_e_handoff.md` links the complete three-seed folders; `handoff_verification.json`
records independently checked checkpoint hashes and saved configuration checks.
This follow-up does not independently recompute predictions.
Final test results and the test-set comparison remain pending Part E evaluation.
Do not label these validation scores as test performance.

`best_config.json` preserves the exact Colab paths from the output. For use outside
that runtime, copy it and change the data/output paths; preserve this evidence copy.
Keep datasets and checkpoint binaries in Drive, not GitHub.

For future runs, Notebook 10 enforces `configs/colab_environment.json` through
`src.training.environment_check` before training. The historical run predates
this gate; its actual version records are retained. See `part_e_handoff.md` for
scope, remaining Part E checks and recipient-access confirmation.
