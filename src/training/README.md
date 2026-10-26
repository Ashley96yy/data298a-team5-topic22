# Issue 1 — SASRec experiment setup

This implementation adapts the team's Drive Notebook 08 TinySASRec for a bounded
real-data integration check. It is not yet a validated reproduction of the original
SASRec paper and does not run the full grid search (Issue 2).

## Steps to reproduce

Run commands from the repository root with Python 3.11.

1. Create the environment and install dependencies:

   ```sh
   python3.11 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

2. Copy the required frozen Drive artifacts into `data/processed/`, preserving
   subdirectories. The local workspace already contains this required subset.
   `load_dataset` requires dataset_info.json, rec_eval.py, sequences.parquet,
   eval/eval_protocol.json, eval/neg_val.npy, eval/neg_test.npy,
   eval/eval_cases.parquet, groups/item_groups.parquet, groups/user_groups.parquet.
   Also include manifest.json. The checker verifies all nine consumed artifacts
   against its SHA-256 entries before importing the shared evaluator. It does not
   claim to verify every artifact listed in the full manifest.

   Source: https://drive.google.com/drive/folders/11-UfjS51JY1ylATnBm7wcLfDpueOL637

3. Validate the configuration and run tests:

   ```sh
   .venv/bin/python -m src.training.config
   .venv/bin/python -m unittest discover -s tests -v
   ```

4. Run the small training/validation check:

   ```sh
   .venv/bin/python -m src.training.integration_check \
     --processed-dir data/processed \
     --output-dir results/issue1_smoke
   ```

   Choose a new output directory for subsequent runs. By default this uses the
   first 256 training users, first 128 validation users, two epochs, batch size 16,
   CPU, seed 42, hidden size 64, dropout 0.2, four heads, two layers, learning rate
   0.001, and AdamW weight decay 0.00001. These are explicit connectivity settings,
   not approved full-experiment settings. The smaller batch controls CPU memory.
   It uses the unchanged shared evaluator with `split='val', sampled=False`.
   The catalog is complete even though the validation-user sample is small.

5. Inspect `results/issue1_smoke/integration_report.json`. It records losses,
   validation metrics, versions, checksum evidence, actual sample sizes, runtime,
   and whether test evaluation occurred (false). A low or zero NDCG is possible
   for this deliberately short run; it is not the final Model 0 result.

6. Before full tuning, confirm the null settings in `configs/config.json` with
   the team, then run:

   ```sh
   .venv/bin/python -m src.training.config --require-ready
   ```

   This checks configuration completeness, not actual training execution. The
   data path is relative to the repository-root working directory. Do not replace
   the data-preparation config.json in shared Drive. Decide whether reporting
   seeds 42/43/44 applies to all eight configurations (24 jobs) or only repeats of
   the selected configuration. Epoch budget and early-stopping policy also need
   agreement. No full experiment has been launched with guessed settings.

## Model and evaluator interface

`src/models/sasrec.py` adapts the model in
`notebooks/model_experiments/08_validation_and_sasrec_smoke_test.ipynb`.
Source provenance is in `configs/smoke_test_reference.json`.

- Dropout, hidden size, number of layers, and heads are configurable.
- Causal attention prevents future-token access. Padded queries can attend only
  their own padding position, avoiding fully masked attention rows; valid queries
  cannot attend padding. Final padded representations are zeroed.
- Training projects only nonpadding target positions into the catalog, reducing
  peak memory. This retains the notebook's tied item embeddings and cross entropy.
- `next_item_scores(tokens)` returns `(batch, n_items + 1)` scores.
- The shared evaluator handles padding exclusion and history masking.
- No test metrics are computed. The shared loader does read test metadata and
  negative arrays; their presence does not authorize tuning on test performance.

Data checks validate dataset identity, Skin Care subset, sequence length 50,
padding index, manifest hashes, user counts/uniqueness, ID ranges, minimum history,
no repeat/holdout overlap, total interaction count, and negative-array row counts.
They do not independently reproduce every ETL or timestamp-leakage check.

## Remaining work

Issue 1's integration and config implementation can be reviewed locally. Full
training settings and team adoption of this smoke-model adaptation remain open.
Issue 2 will add grid execution, best-epoch selection, and checkpoint management.
Changes are local, not pushed to GitHub or posted to Linear.

## Completed local verification

Six tests passed. The actual real-data integration report is at
results/issue1_smoke/integration_report.json. All nine required artifact hashes
matched the supplied manifest. The run trained on 256 users for two epochs and
evaluated 128 validation users against all 25,903 items. Loss decreased from
34.613591 to 28.821976. Validation NDCG@10 was 0.0; no test metrics were computed.
This proves connectivity and finite training/evaluation, not baseline quality.
