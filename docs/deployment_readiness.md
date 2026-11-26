# Demo deployment preparation

Written by: Abhinita Sanabada

Status updated November 25, 2026 (Pacific time).

The demo was verified locally with popularity, Markov and the original Model 0
seed-42 checkpoint. On November 10, 2026, all **30 existing serving and UI tests
passed with zero skips**, including the real-checkpoint Streamlit AppTest.
This verifies local behavior; no hosted deployment or browser-session acceptance
is claimed. The frozen artifacts and checkpoint were downloaded and checked,
not retrained or re-evaluated as a benchmark.

## AWS deployment status

The team AWS account and pipeline host are now configured. The following results
come from the operator's Session Manager output and shutdown confirmation;
this documentation update did not start EC2 or independently query AWS.

- **Frozen data and warehouse:** all 22 dataset files and the pinned manifest
  were verified. The DuckDB 1.4.3 load passed 42 quality checks. The database,
  JSON quality report and CSV checks were uploaded to
  `s3://data298-team5-artifacts-2026/warehouse/aws_run_001/` and downloaded
  again for byte-for-byte comparison. See the
  [AWS dataset and warehouse runbook](../scripts/aws/README.md).
- **Airflow:** all nine tasks, including `quality_gate` and
  `publish_run_record`, succeeded on November 24, 2026, 19:24–19:33 Pacific
  (November 25, 03:24–03:33 UTC). See the
  [deployment record](deployment/aws_airflow.md) for the run ID and task evidence.
  This metadata/ratings refresh used `skip_reviews=True`; it does not replace
  the approved frozen dataset or the original Model 0 evaluation inputs.
- **Shutdown:** the DAG was confirmed paused after all tasks succeeded. The
  operator confirmed EC2 was stopped on November 25. Leave it stopped until
  the next planned deployment task.
- **Still pending:** a verified S3 archive of the Airflow refresh outputs has
  not been recorded. The earlier warehouse S3 verification covers only the
  three warehouse outputs, not this new refresh. Archive the refresh and verify
  readback at the next planned EC2 session using the Airflow runbook.

This completes the recorded warehouse and Airflow validation, but does not
establish a hosted Streamlit demo or Model 1 serving deployment. No hosted demo
URL or browser acceptance result is recorded.

## Prepared local files

The paths below are relative to the repository root of the prepared local
checkout; a fresh Git clone does not contain these ignored artifacts.

- `data/processed/items.parquet`: matches the frozen manifest.
- `app/artifacts/`: item display table, popularity counts, Markov transitions,
  200 example training histories and an artifact manifest with hashes.
- `app/model/model0_seed42.pt`: the original winning checkpoint, SHA-256
  `54ca6cba54f956af7b6189c5c0b98cf30d55e6412f412f7b92b184d89cbd89e9`.

These directories are ignored by Git. Generated artifacts contain product data
and example user histories; the team’s artifact/access decision is still pending.
No artifacts, histories or weights were uploaded to GitHub by this preparation.

## Run the prepared demo locally

From the project folder:

```sh
source .venv/bin/activate
export SASREC_CHECKPOINT="$PWD/app/model/model0_seed42.pt"
export SASREC_CHECKPOINT_SHA256=54ca6cba54f956af7b6189c5c0b98cf30d55e6412f412f7b92b184d89cbd89e9
python -m streamlit run app/Recommendations.py --server.address 127.0.0.1
```

Open the local address printed by Streamlit. Select `sasrec`, load a real example
user, and verify that ten live recommendations appear. An empty history uses
popularity even when SASRec is selected. Check the pipeline-monitor page as a
separate view. Stop the local server with Ctrl+C.

## Rebuild on another machine

Use Python 3.11 and a reviewed checkout. Install `app/requirements-sasrec.txt`
for the trained SASRec demo; `app/requirements.txt` alone serves only the baseline
models and does not install PyTorch.

```sh
python -m pip install -r app/requirements-sasrec.txt
python -m src.serving.recommend --processed data/processed --out app/artifacts
```

The builder needs the original `manifest.json`, `items.parquet` and
`sequences.parquet`. Transfer the verified checkpoint separately. Use both
SASREC environment variables above; the adapter verifies checkpoint and artifact
hashes and the frozen-data identity before registering the model.

For a host that supports an explicit start command, set its working directory to
the repository root and use:

```sh
python -m streamlit run app/Recommendations.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"
```

The artifacts must exist at `app/artifacts/` when the app starts. The model path
is configurable through `SASREC_CHECKPOINT`. A host that automatically discovers
requirements needs its install behavior checked: selecting the baseline-only
requirements file will not prepare SASRec. Host-specific artifact delivery and
access must be configured for the chosen demo host. The existing EC2 pipeline
deployment does not install or publish this Streamlit service automatically.

## Decisions needed before publishing

1. Confirm whether the demo will use the existing team AWS account and EC2 host
   or another host, and name the person responsible for deploying the demo.
2. Confirm the team-approved audience and permission to publish the chosen
   product data and example histories. Earlier demo instructions assumed a
   private repository; do not treat repository visibility as permission to
   publish data.
3. Choose how the host receives the ignored artifact files and checkpoint.
4. Deploy, capture the actual URL, and test product selection, real-user loading,
   model selection, ten recommendations, and the pipeline monitor on that host.

The monitor reads saved run records. A successful Airflow or GitHub Actions run
does not by itself deploy the app or automatically refresh its serving data.

## Verification evidence

[Local verification record](project_notes/demo_readiness_20261110/verification.json)
records the tested versions, input/output hashes and test scope.
[Existing-test output](project_notes/demo_readiness_20261110/test_results.txt)
records all 30 passing tests. Python was 3.11.14 and PyTorch 2.8.0 on this local
machine; that differs from the historical Colab training environment, which is
retained in the training records. These are November 10 results; the tests and
benchmark inference were not rerun for the November 25 documentation update.
