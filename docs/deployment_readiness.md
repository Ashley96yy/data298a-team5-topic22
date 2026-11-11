# Demo deployment preparation

Written by: Abhinita Sanabada

The current demo runs locally with popularity, Markov and the original Model 0
seed-42 checkpoint. On November 10, 2026, all **30 existing serving and UI tests
passed with zero skips**, including the real-checkpoint Streamlit AppTest.
This verifies local behavior; no hosted deployment or browser-session acceptance
is claimed. The frozen artifacts and checkpoint were downloaded and checked,
not retrained or re-evaluated as a benchmark.

## Prepared local files

The project folder is `/Users/tachyon_hunter/Desktop/sem-4/298Aproject/298A`.

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
credentials must be configured after the host/account is selected.

## Decisions needed before publishing

1. Confirm which team account owns the host and who can deploy.
2. Confirm the team-approved audience and permission to publish the chosen
   product data and example histories. Earlier demo instructions assumed a
   private repository; do not treat repository visibility as permission to
   publish data.
3. Choose how the host receives the ignored artifact files and checkpoint.
4. Deploy, capture the actual URL, and test product selection, real-user loading,
   model selection, ten recommendations, and the pipeline monitor on that host.

The monitor reads saved run records. A successful GitHub Actions pipeline run
does not by itself deploy the app or automatically refresh its serving data.

## Verification evidence

[Local verification record](project_notes/demo_readiness_20261110/verification.json)
records the tested versions, input/output hashes and test scope.
[Existing-test output](project_notes/demo_readiness_20261110/test_results.txt)
records all 30 passing tests. Python was 3.11.14 and PyTorch 2.8.0 on this local
machine; that differs from the historical Colab training environment, which is
retained in the training records.
