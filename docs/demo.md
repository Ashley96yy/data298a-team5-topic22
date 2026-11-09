# Demo: live recommendations, pipeline monitor and automated pipeline

Requirement (confirmed by the professor): a deployed prototype with an interface where someone picks an input live and the
model returns real recommendations (not a notebook, not hard-coded), and a data pipeline that runs automatically in the cloud.

## What exists
| Piece | Where | What it does |
|---|---|---|
| Recommender page | `app/Recommendations.py`, `src/serving/recommend.py` | Search products, build a history (or load a real user's history), get live top-k from the full 25,903-item catalog. Popularity and last-item Markov work with the baseline dependencies; the optional SASRec adapter registers a verified checkpoint. Model 1 remains a future adapter. |
| Pipeline monitor page | `app/pages/1_Pipeline_monitor.py`, `src/serving/monitor.py` | Shows the newest pipeline run: stage volumes, cleaning rules and rows removed, 5-core iterations, leakage and quality checks, notebook timings. Every value is read from run files. |
| Pipeline runner | `pipeline/run_pipeline.py` | Runs notebooks 01 to 07 headlessly (they find their data folder through `RECSYS_BASE`, so no notebook is edited), writes a run record and `run_status.json`, stops at the first failure and records it. |
| Cloud workflow | `.github/workflows/pipeline.yml` | Manual GitHub Actions trigger; review snippets skipped by default. Keeps the run record and execution log for seven days, including failures. No schedule is enabled. |

## Flow
```
GitHub Actions (manual trigger) -> pipeline/run_pipeline.py -> notebooks 01-07 -> run record + run_status.json
      -> artifact -> copy into pipeline_runs/<UTC timestamp>/ in the repo -> monitor page shows the newest run
processed data -> python -m src.serving.recommend -> app/artifacts (about 3 MB) -> recommender page
```

## Run locally (repository root)
```sh
pip install -r app/requirements.txt
python -m src.serving.recommend --processed data/processed --out app/artifacts
streamlit run app/Recommendations.py
```

## Add the trained SASRec model

Build artifacts from the **frozen** processed folder containing `manifest.json`,
`items.parquet` and `sequences.parquet`. The builder verifies the two consumed
data files against that manifest and writes `artifact_manifest.json` with source
and output hashes. A baseline-only build without a frozen manifest is still
supported, but cannot be paired with a trained SASRec checkpoint.

```sh
pip install -r app/requirements-sasrec.txt
python -m src.serving.recommend --processed data/processed --out app/artifacts
export SASREC_CHECKPOINT=/absolute/local/path/to/best.pt
export SASREC_CHECKPOINT_SHA256=<independently-verified-sha256>
streamlit run app/Recommendations.py
```

Restart the app after configuring these environment variables. With both set,
`load_recommender()` calls `register_sasrec()` from `src/serving/sasrec.py`.
It checks checkpoint bytes, artifact hashes and matching frozen-data provenance,
loads the architecture recorded in the checkpoint, and registers `sasrec`.
Wrong or incomplete configuration fails visibly rather than showing another
model under the SASRec label. Checkpoints must be trusted project outputs.

The adapter retains the last `max_seq_len` items and left-pads with zero exactly
as the model expects. `Recommender.recommend()` excludes padding and the complete
user-supplied history. An empty history still uses popularity in the interface.
Inference is on CPU; this is a live demo, not a benchmark evaluation.

The reviewed Model 0 seed-42 checkpoint is identified in PR #9's handoff record
by SHA-256 `54ca6cba54f956af7b6189c5c0b98cf30d55e6412f412f7b92b184d89cbd89e9`.
Download the original file from its shared run folder; do not commit weights,
generated app artifacts or user histories to this branch. Baseline deployments
keep using `app/requirements.txt`, without importing PyTorch.

## First GitHub Actions run

Both the demo runner and `.github/workflows/pipeline.yml` must be merged into the
default branch first. GitHub requires a default-branch `workflow_dispatch` file
before a manual run can be triggered:
https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow

Then open **Actions -> data-pipeline -> Run workflow**, select the reviewed
branch and leave **skip_reviews** checked for the initial run. The workflow
installs `pipeline/requirements.txt` on Python 3.11.13, executes notebooks 01-07,
and appends the result to the Actions summary. Inspect the log and download the
`data-pipeline-<run_id>-<attempt>` artifact, including `run_status.json` and all
seven notebook outcomes. A failed notebook keeps a FAILED record; kernel or
dependency failures must not be called a successful pipeline run.

To show the cloud run in the monitor, copy the artifact's timestamped run folder
under `pipeline_runs/` locally or in the deployment. The monitor detects that
layout directly; this copy is still manual. The workflow does not overwrite
the frozen training dataset or publish new serving artifacts.

## Deploy
Hosted app: connect the private repository to a free host such as Streamlit Community Cloud, main file
`app/Recommendations.py`, Python 3.11, requirements `app/requirements.txt` (check how the host chooses between the root and
`app/` requirements files; the root one pulls in torch). Commit `app/artifacts/` for deployment (about 3 MB).

## Tested, and not tested
Tested: recommender logic and both pages with Streamlit's headless test runner; the pipeline runner with stand-in notebooks
(success, failure, skipped reviews, incomplete record); the real `jupyter nbconvert` command on a toy notebook with `RECSYS_BASE`.
Not tested: running the seven real notebooks on a GitHub runner (do one manual run and read the log), the hosted deployment,
and how the pages look in a browser.

## Known limits and decisions
- Runner limits: memory, disk and the free minutes of a private repository may not suffice for the full download. `--skip-reviews`
  (default in the workflow) avoids the multi-GB review file, but then item text has no review snippets, so that run refreshes the
  pipeline and does not replace the frozen dataset.
- The cloud run does not feed the recommender page: its artifacts come from the frozen processed data.
- Getting a cloud run into the app is a manual copy of the artifact into `pipeline_runs/` for now; a bot commit can be added later.
- Confirm with the professor what "automatically" must mean (scheduled, or triggered on demand).
- Check the Amazon Reviews 2023 license before committing item titles, and keep the repository private.
