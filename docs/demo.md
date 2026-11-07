# Demo: live recommendations, pipeline monitor and automated pipeline

Requirement (confirmed by the professor): a deployed prototype with an interface where someone picks an input live and the
model returns real recommendations (not a notebook, not hard-coded), and a data pipeline that runs automatically in the cloud.

## What exists
| Piece | Where | What it does |
|---|---|---|
| Recommender page | `app/Recommendations.py`, `src/serving/recommend.py` | Search products, build a history (or load a real user's history), get live top-k from the full 25,903-item catalog. Popularity and last-item Markov today; SASRec and Model 1 plug in with `Recommender.register(name, fn)`. |
| Pipeline monitor page | `app/pages/1_Pipeline_monitor.py`, `src/serving/monitor.py` | Shows the newest pipeline run: stage volumes, cleaning rules and rows removed, 5-core iterations, leakage and quality checks, notebook timings. Every value is read from run files. |
| Pipeline runner | `pipeline/run_pipeline.py` | Runs notebooks 01 to 07 headlessly (they find their data folder through `RECSYS_BASE`, so no notebook is edited), writes a run record and `run_status.json`, stops at the first failure and records it. |
| Cloud workflow | `.github/workflows/pipeline.yml` | Runs the runner on GitHub Actions (manual trigger now; weekly schedule is commented out) and keeps the run record as an artifact. |

## Flow
```
GitHub Actions (manual or weekly) -> pipeline/run_pipeline.py -> notebooks 01-07 -> run record + run_status.json
      -> artifact -> copy into pipeline_runs/<UTC timestamp>/ in the repo -> monitor page shows the newest run
processed data -> python -m src.serving.recommend -> app/artifacts (about 3 MB) -> recommender page
```

## Run locally (repository root)
```sh
pip install -r app/requirements.txt
python -m src.serving.recommend --processed data/processed --out app/artifacts
streamlit run app/Recommendations.py
```

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
