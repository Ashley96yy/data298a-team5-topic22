# Run the data pipeline in Airflow

This Compose stack runs the seven existing data-preparation notebooks as ordered Airflow tasks. PostgreSQL holds Airflow metadata. The notebook dependencies live in a separate environment in the image. Each task retries twice at one-minute intervals. The default schedule is daily at 09:00 UTC, with no backfill and at most one active run.

## Start from the repository root

Allocate at least 8 GB of memory to Docker and keep free disk space for the download cache and each run's outputs. Run one pipeline stack at a time on a laptop. If Docker kills a notebook kernel during EDA, free memory from other workloads or raise Docker's memory limit before clearing the failed task.

To set up a new checkout:

```sh
mkdir -p pipeline_data/downloads pipeline_runs
if [ ! -e .env ]; then
  (umask 077; printf 'AIRFLOW_UID=%s\nAIRFLOW_JWT_SECRET=%s\n' "$(id -u)" "$(openssl rand -hex 32)" > .env)
fi
docker compose --env-file .env -f orchestration/compose.yaml up --build -d
docker compose --env-file .env -f orchestration/compose.yaml logs airflow-api
```

Add the two variables above to an existing `.env` if needed. You can use `AIRFLOW_PORT` in that file if port 8080 is already occupied. Open <http://localhost:8080> and sign in as `admin` with the password printed in the API logs. The UI binds to localhost. Unpause `recsys_data_pipeline` to enable its schedule. You can also trigger it from the UI. To check scheduling without waiting a day, temporarily set `RECSYS_AIRFLOW_CRON="*/15 * * * *"` in `.env` and recreate the services with the same Compose command.

The default run skips review snippets. The notebooks still download official 5-core ratings and product metadata on the first run. Later runs reuse the archives in `pipeline_data/downloads/`. Set the Airflow run parameter `skip_reviews` to `false` only when you need review snippets and have the extra time and disk space. These are refresh runs. Model evaluation still uses the separately versioned frozen dataset.

## Quality, alerts and recovery

- Notebook 05 runs the twelve leakage checks. The quality gate checks that all twelve passed, verifies generated file hashes and sizes, and checks columns, nulls, ratings, IDs, duplicates, chronology, sequence lengths and federated client assignments.
- The monitor record appears only after the gate passes. A failed run remains visible in Airflow. Look at the failed task logs and the partially written files in its work directory. When the cause is fixed, clear the failed task and its downstream tasks, or trigger a new run. Trigger a new run if the original run was already published.
- A final task failure logs the DAG, task and run ID after retries are exhausted. To receive a Slack message, set `RECSYS_ALERT_WEBHOOK_URL` in the ignored `.env` file and recreate the services. Without the webhook, the alert stays in the Airflow logs. The notification includes no data rows, notebook output or exception text.
- Use `docker compose --env-file .env -f orchestration/compose.yaml down` to stop the stack. The database volume and bind-mounted data survive a normal `down`.

## Paths for the loader and monitor

| Purpose | Path relative to the checkout |
| --- | --- |
| Download cache | `pipeline_data/downloads/` |
| Versioned refresh tables and manifest | `pipeline_data/runs/<run>/processed/` |
| Published Airflow run records | `pipeline_runs/<run>/run_status.json` |
| Frozen experiment run documentation | `docs/data_run/` |

For a refresh, use the `processed_dir` field of a published `PASS` record to locate its tables. The existing Streamlit Pipeline monitor reads `pipeline_runs/`; it can use the same folder without any app changes. For the frozen experiment, confirm the approved dataset version and storage path with the warehouse owner. Keep the production loader pointed at that frozen snapshot unless the team elects to switch versions.

The Airflow CI workflow builds this image, checks the DAG and runs small notebook and quality-gate fixtures. It does not download the Amazon data.
