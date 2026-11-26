# Airflow on the Team 5 EC2 instance

Related work: DAT-89 (AWS deployment), DAT-91 (artifact validation).
Use the existing [Airflow stack](../../orchestration/README.md). This runbook
records its EC2 deployment; it does not change the DAG or frozen experiment.

## Environment

- Ubuntu 24.04, m7i-flex.large (2 vCPUs, about 7.6 GiB usable memory), us-east-2.
- Session Manager shell: `ssm-user`.
- Checkout: `$HOME/data298-deployment/repo`.
- Tested code: `3e20602124bd3152b4b91ae65268a944971f4152` (PR #27 merge).
- Docker Engine 29.10.1, Compose 5.8.0; image `topic22-airflow:local` built
  from the repository's Airflow 3.1.8 / Python 3.11 Dockerfile.
- IAM instance role: `data298-team5-ec2-role`; use its temporary credentials.
  Do not put AWS keys or the local `.env` in Git.

Docker was installed using the [official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).
The previously installed warehouse virtual environment is separate from the
notebook environment inside the Airflow image.

## Prepare a fresh deployment

These are setup commands, not commands to rerun while a pipeline is active.
The checkout must contain the tested Airflow files and Docker Compose must be
installed. In the EC2 Session Manager terminal:

```bash
cd "$HOME/data298-deployment/repo"
mkdir -p pipeline_data/downloads pipeline_runs
if [ ! -e .env ]; then
  (umask 077; printf 'AIRFLOW_UID=%s\nAIRFLOW_JWT_SECRET=%s\nRECSYS_AIRFLOW_CRON=@once\n' \
    "$(id -u)" "$(openssl rand -hex 32)" > .env)
fi
```

For an existing `.env`, preserve the UID and JWT secret and set exactly one
`RECSYS_AIRFLOW_CRON=@once` entry. Confirm these three settings locally without
sharing the secret. The repository defaults to daily scheduling; this EC2
validation used `@once` to avoid recurring runs. Changing to `@once` on a DAG
that already ran is not a way to request another validation run.

```bash
sudo docker compose --env-file .env -f orchestration/compose.yaml build
sudo docker compose --env-file .env -f orchestration/compose.yaml up -d --no-build
curl -fsS http://127.0.0.1:8080/api/v2/monitor/health | python3 -m json.tool
sudo docker compose --env-file .env -f orchestration/compose.yaml \
  exec -T airflow-scheduler airflow dags list-import-errors --output json
sudo docker compose --env-file .env -f orchestration/compose.yaml \
  exec -T airflow-scheduler airflow dags details recsys_data_pipeline --output json
```

Confirm healthy metadatabase, scheduler and DAG processor, no import errors,
and timetable `@once`. No triggerer service is configured; its health fields
are null. The init container exiting with code 0 is expected.

For the first scheduled run, unpause once:

```bash
sudo docker compose --env-file .env -f orchestration/compose.yaml \
  exec -T airflow-scheduler airflow dags unpause recsys_data_pipeline --yes
```

The default `skip_reviews=True` remains enabled. This refresh uses ratings
and metadata; it is not the approved frozen dataset with review snippets.
Do not overwrite `frozen/amazon2023-skincare-v1/` or point the frozen warehouse
loader at these new files.

## Check progress without a browser

The helper only reads container and Airflow status; it does not trigger,
pause, clear or stop anything. It works from any working directory when
invoked using its full path. Run it as the normal Session Manager user:

```bash
bash scripts/aws/airflow_status.sh
bash scripts/aws/airflow_status.sh 'scheduled__2025-01-01T00:00:00+00:00'
```

The helper was merged in PR #28. Until that commit is pulled onto EC2, the
equivalent is:

```bash
sudo docker compose --env-file .env -f orchestration/compose.yaml \
  exec -T airflow-scheduler airflow tasks states-for-dag-run recsys_data_pipeline \
  'scheduled__2025-01-01T00:00:00+00:00' --output table
```

The 2025 logical date comes from the DAG start date, not the actual execution
date. `None` task states mean tasks have not started; use the state table to
distinguish waiting, running, success and failure. Service health alone does
not establish that a pipeline run succeeded.

Airflow binds to the EC2 host's `127.0.0.1:8080`. A Mac browser using that
address reaches the Mac, so it cannot reach this deployment without a tunnel.
Browser access is optional; no public port or security-group change is needed
for the terminal workflow.

## Observed validation, November 24, 2026 (Pacific time)

Evidence below is from the operator's Session Manager output, not a new local
test or independent AWS query. UTC execution was November 25, 03:24–03:33.

| Task | Observed state |
| --- | --- |
| 01_data_acquisition | success |
| 02_eda_and_subset_selection | success |
| 03_cleaning | success |
| 04_subset_kcore_reindex | success |
| 05_split_eval_groups_clients | success |
| 06_item_text_preparation | success |
| 07_freeze_and_sanity_check | success |
| quality_gate | success |
| publish_run_record | success |

Run ID: `scheduled__2025-01-01T00:00:00+00:00`.
First task started `2026-11-25T03:24:46.534061+00:00`; publication finished
`2026-11-25T03:33:19.371762+00:00` (about 8 minutes 33 seconds).
All nine task successes were shown in the operator's 03:34 UTC status check.

## Preserve outputs and stop

The DAG publishes locally; it does not upload to S3. Published records are
under `pipeline_runs/<run-key>/run_status.json`; its `processed_dir` identifies
the refresh tables under `pipeline_data/runs/<run-key>/processed/`. Quality
evidence is under that run's `reports/` directory.

Before stopping, confirm the published record reports `PASS`, archive the
record, processed tables and reports to a new run-specific prefix under
`s3://data298-team5-artifacts-2026/refresh-runs/`, and download the archive
again to compare its SHA-256 with the local archive. Do not include `.env`,
credentials, Docker volumes or the shared raw-download cache. An archive
upload and matching readback for this Airflow run have not yet been recorded.

After saving and verifying the outputs, pause the DAG and stop the stack:

```bash
sudo docker compose --env-file .env -f orchestration/compose.yaml \
  exec -T airflow-scheduler airflow dags pause recsys_data_pipeline --yes
sudo docker compose --env-file .env -f orchestration/compose.yaml down
```

Do not run `down -v`: the metadata and logs volumes must be preserved.
Pausing prevents new scheduling; it does not stop an active run. Wait for the
run to finish before shutdown. Finally use EC2 **Instance state > Stop
instance** and confirm **Stopped**. Closing Session Manager or stopping
containers alone does not stop EC2 compute usage. EBS and S3 storage remain
billable.

Status update, November 25, 2026 (Pacific time): the operator's DAG listing
confirmed `is_paused=True` after all nine tasks succeeded, and the operator
subsequently confirmed that EC2 was stopped. This is operator-reported status,
not a fresh AWS query. The Airflow refresh archive and verified S3 readback
remain pending; complete them during the next planned EC2 session. The earlier
verified `warehouse/aws_run_001/` upload contains separate frozen-warehouse
outputs and does not establish a backup of this Airflow refresh.

References: [Airflow scheduling](https://airflow.apache.org/docs/apache-airflow/3.1.8/authoring-and-scheduling/cron.html),
[Airflow CLI](https://airflow.apache.org/docs/apache-airflow/3.1.8/cli-and-env-variables-ref.html),
[EC2 stop/start](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html).
