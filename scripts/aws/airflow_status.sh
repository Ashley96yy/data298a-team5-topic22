#!/usr/bin/env bash
# Read-only status for the EC2 Compose deployment. Run with bash, not sudo bash.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/aws/airflow_status.sh [RUN_ID]" >&2
  exit 2
fi

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
compose=(sudo docker compose --env-file .env -f orchestration/compose.yaml)

"${compose[@]}" ps --all
if [[ $# -eq 1 ]]; then
  "${compose[@]}" exec -T airflow-scheduler \
    airflow tasks states-for-dag-run recsys_data_pipeline "$1" --output table
else
  "${compose[@]}" exec -T airflow-scheduler \
    airflow dags list-runs recsys_data_pipeline --output table
fi
