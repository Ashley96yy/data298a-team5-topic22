"""Airflow orchestration for the project's original data-preparation notebooks."""
from datetime import datetime, timedelta, timezone
import os
import subprocess

from airflow.sdk import Param, dag, get_current_context, task

from pipeline.airflow_tasks import (DATA_ROOT, NOTEBOOK_PYTHON, notify_failure, publish_record,
                                    run_key, run_notebook)
from pipeline.run_pipeline import NOTEBOOKS


def current_run():
    context = get_current_context()
    run = context['dag_run']
    key = run_key(run.run_id, run.run_after.astimezone(timezone.utc))
    return DATA_ROOT / 'runs' / key, key, context['params']['skip_reviews']


@dag(
    dag_id='recsys_data_pipeline',
    schedule=os.getenv('RECSYS_AIRFLOW_CRON', '0 9 * * *'),
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={
        'retries': 2,
        'retry_delay': timedelta(minutes=1),
        'on_failure_callback': notify_failure,
    },
    params={'skip_reviews': Param(True, type='boolean', description='Skip the full review-text download.')},
    tags=['topic22', 'data-prep'],
)
def data_pipeline():
    @task
    def execute_notebook(name):
        work, _, skip_reviews = current_run()
        run_notebook(name, work, skip_reviews)

    @task(task_id='quality_gate')
    def validate():
        work, _, _ = current_run()
        subprocess.run([NOTEBOOK_PYTHON, '-m', 'pipeline.airflow_tasks', str(work)], check=True)

    @task(task_id='publish_run_record')
    def publish():
        work, run, skip_reviews = current_run()
        publish_record(work, run, skip_reviews)

    steps = [execute_notebook.override(task_id=name)(name) for name in NOTEBOOKS]
    for upstream, downstream in zip(steps, steps[1:]):
        upstream >> downstream
    steps[-1] >> validate() >> publish()


data_pipeline()
