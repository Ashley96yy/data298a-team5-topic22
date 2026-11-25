FROM apache/airflow:3.1.8-python3.11

USER airflow
COPY --chown=airflow:0 pipeline/requirements.txt /tmp/pipeline-requirements.txt
RUN python -m venv /home/airflow/pipeline-venv \
 && /home/airflow/pipeline-venv/bin/python -m pip install --no-cache-dir -r /tmp/pipeline-requirements.txt \
 && /home/airflow/pipeline-venv/bin/python -m ipykernel install --prefix /home/airflow/pipeline-venv --name python3

WORKDIR /opt/recsys
COPY --chown=airflow:0 pipeline/ pipeline/
COPY --chown=airflow:0 notebooks/data_prep/ notebooks/data_prep/
COPY --chown=airflow:0 src/serving/monitor.py src/serving/monitor.py
COPY --chown=airflow:0 tests/test_pipeline*.py tests/test_airflow_pipeline.py tests/
COPY --chown=airflow:0 orchestration/dags/ /opt/airflow/dags/
ENV PYTHONPATH=/opt/recsys
