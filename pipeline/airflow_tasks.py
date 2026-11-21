"""Run the existing notebooks in isolated Airflow workspaces and publish validated records."""
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

from pipeline.run_pipeline import NOTEBOOKS, _now, collect_record, default_executor, patch_config

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / 'pipeline_data'
RECORD_ROOT = REPO / 'pipeline_runs'
NOTEBOOK_ROOT = REPO / 'notebooks' / 'data_prep'
NOTEBOOK_PYTHON = '/home/airflow/pipeline-venv/bin/python'
LOGGER = logging.getLogger(__name__)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run_key(run_id, run_after):
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return f'{run_after:%Y%m%dT%H%M%SZ}-{digest}'


def run_notebook(name, work, skip_reviews):
    """Reuse the notebook process and keep each run's intermediate data separate."""
    if (RECORD_ROOT / work.name).exists():
        raise ValueError('This run is already published. Trigger a new run to refresh the data.')
    work.mkdir(parents=True, exist_ok=True)
    executed = work / 'executed_notebooks'
    executed.mkdir(exist_ok=True)
    downloads = DATA_ROOT / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RECSYS_BASE=str(work), RECSYS_DOWNLOAD_DIR=str(downloads),
               MPLBACKEND='Agg')
    started = _now()
    begin = time.monotonic()
    default_executor(NOTEBOOK_ROOT / f'{name}.ipynb', executed / f'{name}.ipynb',
                     env, 7200, python_executable=NOTEBOOK_PYTHON)
    if name == NOTEBOOKS[0] and skip_reviews:
        patch_config(work, include_review_text=False)
    result = {'name': name, 'status': 'PASS', 'seconds': round(time.monotonic() - begin),
              'started_utc': started}
    (executed / f'{name}.json').write_text(json.dumps(result), encoding='utf-8')


def check_quality(work):
    """Require the data and leakage reports to pass and verify the generated manifest."""
    processed = (work / 'processed').resolve()
    leakage = _read(processed / 'leakage_report.json')
    checks = leakage.get('checks')
    if (leakage.get('all_passed') is not True or not isinstance(checks, dict)
            or len(checks) != 12 or any(v is not True for v in checks.values())):
        raise ValueError('The leakage report did not pass all twelve checks.')
    files = _read(processed / 'manifest.json').get('processed_files')
    required = {'leakage_report.json', 'dataset_info.json', 'interactions.parquet',
                'sequences.parquet', 'clients/clients.parquet'}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError('The manifest is missing files required by the quality gate.')
    for name, entry in files.items():
        path = (processed / name).resolve()
        if not path.is_relative_to(processed) or not path.is_file():
            raise ValueError(f'Manifest path is missing or invalid: {name}')
        with path.open('rb') as data:
            digest = hashlib.file_digest(data, 'sha256').hexdigest()
        if path.stat().st_size != entry['bytes'] or digest != entry['sha256']:
            raise ValueError(f'Manifest integrity check failed: {name}')


    import pandas as pd

    info = _read(processed / 'dataset_info.json')
    inter = pd.read_parquet(processed / 'interactions.parquet')
    seq = pd.read_parquet(processed / 'sequences.parquet')
    clients = pd.read_parquet(processed / 'clients' / 'clients.parquet')
    fields = (
        (inter, ['user_idx', 'item_idx', 'rating', 'timestamp', 'pos', 'split', 'row_id']),
        (seq, ['user_idx', 'train_items', 'train_ts', 'n_train', 'val_item', 'val_ts',
               'test_item', 'test_ts']),
        (clients, ['client_id', 'n_local_train']),
    )
    for table, cols in fields:
        if not set(cols).issubset(table.columns):
            raise ValueError(f'Missing data quality columns: {sorted(set(cols) - set(table.columns))}')
    users = info['n_users']
    items = info['n_items']
    counts = inter.groupby(['user_idx', 'split']).size().unstack(fill_value=0)
    indexed_seq = seq.set_index('user_idx').sort_index()
    checks = {
        'columns': True,
        'nulls': all(not table[cols].isna().any().any() for table, cols in fields),
        'ratings': inter.rating.isin([1, 2, 3, 4, 5]).all(),
        'IDs': (len(seq) == users and seq.user_idx.is_unique
                and inter.user_idx.between(0, users - 1).all() and inter.user_idx.nunique() == users
                and inter.item_idx.between(1, items).all() and inter.item_idx.nunique() == items
                and len(inter) == info['n_interactions']),
        'duplicates': not inter.duplicated(['user_idx', 'item_idx']).any() and inter.row_id.is_unique,
        'chronology': all(len(t) > 0 and all(a <= b for a, b in zip(t, t[1:])) and t[-1] <= v <= x
                          for t, v, x in zip(seq.train_ts, seq.val_ts, seq.test_ts)),
        'sequence lengths': (set(counts.columns) == {'train', 'val', 'test'}
                             and counts.index.equals(indexed_seq.index)
                             and (counts['train'] == indexed_seq.n_train).all()
                             and (counts[['val', 'test']] == 1).all().all()
                             and seq.n_train.ge(info['kcore'] - 2).all()
                             and seq.train_items.str.len().eq(seq.n_train).all()
                             and seq.train_ts.str.len().eq(seq.n_train).all()),
        'federated clients': (clients.client_id.is_unique
                              and set(clients.client_id) == set(seq.user_idx)
                              and clients.n_local_train.eq(clients.client_id.map(indexed_seq.n_train)).all()),
    }
    failures = [name for name, passed in checks.items() if not passed]
    status = {'status': 'FAIL' if failures else 'PASS', 'users': users, 'items': items,
              'interactions': len(inter), 'checks': list(checks)}
    if failures:
        status['failed_checks'] = failures
    (work / 'reports' / 'quality_gate.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    if failures:
        raise ValueError(f'Data quality checks failed: {", ".join(failures)}')


def publish_record(work, run, skip_reviews):
    """Publish atomically so the monitor only picks up complete, validated records."""
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    target = RECORD_ROOT / run
    if target.exists():
        if _read(target / 'run_status.json').get('status') != 'PASS':
            raise ValueError(f'Unexpected existing publication at {target}')
        return
    staged = RECORD_ROOT / '.staging' / run
    staged.mkdir(parents=True, exist_ok=True)
    collect_record(work, staged)
    executed = work / 'executed_notebooks'
    notebooks = [_read(executed / f'{name}.json') for name in NOTEBOOKS]
    status = {
        'status': 'PASS',
        'started_utc': notebooks[0]['started_utc'],
        'finished_utc': _now(),
        'skip_reviews': skip_reviews,
        'note': ('Review snippets skipped; this refresh differs from the frozen experiment.'
                 if skip_reviews else 'Local refresh; frozen experiment files are kept separately.'),
        'notebooks': notebooks,
        'processed_dir': str(work.relative_to(REPO) / 'processed'),
    }
    (staged / 'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    staged.rename(target)


def notify_failure(context):
    """Log an alert and optionally deliver the same summary to a Slack webhook."""
    task = context['task_instance']
    message = f'Airflow task failed after retries: {task.dag_id}/{task.task_id} ({task.run_id})'
    LOGGER.error('%s', message)
    webhook = os.environ.get('RECSYS_ALERT_WEBHOOK_URL')
    if not webhook:
        return
    try:
        request = Request(webhook, data=json.dumps({'text': message}).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=10):
            LOGGER.info('Delivered pipeline failure alert.')
    except (OSError, ValueError) as error:
        LOGGER.error('Could not deliver pipeline failure alert (%s).', type(error).__name__)


if __name__ == '__main__':
    check_quality(Path(sys.argv[1]))
