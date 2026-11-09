"""Run the seven data-preparation notebooks headlessly and write a run record the monitor page can read.

    python pipeline/run_pipeline.py --notebooks notebooks/data_prep --base pipeline_data \
        --out pipeline_runs/$(date -u +%Y%m%dT%H%M%SZ) [--skip-reviews] [--markdown summary.md]

The notebooks locate their data folder through the RECSYS_BASE environment variable, so nothing in them is edited.
--skip-reviews turns off the multi-GB review download (config.json: include_review_text = false, patched after
notebook 01 has written it). Item text then has no review snippets, so such a run is a pipeline refresh, not a
replacement of the frozen dataset.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

NOTEBOOKS = ['01_data_acquisition', '02_eda_and_subset_selection', '03_cleaning', '04_subset_kcore_reindex',
             '05_split_eval_groups_clients', '06_item_text_preparation', '07_freeze_and_sanity_check']

# (source relative to the data folder, destination relative to the run record)
RECORD_FILES = [
    ('processed/manifest.json', 'manifest.json'),
    ('processed/dataset_info.json', 'dataset_info.json'),
    ('processed/leakage_report.json', 'leakage_report.json'),
    ('processed/DATA_CARD.md', 'DATA_CARD.md'),
    ('config.json', 'data_prep_config.json'),
    ('interim/cleaning_log.json', 'records/cleaning_log.json'),
    ('interim/subset_choice.json', 'records/subset_choice.json'),
    ('interim/index_info.json', 'records/index_info.json'),
    ('processed/eval/eval_protocol.json', 'records/eval_protocol.json'),
    ('processed/clients/client_stats.json', 'records/client_stats.json'),
    ('processed/cold_start/cold_start_info.json', 'records/cold_start_info.json'),
]
EXCLUDED_REPORTS = {'sasrec_smoke_test.json'}      # invalid file (metrics above 1); never part of a record


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def default_executor(notebook, executed, env, timeout):
    subprocess.run([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', str(notebook),
                    '--output', executed.stem, '--output-dir', str(executed.parent),
                    '--ExecutePreprocessor.kernel_name=python3', f'--ExecutePreprocessor.timeout={timeout}'],
                   env=env, check=True)


def patch_config(base, **changes):
    path = Path(base) / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8'))      # raises if notebook 01 did not write it
    config.update(changes)
    path.write_text(json.dumps(config, indent=2), encoding='utf-8')


def collect_record(base, out):
    base, out = Path(base), Path(out)
    missing = []
    for src, dst in RECORD_FILES:
        if not (base / src).exists():
            missing.append(src)
            continue
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / src, out / dst)
    reports = base / 'reports'
    if reports.exists():
        (out / 'reports' / 'figures').mkdir(parents=True, exist_ok=True)
        for f in reports.iterdir():
            if f.is_file() and f.suffix in ('.csv', '.json') and f.name not in EXCLUDED_REPORTS:
                shutil.copy2(f, out / 'reports' / f.name)
        for f in (reports / 'figures').glob('*.png'):
            shutil.copy2(f, out / 'reports' / 'figures' / f.name)
    else:
        missing.append('reports/')
    if missing:
        raise FileNotFoundError('run record incomplete, missing: ' + ', '.join(missing))


def _write(out, status):
    (Path(out) / 'run_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


def status_markdown(status):
    lines = [f"## Data pipeline run: {status['status']}", '',
             f"Started {status['started_utc']}" + (f", finished {status['finished_utc']}" if status.get('finished_utc') else ''), '',
             '| Notebook | Result | Seconds |', '|---|---|---|']
    for n in status['notebooks']:
        lines.append(f"| {n['name']} | {n['status']} | {n['seconds']} |")
    if status.get('note'):
        lines += ['', f"Note: {status['note']}"]
    if status.get('failed_step'):
        lines += ['', f"Failed at: {status['failed_step']}"]
    return '\n'.join(lines) + '\n'


def run_pipeline(notebooks, base, out, skip_reviews=False, timeout=7200, executor=default_executor):
    notebooks, base, out = Path(notebooks), Path(base), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'executed_notebooks').mkdir()
    base.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RECSYS_BASE=str(base.resolve()), MPLBACKEND='Agg')
    status = {'status': 'RUNNING', 'started_utc': _now(), 'finished_utc': None, 'skip_reviews': skip_reviews,
              'python': platform.python_version(), 'pandas': _version('pandas'), 'numpy': _version('numpy'),
              'note': 'review snippets skipped: item text differs from the frozen dataset' if skip_reviews else '',
              'notebooks': []}
    _write(out, status)

    def fail(step, error):
        status.update(status='FAILED', failed_step=step, finished_utc=_now(), error=f'{type(error).__name__}: {error}')
        _write(out, status)
        return status

    for name in NOTEBOOKS:
        begin = time.time()
        try:
            executor(notebooks / f'{name}.ipynb', out / 'executed_notebooks' / f'{name}.ipynb', env, timeout)
        except Exception as error:
            status['notebooks'].append({'name': name, 'status': 'FAILED', 'seconds': round(time.time() - begin)})
            return fail(name, error)
        status['notebooks'].append({'name': name, 'status': 'PASS', 'seconds': round(time.time() - begin)})
        if name == NOTEBOOKS[0] and skip_reviews:
            try:
                patch_config(base, include_review_text=False)
            except Exception as error:
                return fail('patch config.json', error)
        _write(out, status)
    try:
        collect_record(base, out)
    except Exception as error:
        return fail('collect run record', error)
    status.update(status='PASS', finished_utc=_now())
    _write(out, status)
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--notebooks', default='notebooks/data_prep')
    ap.add_argument('--base', default='pipeline_data')
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-reviews', action='store_true')
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--markdown', help='append a summary to this file (for example $GITHUB_STEP_SUMMARY)')
    args = ap.parse_args(argv)
    status = run_pipeline(args.notebooks, args.base, args.out, args.skip_reviews, args.timeout)
    text = status_markdown(status)
    print(text)
    if args.markdown:
        with open(args.markdown, 'a', encoding='utf-8') as f:
            f.write(text)
    return 0 if status['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
