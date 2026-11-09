"""Read the record of a data-pipeline run (manual or automated) for the monitor page.

A run directory has the same layout either way:
  manifest.json, dataset_info.json, leakage_report.json, records/cleaning_log.json,
  reports/{quality_gate.json, kcore_iterations.csv, dataset_statistics.csv}, optional run_status.json.
Automated runs (pipeline/run_pipeline.py) live in pipeline_runs/<UTC timestamp>/ and always have run_status.json.
The hand-run record committed from the first execution lives in docs/data_run/.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def _csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def format_utc(value):
    """'2026-10-18T03:19:49.529511+00:00' -> '2026-10-18 03:19 UTC'; anything unparseable is returned unchanged."""
    if not value:
        return value
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return value
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M UTC')


def find_run_dir(root):
    """Newest automated run (any status, so failures are visible), else the committed manual record, else None."""
    root = Path(root)
    runs = sorted(p.parent for p in (root / 'pipeline_runs').glob('*/run_status.json')) if (root / 'pipeline_runs').exists() else []
    if runs:
        return runs[-1]                                   # directory names are UTC timestamps: last = newest
    manual = root / 'docs' / 'data_run'
    return manual if (manual / 'manifest.json').exists() else None


def load_run(run_dir):
    d = Path(run_dir)
    status = _json(d / 'run_status.json')
    return {
        'path': d,
        'source': 'automated run' if status else 'manual run record',
        'status_file': status,
        'manifest': _json(d / 'manifest.json'),
        'dataset_info': _json(d / 'dataset_info.json'),
        'leakage': _json(d / 'leakage_report.json'),
        'quality': _json(d / 'reports' / 'quality_gate.json'),
        'cleaning': _json(d / 'records' / 'cleaning_log.json'),
        'kcore': _csv(d / 'reports' / 'kcore_iterations.csv'),
        'stats': _csv(d / 'reports' / 'dataset_statistics.csv'),
    }


def summary(run):
    """Headline numbers for the page; every value comes from a run file, missing files give None."""
    status = run['status_file']
    manifest = run['manifest']
    leak = run['leakage'] or {}
    checks = leak.get('checks', {})
    stats = run['stats']
    final = stats.iloc[-1] if stats is not None and len(stats) else None
    if status:
        overall = status.get('status')
        last = status.get('finished_utc') or status.get('started_utc')
    else:
        overall = (run['quality'] or {}).get('status')
        last = manifest.get('created_utc') if manifest else None
    return {
        'source': run['source'],
        'status': overall,
        'last_run_utc': format_utc(last),
        'leakage_passed': sum(bool(v) for v in checks.values()),
        'leakage_total': len(checks),
        'quality_status': (run['quality'] or {}).get('status'),
        'final_users': int(final['users']) if final is not None else None,
        'final_items': int(final['items']) if final is not None else None,
        'final_interactions': int(final['interactions']) if final is not None else None,
        'versions': (manifest or {}).get('versions'),
    }


def stage_table(run):
    stats = run['stats']
    if stats is None:
        return pd.DataFrame(columns=['stage', 'users', 'items', 'interactions', 'avg_seq_len'])
    return stats[['stage', 'users', 'items', 'interactions', 'avg_seq_len']].copy()


def cleaning_table(run):
    steps = (run['cleaning'] or {}).get('steps', [])
    return pd.DataFrame([{'rule': s['step'], 'rows_before': s['rows_before'], 'rows_after': s['rows_after'],
                          'removed': s['removed']} for s in steps],
                        columns=['rule', 'rows_before', 'rows_after', 'removed'])


def checks_table(run):
    rows = [{'group': 'Leakage', 'check': name, 'result': 'PASS' if ok else 'FAIL'}
            for name, ok in ((run['leakage'] or {}).get('checks', {})).items()]
    quality = run['quality'] or {}
    rows += [{'group': 'Quality gate', 'check': c, 'result': quality.get('status', 'UNKNOWN')}
             for c in quality.get('checks', [])]
    return pd.DataFrame(rows, columns=['group', 'check', 'result'])


def timing_table(run):
    status = run['status_file'] or {}
    return pd.DataFrame(status.get('notebooks', []))
