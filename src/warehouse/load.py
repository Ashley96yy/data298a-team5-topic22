"""Build a new local DuckDB snapshot; fail closed on manifest or SQL checks."""
import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

TABLES = {
    'interactions': 'interactions.parquet', 'user_map': 'user_map.parquet',
    'item_map': 'item_map.parquet', 'items': 'items.parquet',
    'item_text': 'item_text.parquet', 'sequences': 'sequences.parquet',
    'train_reviews': 'train_reviews.parquet',
    'interaction_side_info': 'interaction_side_info.parquet',
    'eval_cases': 'eval/eval_cases.parquet',
    'item_groups': 'groups/item_groups.parquet',
    'user_groups': 'groups/user_groups.parquet',
    'clients': 'clients/clients.parquet',
    'cold_sequences': 'cold_start/cold_sequences.parquet',
}
REQUIRED = set(TABLES.values()) | {
    'dataset_info.json', 'rec_eval.py', 'leakage_report.json',
    'eval/eval_protocol.json', 'eval/neg_val.npy', 'eval/neg_test.npy',
    'clients/client_stats.json', 'cold_start/cold_start_info.json',
    'cold_start/cold_items.npy',
}
PROJECT = Path(__file__).resolve().parents[2]
SQL_PATH = Path(__file__).with_name('quality_checks.sql')


class VerificationError(ValueError):
    """Frozen inputs do not match their trusted record."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_inputs(processed, trusted_manifest):
    """Verify every manifest entry, including evaluator/NumPy files not loaded as tables."""
    root = Path(processed).resolve()
    trusted_manifest = Path(trusted_manifest)
    trusted_hash = digest(trusted_manifest)
    if digest(root / 'manifest.json') != trusted_hash:
        raise VerificationError('Local manifest differs from the trusted manifest')
    manifest = json.loads(trusted_manifest.read_text())
    entries = manifest['processed_files']
    if not REQUIRED.issubset(entries):
        raise VerificationError('Manifest is missing required frozen artifacts')
    verified = {}
    errors = []
    for name, expected in sorted(entries.items()):
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            errors.append(f'Unsafe manifest path: {name}')
        elif not path.is_file():
            errors.append(f'Missing file: {name}')
        elif path.stat().st_size != expected['bytes']:
            errors.append(f'Size mismatch: {name}')
        elif digest(path) != expected['sha256']:
            errors.append(f'SHA-256 mismatch: {name}')
        else:
            verified[name] = {'bytes': expected['bytes'], 'sha256': expected['sha256']}
    if errors:
        raise VerificationError('; '.join(errors))
    info = json.loads((root / 'dataset_info.json').read_text())
    if info != manifest['dataset_info']:
        raise VerificationError('Dataset statistics differ from the trusted manifest')
    return manifest, trusted_hash, verified


def quality_checks(con, info):
    checks = []

    def check(name, sql, expected=0, params=None):
        actual = con.execute(sql, params or []).fetchone()[0]
        checks.append({'name': name, 'actual': actual, 'expected': expected,
                       'passed': actual == expected})

    counts = {'interactions': info['n_interactions'], 'user_map': info['n_users'],
              'item_map': info['n_items'], 'items': info['n_items'],
              'item_text': info['n_items'], 'sequences': info['n_users'],
              'clients': info['n_users'], 'user_groups': info['n_users'],
              'item_groups': info['n_items'], 'eval_cases': 2 * info['n_users']}
    for table, expected in counts.items():
        check(f'{table}_row_count', f'SELECT count(*) FROM {table}', expected)
    for table, key, lo, hi in [('user_map', 'user_idx', 0, info['n_users'] - 1),
                               ('item_map', 'item_idx', 1, info['n_items'])]:
        check(f'{table}_id_range', f'SELECT count(*) FROM {table} WHERE {key} IS NULL '
              f'OR {key} NOT BETWEEN ? AND ?', params=[lo, hi])
    for table, key in [('user_map', 'user_idx'), ('user_map', 'user_id'),
                       ('item_map', 'item_idx'), ('item_map', 'parent_asin'),
                       ('sequences', 'user_idx'), ('user_groups', 'user_idx'),
                       ('item_groups', 'item_idx'), ('clients', 'client_id')]:
        check(f'{table}_{key}_unique_not_null', f'SELECT count(*) FROM '
              f'(SELECT {key} FROM {table} GROUP BY {key} HAVING {key} IS NULL OR count(*)<>1)')
    for split in ('train', 'val', 'test'):
        expected = info['n_users'] if split != 'train' else info['n_interactions'] - 2 * info['n_users']
        check(f'{split}_count', 'SELECT count(*) FROM interactions WHERE split=?', expected, [split])
    for key in ('user_idx', 'item_idx'):
        check(f'{key}_kcore', f'SELECT count(*) FROM (SELECT {key} FROM interactions '
              f'GROUP BY {key} HAVING count(*) < ?)', params=[info['kcore']])
    chunks = re.split(r'^-- check: ([a-z_]+)\s*$', SQL_PATH.read_text(), flags=re.MULTILINE)
    for name, sql in zip(chunks[1::2], chunks[2::2]):
        check(name, sql)
    return checks


def code_provenance():
    def git(*args):
        result = subprocess.run(['git', '-C', str(PROJECT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'git_commit': git('rev-parse', 'HEAD'),
            'working_tree_dirty': bool(git('status', '--porcelain')),
            'source_sha256': {p.name: digest(p) for p in [Path(__file__), SQL_PATH]},
            'python': platform.python_version(), 'duckdb': duckdb.__version__,
            'platform': platform.platform()}


def write_report(output, report):
    (output / 'quality_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (output / 'quality_checks.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'actual', 'expected', 'passed'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(report['checks'])


def build_warehouse(processed, trusted_manifest, output):
    """Use a new output directory per run. Never modify an existing warehouse or inputs.

    On failure, preserve an aggregate-only report and remove the unpublished database.
    Consumers must require quality_report.json status PASS before opening warehouse.duckdb.
    """
    processed, output = Path(processed).resolve(), Path(output).resolve()
    if output == processed or output.is_relative_to(processed):
        raise ValueError('Output must be outside the frozen processed directory')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    temporary_db = output / 'building.duckdb'
    final_db = output / 'warehouse.duckdb'
    report = {'status': 'FAIL', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'provenance': code_provenance(), 'checks': [], 'tables': {}}
    con = None
    try:
        manifest, manifest_hash, verified = verify_inputs(processed, trusted_manifest)
        report.update(manifest_sha256=manifest_hash, verified_inputs=verified,
                      dataset_info=manifest['dataset_info'])
        con = duckdb.connect(str(temporary_db), config={'threads': 2})
        for table, relative in TABLES.items():
            con.execute(f'CREATE TABLE {table} AS SELECT * FROM read_parquet(?)', [str(processed / relative)])
            report['tables'][table] = {
                'rows': con.execute(f'SELECT count(*) FROM {table}').fetchone()[0],
                'columns': [{'name': row[0], 'type': row[1]} for row in con.execute(f'DESCRIBE {table}').fetchall()]}
        con.execute('CREATE TABLE source_files (path VARCHAR, sha256 VARCHAR, bytes BIGINT)')
        con.executemany('INSERT INTO source_files VALUES (?, ?, ?)',
                        [(name, rec['sha256'], rec['bytes']) for name, rec in verified.items()])
        con.execute('CREATE TABLE frozen_metadata (path VARCHAR, content JSON)')
        for name in verified:
            if name.endswith('.json'):
                con.execute('INSERT INTO frozen_metadata VALUES (?, ?)', [name, (processed / name).read_text()])
        con.execute("""CREATE VIEW sequence_events AS
            SELECT user_idx, unnest(train_items) item_idx, unnest(train_ts) AS "timestamp",
                   generate_subscripts(train_items, 1)-1 pos, 'train' split FROM sequences
            UNION ALL SELECT user_idx, val_item, val_ts, n_train, 'val' FROM sequences
            UNION ALL SELECT user_idx, test_item, test_ts, n_train+1, 'test' FROM sequences""")
        con.execute("""CREATE VIEW split_summary AS SELECT split, count(*) interactions,
            count(DISTINCT user_idx) users, count(DISTINCT item_idx) items
            FROM interactions GROUP BY split""")
        report['checks'] = quality_checks(con, manifest['dataset_info'])
        report['split_summary'] = [dict(zip(['split', 'interactions', 'users', 'items'], row))
                                   for row in con.execute('SELECT * FROM split_summary ORDER BY split').fetchall()]
        failed = [c['name'] for c in report['checks'] if not c['passed']]
        if failed:
            raise VerificationError('Quality checks failed: ' + ', '.join(failed))
        # Detect changed source files or manifest during loading before publishing.
        _, final_hash, final_inputs = verify_inputs(processed, trusted_manifest)
        if final_hash != manifest_hash or final_inputs != verified:
            raise VerificationError('Frozen inputs changed while building the warehouse')
        report['status'] = 'PASS'
        con.execute('CREATE TABLE build_metadata (content JSON)')
        con.execute('INSERT INTO build_metadata VALUES (?)', [json.dumps(report)])
        con.close()
        con = None
        report.update(status='PASS', database='warehouse.duckdb', database_sha256=digest(temporary_db))
        temporary_db.rename(final_db)
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error_type'] = type(exc).__name__
        # Do not copy parser exceptions that could include review text or user identifiers.
        report['error'] = str(exc) if isinstance(exc, VerificationError) else 'Build failed; check input availability and schema.'
        if con is not None:
            con.close()
        temporary_db.unlink(missing_ok=True)
        Path(str(temporary_db) + '.wal').unlink(missing_ok=True)
        final_db.unlink(missing_ok=True)
    report['seconds'] = round(time.monotonic() - started, 3)
    write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path, default=Path('data/processed'))
    parser.add_argument('--manifest', type=Path, default=PROJECT / 'docs/data_run/manifest.json',
                        help='Trusted frozen manifest, independent of the input directory')
    parser.add_argument('--output', type=Path, required=True, help='New run directory (must not exist)')
    args = parser.parse_args()
    report = build_warehouse(args.processed, args.manifest, args.output)
    print(f"{report['status']}: {len(report['checks'])} quality checks; report: {args.output / 'quality_report.json'}")
    if report['status'] != 'PASS':
        print(report['error'])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
