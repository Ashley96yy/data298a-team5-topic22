import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.serving.monitor import (checks_table, cleaning_table, find_run_dir, format_utc, load_run, stage_table,
                                 summary, timing_table)


def write_record(d, with_status=None, quality='PASS'):
    d = Path(d)
    (d / 'records').mkdir(parents=True, exist_ok=True)
    (d / 'reports').mkdir(parents=True, exist_ok=True)
    (d / 'manifest.json').write_text(json.dumps({'created_utc': '2026-10-18T03:19:49+00:00', 'versions': {'python': '3.11'}}))
    (d / 'leakage_report.json').write_text(json.dumps({'checks': {'a': True, 'b': True, 'c': False}, 'all_passed': False}))
    (d / 'reports' / 'quality_gate.json').write_text(json.dumps({'status': quality, 'checks': ['nulls', 'ratings']}))
    (d / 'records' / 'cleaning_log.json').write_text(json.dumps({'steps': [
        {'step': 'rule 1', 'rows_before': 10, 'rows_after': 8, 'removed': 2, 'note': ''}]}))
    pd.DataFrame({'iteration': [1, 2], 'rows_before': [10, 8], 'rows_after': [8, 8], 'users': [3, 3], 'items': [4, 4]}
                 ).to_csv(d / 'reports' / 'kcore_iterations.csv', index=False)
    pd.DataFrame({'stage': ['raw', 'final'], 'users': [5, 3], 'items': [6, 4], 'interactions': [10, 8],
                  'avg_seq_len': [2.0, 2.7]}).to_csv(d / 'reports' / 'dataset_statistics.csv', index=False)
    if with_status:
        (d / 'run_status.json').write_text(json.dumps(with_status))


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_run_gives_none(self):
        self.assertIsNone(find_run_dir(self.root))

    def test_falls_back_to_the_committed_manual_record(self):
        write_record(self.root / 'docs' / 'data_run')
        self.assertEqual(find_run_dir(self.root), self.root / 'docs' / 'data_run')

    def test_newest_automated_run_wins_even_if_it_failed(self):
        write_record(self.root / 'docs' / 'data_run')
        write_record(self.root / 'pipeline_runs' / '20261101T090000Z', with_status={'status': 'PASS'})
        failed = self.root / 'pipeline_runs' / '20261108T090000Z'
        failed.mkdir(parents=True)
        (failed / 'run_status.json').write_text(json.dumps({'status': 'FAILED', 'failed_step': '03_cleaning', 'notebooks': []}))
        self.assertEqual(find_run_dir(self.root), failed)

    def test_summary_counts_come_from_the_files(self):
        write_record(self.root)
        s = summary(load_run(self.root))
        self.assertEqual((s['leakage_passed'], s['leakage_total']), (2, 3))
        self.assertEqual(s['final_interactions'], 8)
        self.assertEqual(s['last_run_utc'], '2026-10-18 03:19 UTC')
        self.assertEqual(s['source'], 'manual run record')

    def test_automated_status_overrides_the_headline(self):
        write_record(self.root, with_status={'status': 'PASS', 'started_utc': 'a', 'finished_utc': 'b', 'notebooks': [
            {'name': '01', 'status': 'PASS', 'seconds': 5}]})
        run = load_run(self.root)
        s = summary(run)
        self.assertEqual((s['status'], s['last_run_utc'], s['source']), ('PASS', 'b', 'automated run'))
        self.assertEqual(len(timing_table(run)), 1)

    def test_tables(self):
        write_record(self.root)
        run = load_run(self.root)
        self.assertEqual(cleaning_table(run)['removed'].tolist(), [2])
        self.assertEqual(stage_table(run)['stage'].tolist(), ['raw', 'final'])
        checks = checks_table(run)
        self.assertEqual(checks[checks.group == 'Leakage']['result'].tolist(), ['PASS', 'PASS', 'FAIL'])
        self.assertEqual(checks[checks.group == 'Quality gate']['result'].tolist(), ['PASS', 'PASS'])

    def test_missing_files_do_not_crash(self):
        (self.root / 'manifest.json').write_text(json.dumps({'created_utc': 'x'}))
        run = load_run(self.root)
        self.assertEqual(len(stage_table(run)), 0)
        self.assertEqual(len(cleaning_table(run)), 0)
        self.assertEqual(summary(run)['leakage_total'], 0)

    def test_format_utc(self):
        self.assertEqual(format_utc('2026-10-18T03:19:49.529511+00:00'), '2026-10-18 03:19 UTC')
        self.assertEqual(format_utc('2026-11-07T21:05:00Z'), '2026-11-07 21:05 UTC')
        self.assertEqual(format_utc('2026-10-17T20:19:00-07:00'), '2026-10-18 03:19 UTC')   # converted to UTC
        self.assertEqual(format_utc('not a date'), 'not a date')
        self.assertIsNone(format_utc(None))


if __name__ == '__main__':
    unittest.main()
