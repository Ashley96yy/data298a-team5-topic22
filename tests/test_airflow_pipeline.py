"""Test the publication gate on tiny Parquet tables and a deliberately failed check."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    import pandas as pd
    import pyarrow  # noqa: F401
except ImportError:
    pd = None

from pipeline.airflow_tasks import check_quality


@unittest.skipUnless(pd is not None, 'needs pipeline/requirements.txt')
class QualityGateTest(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.work = Path(scratch.name)
        self.processed = self.work / 'processed'
        (self.processed / 'clients').mkdir(parents=True)
        (self.work / 'reports').mkdir()
        self.inter = pd.DataFrame({
            'user_idx': [0] * 5, 'item_idx': [1, 2, 3, 4, 5], 'rating': [5] * 5,
            'timestamp': [1, 2, 3, 4, 5], 'pos': [0, 1, 2, 3, 4],
            'split': ['train', 'train', 'train', 'val', 'test'], 'row_id': [0, 1, 2, 3, 4],
        })
        pd.DataFrame({
            'user_idx': [0], 'train_items': [[1, 2, 3]], 'train_ts': [[1, 2, 3]],
            'n_train': [3], 'val_item': [4], 'val_ts': [4], 'test_item': [5], 'test_ts': [5],
        }).to_parquet(self.processed / 'sequences.parquet')
        pd.DataFrame({'client_id': [0], 'n_local_train': [3]}).to_parquet(
            self.processed / 'clients' / 'clients.parquet')
        (self.processed / 'dataset_info.json').write_text(json.dumps(
            {'n_users': 1, 'n_items': 5, 'n_interactions': 5, 'kcore': 5}))
        leakage = {'all_passed': True, 'checks': {str(i): True for i in range(12)}}
        (self.processed / 'leakage_report.json').write_text(json.dumps(leakage))

    def write_manifest(self):
        files = {}
        for path in self.processed.rglob('*'):
            if path.is_file():
                data = path.read_bytes()
                files[str(path.relative_to(self.processed))] = {
                    'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        (self.processed / 'manifest.json').write_text(json.dumps({'processed_files': files}))

    def test_valid_parquet_passes_and_corrupt_file_is_rejected(self):
        path = self.processed / 'interactions.parquet'
        self.inter.to_parquet(path)
        self.write_manifest()
        check_quality(self.work)
        self.assertEqual(json.loads((self.work / 'reports' / 'quality_gate.json').read_text())['status'], 'PASS')
        path.write_bytes(path.read_bytes() + b'invalid')
        with self.assertRaisesRegex(ValueError, 'Manifest integrity check failed'):
            check_quality(self.work)

    def test_invalid_rating_fails_with_a_consistent_manifest(self):
        self.inter.loc[0, 'rating'] = 6
        self.inter.to_parquet(self.processed / 'interactions.parquet')
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'ratings'):
            check_quality(self.work)
