"""Small relational fixtures test rejection and publication, without frozen data access."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from src.warehouse.load import REQUIRED, TABLES, build_warehouse, digest, verify_inputs


class WarehouseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.processed = self.root / 'processed'
        self.processed.mkdir()
        self.trusted = self.root / 'trusted.json'
        self.output = self.root / 'run'
        self.con = duckdb.connect()
        self.addCleanup(self.con.close)
        self.info = {'n_users': 2, 'n_items': 5, 'n_interactions': 10, 'kcore': 2}
        queries = {
            'user_map': "SELECT i::INTEGER user_idx, 'u'||i user_id FROM range(2) r(i)",
            'item_map': "SELECT i::INTEGER item_idx, 'p'||i parent_asin FROM range(1,6) r(i)",
            'items': "SELECT i::INTEGER item_idx, 'p'||i parent_asin, 'product'||i title FROM range(1,6) r(i)",
            'item_text': "SELECT i::INTEGER item_idx, 'p'||i parent_asin, 'text'||i item_text FROM range(1,6) r(i)",
            'interactions': """SELECT u::INTEGER user_idx, ((p+u)%5+1)::INTEGER item_idx,
                p::BIGINT AS "timestamp", p::INTEGER pos, (u*5+p)::BIGINT row_id,
                CASE p WHEN 4 THEN 'test' WHEN 3 THEN 'val' ELSE 'train' END split
                FROM range(2) x(u) CROSS JOIN range(5) y(p)""",
            'sequences': """SELECT u::INTEGER user_idx, [1+u,2+u,3+u] train_items,
                [0,1,2] train_ts, (4+u)::INTEGER val_item, 3::BIGINT val_ts,
                ((4+u)%5+1)::INTEGER test_item, 4::BIGINT test_ts, 3::INTEGER n_train FROM range(2) x(u)""",
            'train_reviews': "SELECT 'u0' user_id,'p1' parent_asin,1::INTEGER item_idx,'train' split",
            'interaction_side_info': 'SELECT user_idx,item_idx,split FROM interactions',
            'eval_cases': "SELECT user_idx,item_idx AS target,split FROM interactions WHERE split<>'train'",
            'item_groups': """SELECT m.item_idx, count(i.item_idx)::BIGINT train_count FROM item_map m
                 LEFT JOIN interactions i ON m.item_idx=i.item_idx AND i.split='train' GROUP BY m.item_idx""",
            'user_groups': 'SELECT user_idx,n_train FROM sequences',
            'clients': 'SELECT user_idx client_id,n_train n_local_train,n_train+2 n_local_total FROM sequences',
            'cold_sequences': 'SELECT * FROM sequences',
        }
        for table, query in queries.items():
            self.con.execute(f'CREATE TABLE {table} AS {query}')
        self.export()
        for name in REQUIRED - set(TABLES.values()):
            p = self.processed / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.info) if name == 'dataset_info.json' else '{}')
        self.freeze()

    def export(self):
        for table, name in TABLES.items():
            p = self.processed / name
            p.parent.mkdir(parents=True, exist_ok=True)
            self.con.execute(f'COPY {table} TO ? (FORMAT PARQUET)', [str(p)])

    def freeze(self):
        manifest = {'dataset_info': self.info, 'processed_files': {
            name: {'bytes': (self.processed/name).stat().st_size, 'sha256': digest(self.processed/name)}
            for name in REQUIRED}}
        raw = json.dumps(manifest)
        self.trusted.write_text(raw)
        (self.processed/'manifest.json').write_text(raw)

    def build(self):
        return build_warehouse(self.processed, self.trusted, self.output)

    def assert_failed(self, report):
        self.assertEqual(report['status'], 'FAIL', report)
        self.assertFalse((self.output/'warehouse.duckdb').exists())
        self.assertFalse((self.output/'building.duckdb').exists())
        self.assertTrue((self.output/'quality_report.json').is_file())

    def test_complete_build_persists_without_source_files(self):
        report = self.build()
        self.assertEqual(report['status'], 'PASS', report)
        self.assertEqual(len(report['verified_inputs']), 22)
        self.processed.rename(self.root/'moved_inputs')
        with duckdb.connect(str(self.output/'warehouse.duckdb'), read_only=True) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM interactions').fetchone()[0], 10)
            self.assertEqual(db.execute('SELECT sum(interactions) FROM split_summary').fetchone()[0], 10)
            metadata = json.loads(db.execute('SELECT content FROM build_metadata').fetchone()[0])
            self.assertEqual(metadata['status'], 'PASS')
        self.assertEqual(digest(self.output/'warehouse.duckdb'), report['database_sha256'])
        self.assertNotIn(b'\r\n', (self.output/'quality_checks.csv').read_bytes())

    def test_existing_output_is_never_overwritten(self):
        self.assertEqual(self.build()['status'], 'PASS')
        before = digest(self.output/'warehouse.duckdb')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(digest(self.output/'warehouse.duckdb'), before)

    def test_missing_file_fails_before_loading(self):
        (self.processed/'eval/neg_test.npy').unlink()
        self.assert_failed(self.build())

    def test_same_size_corruption_rejected(self):
        (self.processed/'rec_eval.py').write_text('[]')
        report = self.build()
        self.assert_failed(report)
        self.assertIn('SHA-256 mismatch', report['error'])

    def test_replaced_local_manifest_rejected(self):
        (self.processed/'manifest.json').write_text('{}')
        self.assert_failed(self.build())

    def test_counts_checked_independently_of_hashes(self):
        self.info['n_interactions'] = 11
        (self.processed/'dataset_info.json').write_text(json.dumps(self.info))
        self.freeze()
        report = self.build()
        self.assert_failed(report)
        self.assertIn('interactions_row_count', report['error'])

    def test_duplicate_identity_rejected(self):
        self.con.execute("UPDATE user_map SET user_id='u0'")
        self.export(); self.freeze()
        report = self.build()
        self.assert_failed(report)
        self.assertIn('user_map_user_id_unique_not_null', report['error'])

    def test_chronology_rejected(self):
        self.con.execute("UPDATE interactions SET timestamp=100 WHERE split='train'")
        self.export(); self.freeze()
        report = self.build()
        self.assert_failed(report)
        self.assertIn('chronological_positions_and_splits', report['error'])

    def test_sequence_leakage_rejected(self):
        self.con.execute('UPDATE sequences SET val_item=1 WHERE user_idx=0')
        self.export(); self.freeze()
        report = self.build()
        self.assert_failed(report)
        self.assertIn('sequence_interaction_agreement', report['error'])

    def test_heldout_review_cannot_enter_training_text(self):
        self.con.execute("UPDATE train_reviews SET parent_asin='p5',item_idx=5")
        self.export(); self.freeze()
        report = self.build()
        self.assert_failed(report)
        self.assertIn('training_review_provenance', report['error'])

    def test_source_change_during_load_prevents_publication(self):
        original = verify_inputs(self.processed, self.trusted)
        with patch('src.warehouse.load.verify_inputs', side_effect=[original, ValueError('changed')]):
            self.assert_failed(self.build())

    def test_manifest_path_escape_rejected(self):
        manifest = json.loads(self.trusted.read_text())
        manifest['processed_files']['../outside'] = {'bytes': 0, 'sha256': ''}
        self.trusted.write_text(json.dumps(manifest))
        (self.processed/'manifest.json').write_text(self.trusted.read_text())
        report = self.build()
        self.assert_failed(report)
        self.assertIn('Unsafe manifest path', report['error'])


if __name__ == '__main__':
    unittest.main()
