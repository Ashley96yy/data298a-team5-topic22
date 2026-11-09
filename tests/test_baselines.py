import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baselines.simple import build_stats, itemknn_scorer, main, markov_scorer, popularity_scorer, provenance
from src.evaluation.common import EVALUATOR_FILES, digest


class BaselineTests(unittest.TestCase):
    def setUp(self):
        # 3 users, items 1..5. Transitions: 1->2 (x2), 2->3, 4->5.
        self.train = [[1, 2, 3], [1, 2], [4, 5]]
        self.pop, self.T, self.S = build_stats(self.train, n_items=5)

    def test_popularity_counts(self):
        self.assertEqual(self.pop.tolist(), [0, 2, 2, 1, 1, 1])
        scores = popularity_scorer(self.pop)(np.zeros((2, 3), dtype=int), None)
        self.assertEqual(scores.shape, (2, 6))
        self.assertGreater(scores[0, 1], scores[0, 3])

    def test_markov_ranks_most_frequent_successor_first(self):
        scores = markov_scorer(self.T, self.pop)(np.array([[0, 0, 1]]), None)
        self.assertEqual(int(np.argmax(scores[0, 1:])) + 1, 2)      # after item 1 comes 2 (twice)
        scores = markov_scorer(self.T, self.pop)(np.array([[0, 0, 4]]), None)
        self.assertEqual(int(np.argmax(scores[0, 1:])) + 1, 5)

    def test_markov_padding_only_user_falls_back_to_popularity(self):
        scores = markov_scorer(self.T, self.pop)(np.array([[0, 0, 0]]), None)
        self.assertTrue(np.all(scores[0, 1:] > 0))
        self.assertGreater(scores[0, 1], scores[0, 5])

    def test_itemknn_uses_whole_history_and_is_finite(self):
        scores = itemknn_scorer(self.S, self.pop)(np.array([[0, 1, 2]]), None)
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[0, 3], scores[0, 5])               # 3 co-occurs with 1 and 2; 5 does not
        self.assertAlmostEqual(self.S[1, 2], 2 / np.sqrt(2 * 2))     # co-occur in 2 users; each item in 2 users

    def test_cosine_is_symmetric_and_zero_diagonal(self):
        dense = self.S.toarray()
        self.assertTrue(np.allclose(dense, dense.T))
        self.assertTrue(np.allclose(np.diag(dense), 0))


def fake_processed(root, with_manifest=True, tamper=None, drop=None, omit_from_manifest=None):
    """A stand-in frozen dataset: the nine files the evaluator needs, plus a manifest with their SHA-256."""
    root = Path(root)
    files = {}
    for i, name in enumerate(EVALUATOR_FILES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f'content {i} {name}'.encode())
        files[name] = {'sha256': digest(path)}
    if omit_from_manifest:
        del files[omit_from_manifest]
    if with_manifest:
        (root / 'manifest.json').write_text(json.dumps({'processed_files': files}))
    if tamper:
        (root / tamper).write_bytes(b'changed after the manifest was written')
    if drop:
        (root / drop).unlink()
    return root


class PopularityTermTests(unittest.TestCase):
    def test_the_popularity_term_can_reorder_a_near_tie_in_the_cosine_scorer(self):
        # Documents the behaviour described in the module docstring: a fixed additive term, not a pure tie-break.
        import scipy.sparse as sp
        S = sp.lil_matrix((4, 4))
        S[3, 1], S[3, 2] = 0.5000000, 0.5000005               # raw scores: item 2 is (very slightly) ahead
        pop = np.array([0.0, 100.0, 1.0, 0.0])                # scaled popularity: item 1 -> 1, item 2 -> 0.01
        scores = itemknn_scorer(S.tocsr(), pop)(np.array([[0, 0, 3]]), None)
        self.assertGreater(scores[0, 1], scores[0, 2])        # order reversed by 1e-6 * popularity

    def test_the_markov_term_only_reorders_equal_counts(self):
        T, pop = build_stats([[1, 2], [1, 2], [1, 3]], n_items=3)[1], np.array([0.0, 3.0, 2.0, 1.0])
        scores = markov_scorer(T, pop)(np.array([[0, 1]]), None)
        self.assertGreater(scores[0, 2], scores[0, 3])        # counts 2 versus 1: popularity cannot change that


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / 'results'

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, processed):
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(io.StringIO()):
            main(['--processed', str(processed), '--output', str(self.out)])
        return str(ctx.exception)

    def test_missing_manifest_stops_everything_and_writes_nothing(self):
        message = self.run_main(fake_processed(self.root / 'p', with_manifest=False))
        self.assertIn('could not be verified', message)
        self.assertFalse(self.out.exists())

    def test_a_file_changed_after_the_manifest_is_rejected(self):
        message = self.run_main(fake_processed(self.root / 'p', tamper='sequences.parquet'))
        self.assertIn('checksum mismatch', message.lower())
        self.assertFalse(self.out.exists())

    def test_a_missing_input_file_is_rejected(self):
        message = self.run_main(fake_processed(self.root / 'p', drop='eval/neg_test.npy'))
        self.assertIn('could not be verified', message)
        self.assertFalse(self.out.exists())

    def test_a_file_missing_from_the_manifest_is_rejected(self):
        message = self.run_main(fake_processed(self.root / 'p', omit_from_manifest='rec_eval.py'))
        self.assertIn('could not be verified', message)
        self.assertFalse(self.out.exists())


class ProvenanceTests(unittest.TestCase):
    def test_provenance_records_hashes_counts_code_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = fake_processed(tmp)
            hashes = {name: digest(Path(tmp) / name) for name in EVALUATOR_FILES}
            data = {'n_users': 5, 'n_items': 7, 'max_seq_len': 50}
            p = provenance(processed, hashes, data, {'val': 5, 'test': 5})
            self.assertEqual(p['manifest_sha256'], digest(Path(tmp) / 'manifest.json'))
            self.assertEqual(p['verified_files'], hashes)
            self.assertEqual(p['dataset'], {'n_users': 5, 'n_items': 7, 'max_seq_len': 50, 'cases': {'val': 5, 'test': 5}})
            self.assertEqual(len(p['code']['simple_py_sha256']), 64)
            self.assertTrue(p['environment']['python'])
            self.assertIn('fixed additive', p['scorers']['popularity_term'])


if __name__ == '__main__':
    unittest.main()
