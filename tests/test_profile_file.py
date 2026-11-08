import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.profile_file import read_profile_file, write_profile_file


class ProfileFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'p.npz'
        self.emb = np.random.default_rng(0).normal(size=(4, 6))

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, **overrides):
        args = dict(user_idx=[3, 1, 0, 2], embeddings=self.emb, source='llm_profile', train_items_only=True)
        args.update(overrides)
        write_profile_file(self.path, **args)

    def test_round_trip_and_hash(self):
        self.write()
        p = read_profile_file(self.path)
        self.assertEqual(p['user_idx'].tolist(), [3, 1, 0, 2])
        self.assertTrue(np.allclose(p['embeddings'], self.emb, atol=1e-6))
        self.assertEqual(p['source'], 'llm_profile')
        self.assertEqual(len(p['sha256']), 64)
        self.assertEqual(read_profile_file(self.path)['sha256'], p['sha256'])      # stable

    def test_refuses_profiles_not_built_from_training_items(self):
        self.write(train_items_only=False)
        with self.assertRaises(ValueError):
            read_profile_file(self.path)

    def test_synthetic_needs_explicit_permission(self):
        self.write(source='synthetic')
        with self.assertRaises(ValueError):
            read_profile_file(self.path)
        self.assertEqual(read_profile_file(self.path, allow_synthetic=True)['source'], 'synthetic')

    def test_rejects_bad_content(self):
        bad_cases = [dict(user_idx=[0, 0, 1, 2]),                                   # duplicate ids
                     dict(user_idx=[-1, 0, 1, 2]),                                  # negative id
                     dict(user_idx=[0, 1, 2]),                                      # 3 ids, 4 embedding rows
                     dict(source='made_up'),
                     dict(embeddings=np.where(np.eye(4, 6) > 0, np.nan, 1.0))]     # non-finite values
        for bad in bad_cases:
            with self.subTest(bad=sorted(bad)):
                self.write(**bad)
                with self.assertRaises(ValueError):
                    read_profile_file(self.path)

    def test_missing_keys_and_missing_file(self):
        np.savez(self.path, user_idx=np.arange(2))
        with self.assertRaises(ValueError):
            read_profile_file(self.path)
        with self.assertRaises(FileNotFoundError):
            read_profile_file(Path(self.tmp.name) / 'nope.npz')


if __name__ == '__main__':
    unittest.main()
