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

    def write_raw(self, **arrays):
        """Write an NPZ directly (bypassing write_profile_file), the way another pipeline could."""
        base = dict(user_idx=np.array([3, 1, 0, 2]), embeddings=self.emb.astype(np.float32),
                    source=np.array('llm_profile'), train_items_only=np.array(True))
        base.update(arrays)
        np.savez(self.path, **base)

    def test_refuses_profiles_not_built_from_training_items(self):
        self.write_raw(train_items_only=np.array(False))
        with self.assertRaises(ValueError):
            read_profile_file(self.path)

    def test_writer_accepts_only_a_real_scalar_boolean_true(self):
        for ok in (True, np.True_, np.array(True)):
            with self.subTest(ok=repr(ok)):
                self.write(train_items_only=ok)
                self.assertEqual(read_profile_file(self.path)['source'], 'llm_profile')
        for bad in ('False', 'false', 'True', 'true', '0', '', 0, 1, 2, 1.0, None, [True], np.array([True]),
                    np.array('True'), np.array(1)):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    self.write(train_items_only=bad)

    def test_writer_refuses_to_write_a_false_declaration(self):
        for false in (False, np.False_, np.array(False)):
            with self.subTest(false=repr(false)):
                with self.assertRaises(ValueError):
                    self.write(train_items_only=false)

    def test_reader_rejects_mistyped_declarations_in_a_directly_written_npz(self):
        for bad in (np.array('False'), np.array('false'), np.array('True'), np.array('0'), np.array(0), np.array(1),
                    np.array(2), np.array(1.0), np.array([True, True]), np.array([True]), np.array([], dtype=bool)):
            with self.subTest(bad=repr(bad)):
                self.write_raw(train_items_only=bad)
                with self.assertRaises(ValueError):
                    read_profile_file(self.path)

    def test_reader_accepts_a_directly_written_numpy_bool(self):
        self.write_raw(train_items_only=np.array(np.True_))
        self.assertEqual(read_profile_file(self.path)['source'], 'llm_profile')

    def test_reader_rejects_a_non_scalar_or_non_string_source(self):
        for bad in (np.array(['llm_profile']), np.array(3)):
            with self.subTest(bad=repr(bad)):
                self.write_raw(source=bad)
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
