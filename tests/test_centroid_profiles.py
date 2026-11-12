import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.centroid_profiles import build_centroid_profiles, check_item_embeddings, main, shuffle_profiles
from src.data.profile_file import file_sha256, read_profile_file


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def item_embeddings(n_items=5, d=4, seed=0):
    E = np.zeros((n_items + 1, d), dtype=np.float32)
    E[1:] = np.random.default_rng(seed).normal(size=(n_items, d))
    E[1:] /= np.linalg.norm(E[1:], axis=1, keepdims=True)
    return E


class BuildTests(unittest.TestCase):
    def test_profile_is_the_normalised_mean_of_the_last_window_training_items(self):
        E = item_embeddings()
        seq = pd.DataFrame({'user_idx': [7, 3], 'train_items': [[1, 2, 3], [4, 5, 1, 2]]})
        users, profiles = build_centroid_profiles(seq, E, window=2)
        self.assertEqual(users.tolist(), [7, 3])
        self.assertTrue(np.allclose(profiles[0], unit(E[2] + E[3]), atol=1e-6))      # only the last 2 items
        self.assertTrue(np.allclose(profiles[1], unit(E[1] + E[2]), atol=1e-6))
        self.assertTrue(np.allclose(np.linalg.norm(profiles, axis=1), 1.0, atol=1e-6))

    def test_only_training_items_are_read(self):
        E = item_embeddings()
        base = pd.DataFrame({'user_idx': [0], 'train_items': [[1, 2, 3]]})
        with_eval = base.assign(val_item=[4], test_item=[5])                              # extra columns must not matter
        self.assertTrue(np.array_equal(build_centroid_profiles(base, E)[1], build_centroid_profiles(with_eval, E)[1]))

    def test_bad_inputs_are_rejected(self):
        E = item_embeddings()
        for items in ([0, 1], [1, 6], []):
            with self.subTest(items=items):
                with self.assertRaises(ValueError):
                    build_centroid_profiles(pd.DataFrame({'user_idx': [0], 'train_items': [items]}), E)
        bad = E.copy(); bad[0, 0] = 0.5
        with self.assertRaises(ValueError):
            check_item_embeddings(bad, 5)                                                  # padding row must be zero
        with self.assertRaises(ValueError):
            check_item_embeddings(E[:-1], 5)                                               # wrong number of rows
        nan = E.copy(); nan[2, 1] = np.nan
        with self.assertRaises(ValueError):
            check_item_embeddings(nan, 5)


class ShuffleTests(unittest.TestCase):
    def test_every_user_gets_another_users_profile_and_no_vector_is_lost(self):
        profiles = np.random.default_rng(1).normal(size=(50, 4)).astype(np.float32)
        shuffled = shuffle_profiles(profiles, seed=7)
        self.assertFalse(any(np.array_equal(shuffled[i], profiles[i]) for i in range(50)))     # nobody keeps their own
        self.assertEqual(sorted(map(tuple, shuffled.tolist())), sorted(map(tuple, profiles.tolist())))   # same vectors
        self.assertTrue(np.array_equal(shuffled, shuffle_profiles(profiles, seed=7)))            # deterministic
        self.assertFalse(np.array_equal(shuffled, shuffle_profiles(profiles, seed=8)))

    def test_needs_two_users(self):
        with self.assertRaises(ValueError):
            shuffle_profiles(np.ones((1, 3), dtype=np.float32), seed=0)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.processed = root / 'processed'; self.processed.mkdir()
        (self.processed / 'manifest.json').write_text('{"frozen": true}')
        pd.DataFrame({'user_idx': [0, 1, 2], 'train_items': [[1, 2, 3], [2, 3, 4], [5, 1, 2]]}).to_parquet(self.processed / 'sequences.parquet')
        self.emb = root / 'emb.npy'
        np.save(self.emb, item_embeddings())
        self.meta = root / 'meta.json'
        self.out = root / 'out' / 'profiles.npz'

    def tearDown(self):
        self.tmp.cleanup()

    def write_meta(self, manifest):
        self.meta.write_text(json.dumps({'manifest_sha256': manifest, 'n_items': 5, 'model_id': 'test-encoder', 'revision': 'abc'}))

    def run_main(self):
        main(['--processed', str(self.processed), '--item-embeddings', str(self.emb), '--item-metadata', str(self.meta), '--output', str(self.out)])

    def test_writes_a_valid_profile_file_with_the_control_source(self):
        self.write_meta(file_sha256(self.processed / 'manifest.json'))
        self.run_main()
        p = read_profile_file(self.out)                                                    # the strict reader accepts it
        self.assertEqual(p['source'], 'item_text_centroid')
        self.assertEqual(p['user_idx'].tolist(), [0, 1, 2])
        self.assertEqual(p['embeddings'].shape, (3, 4))
        meta = json.loads(self.out.with_suffix('.meta.json').read_text())
        self.assertEqual(meta['item_encoder'], 'test-encoder')
        self.assertEqual(meta['item_embedding_sha256'], file_sha256(self.emb))

    def test_shuffle_option_writes_a_file_labelled_as_a_control(self):
        self.write_meta(file_sha256(self.processed / 'manifest.json'))
        main(['--processed', str(self.processed), '--item-embeddings', str(self.emb), '--item-metadata', str(self.meta),
              '--output', str(self.out), '--shuffle-seed', '3'])
        p = read_profile_file(self.out)
        self.assertEqual(p['source'], 'shuffled_control')
        self.assertEqual(json.loads(self.out.with_suffix('.meta.json').read_text())['shuffle_seed'], 3)

    def test_embeddings_from_a_different_frozen_dataset_are_refused(self):
        self.write_meta('0' * 64)
        with self.assertRaises(SystemExit) as ctx:
            self.run_main()
        self.assertIn('different frozen dataset', str(ctx.exception))
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
