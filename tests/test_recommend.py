import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.serving.recommend import Recommender, build_artifacts


def toy_processed(root):
    items = pd.DataFrame({'item_idx': [1, 2, 3, 4, 5],
                          'title': ['Alpha Cream', 'Beta Serum', 'Gamma Lotion', 'Delta Mask', 'Epsilon Oil'],
                          'brand': ['A', 'B', 'C', 'D', 'E'], 'cat_l2': ['Face'] * 5,
                          'price_num': [1.0, 2.0, np.nan, 4.0, 5.0]})
    seq = pd.DataFrame({'user_idx': [0, 1, 2], 'train_items': [[1, 2, 3], [1, 2], [4, 5]],
                        'val_item': [4, 3, 1], 'test_item': [5, 4, 2]})
    items.to_parquet(Path(root) / 'items.parquet')
    seq.to_parquet(Path(root) / 'sequences.parquet')


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        proc, art = Path(self.tmp.name) / 'p', Path(self.tmp.name) / 'a'
        proc.mkdir()
        toy_processed(proc)
        self.info = build_artifacts(proc, art, n_example_users=2)
        self.rec = Recommender(art)

    def tearDown(self):
        self.tmp.cleanup()

    def test_artifacts_have_expected_counts(self):
        self.assertEqual(self.info['n_items'], 5)
        self.assertEqual(self.rec.popularity.tolist(), [0, 2, 2, 1, 1, 1])

    def test_markov_follows_most_frequent_successor(self):
        out = self.rec.recommend([1], 'markov', k=3)
        self.assertEqual(int(out.loc[0, 'item_idx']), 2)                 # 1 -> 2 happens twice

    def test_history_items_are_never_recommended(self):
        for method in ('popularity', 'markov'):
            out = self.rec.recommend([1, 2], method, k=3)
            self.assertFalse(set(out['item_idx']) & {1, 2})

    def test_empty_history_falls_back_to_popularity(self):
        out = self.rec.recommend([], 'markov', k=2)
        self.assertEqual(set(out['item_idx']), {1, 2})                   # the two most popular items

    def test_k_is_capped_by_remaining_items(self):
        self.assertEqual(len(self.rec.recommend([1, 2, 3, 4], 'popularity', k=10)), 1)

    def test_rejects_bad_ids_and_methods(self):
        with self.assertRaises(ValueError):
            self.rec.recommend([99], 'markov')
        with self.assertRaises(ValueError):
            self.rec.recommend([1], 'nope')

    def test_search_matches_title_and_brand(self):
        self.assertEqual(self.rec.search('serum')['item_idx'].tolist(), [2])
        self.assertEqual(self.rec.search('c')['item_idx'].tolist()[0], 1)  # "Alpha Cream" contains "c"
        self.assertTrue(self.rec.search('   ').empty)

    def test_registered_scorer_is_used(self):
        self.rec.register('custom', lambda history: np.arange(6, dtype=float))
        out = self.rec.recommend([], 'custom', k=2)
        self.assertEqual(out['item_idx'].tolist(), [5, 4])

    def test_missing_inputs_fail_clearly_and_leave_no_output_folder(self):
        empty, out = Path(self.tmp.name) / 'empty', Path(self.tmp.name) / 'never'
        empty.mkdir()
        with self.assertRaises(FileNotFoundError) as ctx:
            build_artifacts(empty, out)
        self.assertIn('items.parquet', str(ctx.exception))
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
