import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.retrieval.item_index import ItemIndex, build_index


def toy_index():
    items = pd.DataFrame({
        'item_idx': [1, 2, 3, 4, 5, 6],
        'title': ['Alpha Cream', 'Beta Serum', 'Gamma Lotion', 'Delta Mask', 'Epsilon Balm', 'Zeta Wash'],
        'brand': ['CeraVe', 'Olay', 'CeraVe', 'Aveeno', "Burt's Bees", 'Dove'],
        'category_path': ['Skin Care > Face > Creams & Moisturizers', 'Skin Care > Face > Treatments & Masks > Serums',
                          'Skin Care > Body > Moisturizers', 'Skin Care > Face > Treatments & Masks',
                          'Skin Care > Lip Care > Balms & Moisturizers', None],
        'cat_l2': ['Face', 'Face', 'Body', 'Face', 'Lip Care', None],
        'cat_l3': ['Creams & Moisturizers', 'Treatments & Masks', 'Moisturizers', 'Treatments & Masks',
                   'Balms & Moisturizers', None],
        'price_num': [12.0, 30.0, np.nan, 8.5, 4.0, 25.0],
        'price_band': ['$10-25', '$25-50', '', 'under $10', 'under $10', '$25-50'],
    })
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(6, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    meta = {'encoder': {'model': 'none', 'revision': None, 'query_prefix': ''}}
    return emb, items, meta


class ItemIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.emb, self.items, meta = toy_index()
        self.dir = Path(self.tmp.name) / 'idx'
        build_index(self.emb, self.items, meta, self.dir)
        self.index = ItemIndex.load(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_order_matches_brute_force(self):
        q = self.emb[0] + 0.1 * self.emb[3]
        hits = self.index.search_vector(q, k=6)
        expected = np.argsort(-(self.emb @ (q / np.linalg.norm(q))))
        self.assertEqual(hits.item_idx.tolist(), self.items.item_idx.to_numpy()[expected].tolist())
        self.assertTrue((np.diff(hits.score) <= 1e-9).all())
        self.assertEqual(hits['rank'].tolist(), list(range(1, 7)))
        self.assertEqual(hits.item_idx.iloc[0], 1)

    def test_price_filter_is_inclusive_and_drops_unpriced(self):
        hits = self.index.search_vector(self.emb[0], k=10, price_max=12)
        self.assertEqual(set(hits.item_idx), {1, 4, 5})
        hits = self.index.search_vector(self.emb[0], k=10, price_min=12, price_max=25)
        self.assertEqual(set(hits.item_idx), {1, 6})
        hits = self.index.search_vector(self.emb[0], k=10, price_max=12, include_unpriced=True)
        self.assertEqual(set(hits.item_idx), {1, 3, 4, 5})
        with self.assertRaises(ValueError):
            self.index.search_vector(self.emb[0], price_min=30, price_max=10)

    def test_category_level_and_path_prefix(self):
        self.assertEqual(set(self.index.search_vector(self.emb[0], k=10, categories=['face']).item_idx), {1, 2, 4})
        self.assertEqual(set(self.index.search_vector(self.emb[0], k=10, categories='Treatments & Masks').item_idx), {2, 4})
        self.assertEqual(set(self.index.search_vector(self.emb[0], k=10, categories=['Face > Treatments & Masks > Serums']).item_idx), {2})
        self.assertEqual(set(self.index.search_vector(self.emb[0], k=10, categories=['Body', 'Lip Care']).item_idx), {3, 5})
        with self.assertRaises(ValueError) as e:
            self.index.search_vector(self.emb[0], categories=['Fcae'])
        self.assertIn('face', str(e.exception))

    def test_brand_exclude_and_combined_filters(self):
        self.assertEqual(set(self.index.search_vector(self.emb[0], k=10, brands=['cerave']).item_idx), {1, 3})
        with self.assertRaises(ValueError):
            self.index.search_vector(self.emb[0], brands=['Cerave Pro'])
        hits = self.index.search_vector(self.emb[0], k=10, categories=['Face'], price_max=20, exclude_items=[1, 999])
        self.assertEqual(hits.item_idx.tolist(), [4])

    def test_no_match_and_min_score_give_empty_results(self):
        self.assertEqual(len(self.index.search_vector(self.emb[0], categories=['Lip Care'], price_min=50)), 0)
        self.assertEqual(len(self.index.search_vector(self.emb[0], min_score=1.01)), 0)
        self.assertEqual(len(self.index.search_vector(self.emb[0], k=3)), 3)

    def test_bad_inputs_and_tampered_files_are_rejected(self):
        for bad in (np.zeros(8), np.ones(5), [np.nan] * 8):
            with self.assertRaises(ValueError):
                self.index.search_vector(bad)
        with self.assertRaises(ValueError):
            self.index.search_vector(self.emb[0], k=0)
        np.save(self.dir / 'embeddings.npy', self.emb * 1.0001, allow_pickle=False)
        with self.assertRaises(ValueError):
            ItemIndex.load(self.dir)

    def test_missing_category_is_not_a_category(self):
        items = self.items.copy()
        items.loc[5, 'category_path'] = float('nan')                       # how pandas may read a missing value
        index = ItemIndex(self.emb, items, {'encoder': {}})
        self.assertEqual(index.levels[5], [])
        self.assertNotIn('nan', set(index.categories().category))
        self.assertNotIn(6, set(index.search_vector(self.emb[0], k=10, categories=['Face']).item_idx))

    def test_category_listing(self):
        cats = self.index.categories(level=2)
        self.assertEqual(dict(zip(cats.cat_l2, cats.n_items)), {'Face': 3, 'Body': 1, 'Lip Care': 1})
        self.assertIn('treatments & masks', set(self.index.categories().category))


if __name__ == '__main__':
    unittest.main()
