import unittest

import numpy as np

from src.baselines.simple import build_stats, itemknn_scorer, markov_scorer, popularity_scorer


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


if __name__ == '__main__':
    unittest.main()
