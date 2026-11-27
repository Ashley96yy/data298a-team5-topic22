"""Failure study on synthetic tables: features, rule order, sampling and the outputs."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import failure_study as F

DAY_MS = 86_400_000


def tables():
    # Items 1..8. Brands: 1-3 'a', 4-5 'b', 6 '' (unknown), 7-8 'c'. Categories: x for 1-4 and 7, y for 5, 6 and 8.
    items = pd.DataFrame({'item_idx': range(1, 9), 'brand_key': ['a', 'a', 'a', 'b', 'b', '', 'c', 'c'],
                          'cat_l2': ['x', 'x', 'x', 'x', 'y', 'y', 'x', 'y'],
                          'title': [f'item {i}' for i in range(1, 9)]}).set_index('item_idx')
    groups = pd.DataFrame({'item_idx': range(1, 9), 'item_group': ['head'] * 5 + ['tail', 'head', 'mid'],
                           'train_count': [9, 9, 9, 9, 9, 1, 9, 3]}).set_index('item_idx')
    sequences = pd.DataFrame({
        'user_idx': [0, 1, 2, 3, 4, 5],
        'train_items': [[1, 2], [1, 2], [1, 2], [4, 5], [1, 2], [1, 2]],
        'val_item': [3, 3, 3, 4, 3, 3],
        'test_item': [2, 8, 4, 5, 6, 3],
        'val_ts': [0] * 6,
        'test_ts': [10 * DAY_MS, 10 * DAY_MS, 10 * DAY_MS, 10 * DAY_MS, 10 * DAY_MS, 400 * DAY_MS],
        'n_train': [2, 2, 2, 2, 2, 2]})
    return sequences, items, groups


class FeatureAndRuleTests(unittest.TestCase):
    def setUp(self):
        self.sequences, self.items, self.groups = tables()
        # user 0: rank 15 in one seed (near miss); others are far misses.
        self.ranks = np.array([[15, 300, 300, 300, 300, 300], [40, 300, 300, 300, 300, 300],
                               [40, 300, 300, 300, 300, 300]], dtype=float)
        self.features = F.case_features(self.sequences, self.items, self.groups, self.ranks)

    def test_features(self):
        f = self.features.set_index('user_idx')
        self.assertTrue(f.loc[0, 'near_miss'])
        self.assertTrue(f.loc[1, 'unpopular_target'])                 # item 8 is mid
        self.assertTrue(f.loc[2, 'new_brand'])                        # brand b, history has only a
        self.assertFalse(f.loc[4, 'new_brand'])                       # unknown brand is never a new brand
        self.assertTrue(f.loc[3, 'category_switch'])                  # last item 4 is x, target 5 is y
        self.assertFalse(f.loc[2, 'category_switch'])                 # last item 3 is x, target 4 is x
        self.assertTrue(f.loc[5, 'long_gap'])
        self.assertAlmostEqual(f.loc[5, 'gap_days'], 400)
        self.assertTrue(f.fail_all.all())

    def test_first_matching_rule_wins(self):
        labels = F.assign_category(self.features).to_list()
        # user 4: tail item 6, so unpopular_target before its category switch
        self.assertEqual(labels, ['near_miss', 'unpopular_target', 'new_brand', 'category_switch',
                                  'unpopular_target', 'long_gap'])

    def test_other_when_no_rule_applies(self):
        quiet = self.features.copy()
        for name in F.CATEGORIES[:-1]:
            quiet[name] = False
        self.assertEqual(set(F.assign_category(quiet)), {'other'})

    def test_taxonomy_counts_sum_to_the_sample(self):
        sample = self.features.assign(category=F.assign_category(self.features), case=range(1, 7))
        sample = sample.set_index('case', drop=False)
        rows = F.taxonomy(sample)
        self.assertEqual([r['category'] for r in rows], F.CATEGORIES)
        self.assertEqual(sum(r['count'] for r in rows), len(sample))
        self.assertAlmostEqual(sum(r['share'] for r in rows), 1.0)
        self.assertEqual(rows[1]['example_case'], 2)

    def test_hits_and_failures_need_every_seed(self):
        ranks = np.array([[5, 300], [12, 300], [5, 3]], dtype=float)
        sequences = self.sequences.iloc[:2].reset_index(drop=True)
        f = F.case_features(sequences, self.items, self.groups, ranks)
        self.assertEqual(f.fail_all.to_list(), [False, False])
        self.assertEqual(f.hit_all.to_list(), [False, False])

    def test_ranks_must_match_users(self):
        with self.assertRaises(ValueError):
            F.case_features(self.sequences, self.items, self.groups, self.ranks[:, :3])


class CommandTests(unittest.TestCase):
    def test_sample_must_have_at_least_fifty_cases(self):
        args = type('A', (), {'sample_size': 49, 'output': '/nonexistent', 'processed': '.', 'model': ('m', [])})
        with self.assertRaisesRegex(ValueError, 'at least 50'):
            F.run(args)

    def test_end_to_end_on_synthetic_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / 'processed'
            (processed / 'groups').mkdir(parents=True)
            rng = np.random.default_rng(0)
            n = 120
            items = pd.DataFrame({'item_idx': range(1, 31), 'brand_key': [f'b{i % 6}' for i in range(30)],
                                  'cat_l2': [f'c{i % 3}' for i in range(30)], 'title': [f't{i}' for i in range(30)]})
            groups = pd.DataFrame({'item_idx': range(1, 31), 'item_group': ['head', 'mid', 'tail'] * 10,
                                   'train_count': [5] * 30})
            sequences = pd.DataFrame({
                'user_idx': np.arange(n), 'train_items': [list(rng.integers(1, 31, 3)) for _ in range(n)],
                'val_item': rng.integers(1, 31, n), 'test_item': rng.integers(1, 31, n),
                'val_ts': np.zeros(n, dtype=np.int64), 'test_ts': rng.integers(0, 300, n) * DAY_MS,
                'n_train': [3] * n})
            items.to_parquet(processed / 'items.parquet')
            groups.to_parquet(processed / 'groups/item_groups.parquet')
            sequences.to_parquet(processed / 'sequences.parquet')
            manifest = {'processed_files': {name: {'sha256': F.digest(processed / name)} for name in F.FILES}}
            (processed / 'manifest.json').write_text(json.dumps(manifest))
            dirs = []
            for seed in F.SEEDS:
                run = root / f'seed_{seed}'
                run.mkdir()
                ranks = np.where(np.arange(n) < 10, 3.0, rng.integers(11, 500, n).astype(float))
                np.savez(run / 'test_ranks.npz', user_idx=np.arange(n), ranks=ranks,
                         topk=rng.integers(1, 31, (n, 20)))
                ndcg = float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean())
                (run / 'test_evaluation.json').write_text(json.dumps({
                    'seed': seed, 'ranks_file': 'test_ranks.npz', 'n_test_cases': n,
                    'tie_policy': {'name': 'average_rank_then_metrics'}, 'test_full': {'NDCG@10': ndcg}}))
                dirs.append(str(run))
            out = root / 'out'
            with contextlib.redirect_stdout(io.StringIO()):
                F.main(['--processed', str(processed), '--model', 'Model 0=' + ','.join(dirs),
                        '--sample-size', '50', '--output', str(out)])
            payload = json.loads((out / 'failure_study.json').read_text())
            self.assertEqual(payload['n_failures'], 110)
            self.assertEqual(sum(r['count'] for r in payload['taxonomy_sample']), 50)
            cases = pd.read_csv(out / 'cases_for_review.csv')
            self.assertEqual(len(cases), 50)
            self.assertEqual(set(cases.columns) >= {'manual_category', 'notes', 'top3_predicted'}, True)
            self.assertIn('| total | 50 |', (out / 'failure_study.md').read_text())
            with self.assertRaises(SystemExit):
                F.main(['--processed', str(processed), '--model', 'Model 0=' + ','.join(dirs),
                        '--output', str(out)])


if __name__ == '__main__':
    unittest.main()
