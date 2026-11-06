"""Parts E and F checks on synthetic fixtures. No Drive access or Amazon experiment.

Pure tests run without PyTorch. Model tests are skipped when torch is missing, and
evaluator tests when the frozen rec_eval.py is not in data/processed.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation import cold_start_data as cold
from src.evaluation.seed_stats import (aggregate_groups, aggregate_metrics, markdown_table,
                                       summarize)

ROOT = Path(__file__).resolve().parents[1]
REC_EVAL = ROOT / 'data/processed/rec_eval.py'
HAS_TORCH = importlib.util.find_spec('torch') is not None


def load_rec_eval():
    spec = importlib.util.spec_from_file_location('shared_rec_eval', REC_EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main_sequences():
    # Items 9 and 10 are cold. User 1 lost its validation item (9), user 2 has a cold test item.
    return pd.DataFrame({'user_idx': [0, 1, 2, 3],
                         'train_items': [[1, 2, 3], [2, 4, 5], [3, 10, 5, 6], [1, 4, 6]],
                         'val_item': [7, 9, 7, 8], 'test_item': [8, 3, 9, 2]})


def cold_frame():
    return pd.DataFrame({'user_idx': [0, 1, 2, 3],
                         'train_items': [[1, 2, 3], [2, 4, 5], [3, 5, 6], [1, 4, 6]],
                         'val_item': [7, -1, 7, 8], 'test_item': [8, 3, 9, 2],
                         'test_is_cold': [False, False, True, False],
                         'n_train': [3, 3, 3, 3], 'usable_for_training': [True] * 4})


def synthetic_data(sequences, n_items=10):
    return {'n_items': n_items, 'n_users': len(sequences), 'max_seq_len': 5,
            'sequences': sequences.reset_index(drop=True), 'protocol': {'ks': [5, 10]},
            'neg_val': np.ones((len(sequences), 3), dtype=np.int64),
            'neg_test': np.ones((len(sequences), 3), dtype=np.int64),
            'info': {'subset': {'kind': 'category'}}}


def write_cold_files(directory, frame, cold_items, info):
    path = Path(directory) / 'cold_start'
    path.mkdir()
    frame.to_parquet(path / 'cold_sequences.parquet', index=False)
    np.save(path / 'cold_items.npy', np.asarray(cold_items, dtype=np.int32))
    (path / 'cold_start_info.json').write_text(json.dumps(info))


INFO = {'n_cold_items': 2, 'n_cold_test_cases': 1, 'n_users_losing_val_case': 1}


class SeedStatsTests(unittest.TestCase):
    def test_sample_std_and_single_run(self):
        s = summarize([0.1, 0.2, 0.3])
        self.assertAlmostEqual(s['mean'], 0.2)
        self.assertAlmostEqual(s['std'], 0.1)
        self.assertIsNone(summarize([0.5])['std'])
        with self.assertRaises(ValueError):
            summarize([0.1, float('nan')])

    def test_aggregate_requires_same_metrics(self):
        agg = aggregate_metrics({43: {'NDCG@10': 0.3}, 42: {'NDCG@10': 0.1}})
        self.assertEqual(agg['NDCG@10']['values'], {'42': 0.1, '43': 0.3})
        with self.assertRaises(ValueError):
            aggregate_metrics({42: {'NDCG@10': 0.1}, 43: {'MRR': 0.1}})

    def test_groups_must_cover_same_cases(self):
        frame = pd.DataFrame({'grouping': ['user_group'], 'group': ['short'],
                              'n_cases': [5], 'NDCG@10': [0.1]})
        out = aggregate_groups({42: frame, 43: frame.assign(**{'NDCG@10': [0.3]})})
        self.assertAlmostEqual(out['NDCG@10_mean'].iloc[0], 0.2)
        with self.assertRaises(ValueError):
            aggregate_groups({42: frame, 43: frame.assign(n_cases=[6])})

    def test_markdown_table(self):
        agg = aggregate_metrics({42: {'NDCG@10': 0.1}, 43: {'NDCG@10': 0.3}})
        self.assertIn('| M | 0.2000 ± 0.1414 |', markdown_table([('M', agg)], ['NDCG@10']))


class ColdVariantTests(unittest.TestCase):
    def test_load_checks_and_removed_items(self):
        data = synthetic_data(main_sequences())
        with tempfile.TemporaryDirectory() as directory:
            write_cold_files(directory, cold_frame(), [9, 10], INFO)
            variant = cold.load_cold_variant(directory, data)
        # User 1 had cold val item 9, user 2 had cold training item 10.
        self.assertEqual({u: v.tolist() for u, v in variant['removed'].items()}, {1: [9], 2: [10]})
        self.assertEqual(cold.validation_users(variant['frame']).tolist(), [0, 2, 3])

    def test_load_rejects_cold_item_left_in_training(self):
        bad = cold_frame()
        bad.at[2, 'train_items'] = [3, 10, 5, 6]
        with tempfile.TemporaryDirectory() as directory:
            write_cold_files(directory, bad, [9, 10], INFO)
            with self.assertRaises(ValueError):
                cold.load_cold_variant(directory, synthetic_data(main_sequences()))

    def test_test_view_keeps_history_and_removes_placeholder(self):
        frame = cold_frame()
        view = cold.test_view(frame)
        for (_, f), (_, v) in zip(frame.iterrows(), view.iterrows()):
            expected = list(f.train_items) + ([f.val_item] if f.val_item != -1 else [])
            self.assertEqual(list(v.train_items) + [v.val_item], expected)
        self.assertNotIn(-1, view.val_item.tolist())

    def test_mask_items_hides_only_that_users_items(self):
        masked = cold.mask_items(lambda x, u: np.zeros((len(u), 5)), {7: np.array([2, 4])})
        scores = masked(np.zeros((2, 3)), np.array([7, 8]))
        self.assertTrue(np.isneginf(scores[0, [2, 4]]).all())
        self.assertTrue(np.isfinite(scores[1]).all())

    def test_chance_matches_brute_force(self):
        out = cold.chance_metrics([4, 12], ks=(5,))
        brute = {'NDCG@5': [], 'Recall@5': [], 'MRR': []}
        for m in (4, 12):
            ranks = np.arange(1, m + 1)
            brute['NDCG@5'].append(np.mean(np.where(ranks <= 5, 1 / np.log2(ranks + 1), 0)))
            brute['Recall@5'].append(np.mean(ranks <= 5))
            brute['MRR'].append(np.mean(1 / ranks))
        for key, values in brute.items():
            self.assertAlmostEqual(out[key], float(np.mean(values)))
        self.assertAlmostEqual(out['mean_relative_rank'], np.mean([5 / 8, 13 / 24]))

    def test_popularity_never_scores_cold_items(self):
        scores = cold.popularity_scores(cold_frame(), 10)(None, np.array([0]))
        self.assertEqual(scores[0, 9], 0)
        self.assertEqual(scores[0, 10], 0)
        self.assertEqual(scores[0, 3], 2)


@unittest.skipUnless(REC_EVAL.exists(), 'frozen rec_eval.py not in data/processed')
class ColdEvaluationTests(unittest.TestCase):
    def test_removed_items_masked_and_ranks_counted(self):
        evaluator = load_rec_eval()
        frame = cold_frame()
        data = synthetic_data(main_sequences())
        view = cold.test_view(frame)
        removed = {2: np.array([10])}
        # Scores rise with item id, so cold item 10 would rank first unless masked.
        scores = lambda x, u: np.tile(np.arange(11, dtype=float), (len(u), 1))
        summary, _ = cold.evaluate_cases(evaluator, data, view, scores, np.array([2]), removed)
        # User 2: history {3, 5, 6, 7} and item 10 hidden leave 1, 2, 4, 8, 9; target 9 is first.
        self.assertEqual(summary['median_rank'], 1.0)
        self.assertEqual(summary['mean_relative_rank'], 1 / 5)


@unittest.skipUnless(HAS_TORCH and REC_EVAL.exists(), 'needs torch and the frozen rec_eval.py')
class MultiSeedTests(unittest.TestCase):
    def setUp(self):
        import torch
        from src.data.sequence_dataset import make_dataloaders
        from src.models.sasrec import SASRec
        from src.training.trainer import TrainingSettings, fit, make_shared_validator

        self.evaluator = load_rec_eval()
        sequences = pd.DataFrame({'user_idx': [0, 1, 2, 3],
                                  'train_items': [[1, 2, 3], [2, 4, 5], [3, 5, 6], [1, 4, 6]],
                                  'val_item': [7, 8, 7, 8], 'test_item': [9, 10, 10, 9]})
        self.data = {**synthetic_data(sequences),
                     'eval_cases': pd.DataFrame({
                         'user_idx': [0, 1, 2, 3] * 2, 'split': ['val'] * 4 + ['test'] * 4,
                         'user_group': ['short', 'long'] * 4,
                         'target_item_group': ['head', 'tail'] * 4})}
        self.temp = tempfile.TemporaryDirectory()
        self.runs = []
        torch.set_num_threads(1)
        for seed in (42, 43):
            loaders = make_dataloaders(sequences, max_len=5, batch_size=2, eval_batch_size=2,
                                       seed=seed, pin_memory=False)
            metadata = {'purpose': 'unit_test', 'model_hyperparameters': {
                            'hidden_size': 8, 'dropout': 0.2, 'learning_rate': 0.001},
                        'model_layers': 1, 'model_heads': 2, 'max_seq_len': 5,
                        'eval_batch_size': 2, 'train_users': 4, 'validation_users': 4,
                        'manifest_sha256': 'm', 'verified_files': {'f': 'h'}, 'commit': str(seed)}
            torch.manual_seed(seed)
            model = SASRec(10, max_len=5, hidden_size=8, num_heads=2, num_layers=1, dropout=0.2)
            settings = TrainingSettings(epochs=2, patience=5, optimizer='Adam', learning_rate=0.001,
                                        weight_decay=0.0, seed=seed)
            validate = make_shared_validator(self.evaluator, self.data, batch_size=2)
            output = Path(self.temp.name) / f'seed_{seed}'
            fit(model, loaders['train'], validate, settings, output, metadata=metadata)
            self.runs.append(output)

    def tearDown(self):
        self.temp.cleanup()

    def test_evaluate_once_then_aggregate(self):
        from src.evaluation import multi_seed

        runs = multi_seed.check_runs(self.runs, self.data, {'f': 'h'}, 'm', allow_partial=False)
        item_cat = np.array(['padding'] + ['a', 'b'] * 5, dtype=object)
        records = [multi_seed.evaluate_run(d, r, self.evaluator, self.data, item_cat, 'cpu', 2)
                   for d, r in zip(self.runs, runs)]
        self.assertEqual([r['seed'] for r in records], [42, 43])
        self.assertTrue((self.runs[0] / 'test_ranks.npz').exists())
        again = multi_seed.evaluate_run(self.runs[0], runs[0], self.evaluator, self.data,
                                        item_cat, 'cpu', 2)
        self.assertEqual(again['test_full'], records[0]['test_full'])
        output = Path(self.temp.name) / 'aggregate'
        multi_seed.write_aggregate(records, output, 'Unit model')
        aggregate = json.loads((output / 'aggregate.json').read_text())
        self.assertEqual(aggregate['seeds'], [42, 43])
        self.assertEqual(aggregate['test_full']['NDCG@10']['n'], 2)
        self.assertIn('± ', (output / 'evaluation_table.md').read_text())

    def test_rejects_mixed_settings_and_partial_runs(self):
        from src.evaluation import multi_seed

        config = json.loads((self.runs[1] / 'run_config.json').read_text())
        config['training']['learning_rate'] = 0.0005
        (self.runs[1] / 'run_config.json').write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            multi_seed.check_runs(self.runs, self.data, {'f': 'h'}, 'm', allow_partial=False)
        with self.assertRaises(ValueError):
            multi_seed.check_runs(self.runs[:1], {**self.data, 'n_users': 5}, {'f': 'h'}, 'm',
                                  allow_partial=False)


if __name__ == '__main__':
    unittest.main()
