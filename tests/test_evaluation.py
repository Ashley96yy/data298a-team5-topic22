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
from src.evaluation.cold_start import check_cold_runs
from src.evaluation.common import (configuration_mismatches, expected_configuration,
                                   partial_reasons)
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
FILLED = {'batch_size': 128, 'epochs': 200, 'num_layers': 2, 'num_heads': 1, 'weight_decay': 0.0,
          'early_stopping_patience': 10, 'seeds': [42, 43, 44]}


def team_config(directory, training=FILLED):
    """The repo config with its training settings replaced, written to a temp file."""
    config = json.loads((ROOT / 'configs/config.json').read_text())
    config['training'] = dict(training)
    path = Path(directory) / 'team_config.json'
    path.write_text(json.dumps(config))
    return path


def recorded_run(expected, seed=42, **changes):
    """(training, metadata, summary) as Part C records them for a run of `expected`."""
    settings, hyper = expected['training'], expected['model_hyperparameters']
    training = {'epochs': settings['epochs'], 'patience': settings['early_stopping_patience'],
                'optimizer': 'Adam', 'learning_rate': hyper['learning_rate'],
                'weight_decay': settings['weight_decay'], 'seed': seed, 'device': 'cuda'}
    metadata = {'model_hyperparameters': dict(hyper), 'batch_size': settings['batch_size'],
                'model_layers': settings['num_layers'], 'model_heads': settings['num_heads'],
                'max_seq_len': expected['data']['max_seq_len'], 'train_users': 10,
                'validation_users': 8, 'variant': 'cold_start', 'verified_files': {'f': 'h'},
                'cold_files': {name: 'c' for name in COLD_NAMES}}
    for key, value in changes.items():
        (training if key in training else metadata)[key] = value
    return training, metadata, {'status': 'PASS'}


COLD_NAMES = ['cold_start/cold_items.npy', 'cold_start/cold_sequences.parquet',
              'cold_start/cold_start_info.json']


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


class ConfigurationCheckTests(unittest.TestCase):
    def test_empty_settings_and_missing_grid_index_are_refused(self):
        with self.assertRaises(ValueError):
            expected_configuration(ROOT / 'configs/config.json', 0)  # repo template: nulls
        with tempfile.TemporaryDirectory() as directory:
            path = team_config(directory)
            with self.assertRaises(ValueError):
                expected_configuration(path)
            with self.assertRaises(ValueError):
                expected_configuration(path, 8)
            expected = expected_configuration(path, 5)
        self.assertEqual(expected['model_hyperparameters'],
                         {'hidden_size': 128, 'dropout': 0.2, 'learning_rate': 0.0005})

    def test_expanded_config_takes_no_grid_index(self):
        with tempfile.TemporaryDirectory() as directory:
            expanded = expected_configuration(team_config(directory), 0)
            path = Path(directory) / 'expanded.json'
            path.write_text(json.dumps(expanded))
            self.assertEqual(expected_configuration(path), expanded)
            with self.assertRaises(ValueError):
                expected_configuration(path, 0)

    def test_every_setting_and_the_seed_are_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = expected_configuration(team_config(directory), 0)
        training, metadata, _ = recorded_run(expected)
        self.assertEqual(configuration_mismatches(training, metadata, expected), [])
        changes = {'batch_size': 256, 'model_heads': 4, 'model_layers': 3, 'epochs': 50,
                   'patience': 3, 'weight_decay': 0.01, 'learning_rate': 0.0005,
                   'max_seq_len': 100, 'seed': 7}
        for key, value in changes.items():
            run = recorded_run(expected, **{key: value})
            found = configuration_mismatches(run[0], run[1], expected)
            self.assertEqual(len(found), 1, key)
        other = recorded_run(expected, model_hyperparameters={
            'hidden_size': 64, 'dropout': 0.5, 'learning_rate': 0.001})
        self.assertIn('model_hyperparameters', configuration_mismatches(*other[:2], expected)[0])

    def test_partial_reasons_cover_training_and_validation(self):
        self.assertEqual(partial_reasons({'train_users': 10, 'validation_users': 8}, 10, 8), [])
        reasons = partial_reasons({'train_users': 10, 'validation_users': 2}, 10, 8)
        self.assertEqual(len(reasons), 1)
        self.assertIn('validated on 2 of 8', reasons[0])
        self.assertEqual(len(partial_reasons({'train_users': 3, 'validation_users': 2}, 10, 8)), 2)


class ColdRunCheckTests(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as directory:
            self.expected = expected_configuration(team_config(directory), 0)
        self.cold_hashes = {name: 'c' for name in COLD_NAMES}

    def check(self, runs, allow_partial=False):
        return check_cold_runs([f'run{i}' for i in range(len(runs))], runs, {'f': 'h'},
                               self.cold_hashes, 10, 8, allow_partial, self.expected)

    def test_full_runs_on_the_team_configuration_pass(self):
        runs = [recorded_run(self.expected, seed) for seed in (42, 43)]
        self.assertEqual(self.check(runs), [42, 43])

    def test_partial_validation_needs_explicit_permission(self):
        runs = [recorded_run(self.expected, validation_users=2)]
        with self.assertRaisesRegex(ValueError, 'validated on 2 of 8'):
            self.check(runs)
        self.assertEqual(self.check(runs, allow_partial=True), [42])

    def test_off_configuration_runs_are_refused_even_when_partial_is_allowed(self):
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            self.check([recorded_run(self.expected, batch_size=64)], allow_partial=True)

    def test_duplicate_seeds_main_runs_and_other_data_are_refused(self):
        with self.assertRaises(ValueError):
            self.check([recorded_run(self.expected), recorded_run(self.expected)])
        with self.assertRaises(ValueError):
            self.check([recorded_run(self.expected, variant=None)])
        with self.assertRaises(ValueError):
            self.check([recorded_run(self.expected, verified_files={'f': 'other'})])


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
                        'model_layers': 1, 'model_heads': 2, 'max_seq_len': 5, 'batch_size': 2,
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

    def test_partial_validation_is_refused_unless_allowed(self):
        from src.evaluation import multi_seed

        config = json.loads((self.runs[0] / 'run_config.json').read_text())
        config['metadata']['validation_users'] = 2
        (self.runs[0] / 'run_config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'validated on 2 of 4'):
            multi_seed.check_runs(self.runs[:1], self.data, {'f': 'h'}, 'm', allow_partial=False)
        multi_seed.check_runs(self.runs[:1], self.data, {'f': 'h'}, 'm', allow_partial=True)

    def test_runs_must_match_the_team_configuration(self):
        from src.evaluation import multi_seed

        expected = {'data': {'max_seq_len': 5},
                    'model_hyperparameters': {'hidden_size': 8, 'dropout': 0.2,
                                              'learning_rate': 0.001},
                    'training': {'batch_size': 2, 'epochs': 2, 'num_layers': 1, 'num_heads': 2,
                                 'weight_decay': 0.0, 'early_stopping_patience': 5,
                                 'seeds': [42, 43]}}
        multi_seed.check_runs(self.runs, self.data, {'f': 'h'}, 'm', False, expected)
        for change in ({'batch_size': 4}, {'seeds': [42]}):
            off = {**expected, 'training': {**expected['training'], **change}}
            with self.assertRaisesRegex(ValueError, 'team configuration'):
                multi_seed.check_runs(self.runs, self.data, {'f': 'h'}, 'm', False, off)

    def test_train_refuses_user_limits_without_permission(self):
        from src.evaluation import multi_seed

        with self.assertRaisesRegex(ValueError, 'allow-partial-runs'):
            multi_seed.train_seeds(self.temp.name, [42], ['--limit-val-users', '100'])


if __name__ == '__main__':
    unittest.main()
