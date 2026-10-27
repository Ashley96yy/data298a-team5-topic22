"""Controlled Part C checks. No Drive access or Amazon experiment is performed."""
import ast
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

from src.data.sequence_dataset import (build_eval_arrays, build_train_arrays,
                                       make_dataloaders)
from src.models.sasrec import SASRec
from src.training.config import expand_grid, load_config
from src.training.train import resolve_config
from src.training.trainer import (TrainingSettings, checked_metrics, fit, load_best,
                                  make_shared_validator, seed_everything, train_epoch)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return pd.DataFrame({'user_idx': [0, 1, 2, 3],
                         'train_items': [[1, 2, 3], [2, 4, 5], [3, 5, 6], [1, 4, 6]],
                         'val_item': [7, 8, 7, 8], 'test_item': [9, 10, 10, 9]})


def model_and_loaders():
    seed_everything(42)
    model = SASRec(10, max_len=5, hidden_size=8, num_heads=2, num_layers=1, dropout=0.2)
    loaders = make_dataloaders(fixture(), max_len=5, batch_size=2,
                               eval_batch_size=2, seed=42, pin_memory=False)
    return model, loaders


def settings(**changes):
    return replace(TrainingSettings(epochs=3, patience=10, optimizer='Adam',
                                     learning_rate=0.001, weight_decay=0.0, seed=42), **changes)


class DataAdapterTests(unittest.TestCase):
    def test_adapter_arrays_match_teammate_notebook(self):
        notebook = json.loads((ROOT / 'src/data/sequence_dataloader.ipynb').read_text())
        names = {'left_pad', 'training_windows', 'build_train_arrays', 'build_eval_arrays'}
        definitions = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                definitions.extend(node for node in ast.parse(''.join(cell['source'])).body
                                   if isinstance(node, ast.FunctionDef) and node.name in names)
        self.assertEqual({n.name for n in definitions}, names)
        namespace = {'np': np}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'reviewed-builders', 'exec'), namespace)
        seq = fixture()
        seq.at[0, 'train_items'] = list(range(1, 14))
        for sliding in (False, True):
            for actual, expected in zip(build_train_arrays(seq, 5, sliding),
                                         namespace['build_train_arrays'](seq, 5, sliding)):
                np.testing.assert_array_equal(actual, expected)
        for split in ('val', 'test'):
            actual = build_eval_arrays(seq, split, 5)
            expected = namespace['build_eval_arrays'](seq, split, 5)
            for a, b in zip(actual[:3], expected[:3]):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(actual[3], expected[3]):
                np.testing.assert_array_equal(a, b)

    def test_batch_contract(self):
        _, loaders = model_and_loaders()
        batch = next(iter(loaders['train']))
        self.assertEqual(batch['input_seq'].shape, (2, 5))
        self.assertEqual(batch['input_seq'].dtype, torch.long)
        self.assertTrue(torch.equal(batch['input_seq'].eq(0), batch['target_seq'].eq(0)))
        val = next(iter(loaders['val']))
        self.assertEqual(val['target'].shape, (2,))
        self.assertEqual(val['history'].shape, (2, 3))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)

    def test_training_uses_nonpadding_targets_and_updates_weights(self):
        model, loaders = model_and_loaders()
        before = model.item_embedding.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss, count = train_epoch(model, loaders['train'], optimizer, 'cpu', 1.0)
        self.assertEqual(count, 8)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.item_embedding.weight))

    def test_best_checkpoint_and_early_stopping(self):
        model, loaders = model_and_loaders()
        scores = iter([0.4, 0.3, 0.2, 0.9])
        validate = Mock(side_effect=lambda _: {'NDCG@10': next(scores)})
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'run'
            summary = fit(model, loaders['train'], validate, settings(patience=2), out,
                           metadata={'fixture': True})
            self.assertEqual(summary['epochs_completed'], 3)
            self.assertEqual(summary['best_epoch'], 1)
            self.assertTrue(summary['early_stopped'])
            self.assertEqual(validate.call_count, 3)
            self.assertFalse(summary['test_evaluated'])
            best = load_best(model, out / 'best.pt')
            self.assertEqual(best['epoch'], 1)
            last = torch.load(out / 'last.pt', weights_only=True)
            self.assertTrue(all(torch.equal(model.state_dict()[k], v)
                                for k, v in last['best_model'].items()))
            self.assertEqual(len(pd.read_csv(out / 'history.csv')), 3)

    def test_epoch_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {'fixture': True}
            for optimizer in ('Adam', 'AdamW'):
                with self.subTest(optimizer=optimizer):
                    cfg = settings(epochs=2, optimizer=optimizer, lr_step_size=1, lr_gamma=0.5)
                    model, loaders = model_and_loaders()
                    full = root / (optimizer + '-full')
                    fit(model, loaders['train'], lambda _: {'NDCG@10': 0.5}, cfg,
                         full, metadata=metadata)
                    model, loaders = model_and_loaders()
                    first = root / (optimizer + '-first')
                    fit(model, loaders['train'], lambda _: {'NDCG@10': 0.5}, replace(cfg, epochs=1),
                         first, metadata=metadata)
                    model, loaders = model_and_loaders()
                    resumed = root / (optimizer + '-resumed')
                    fit(model, loaders['train'], lambda _: {'NDCG@10': 0.5}, cfg,
                         resumed, metadata=metadata, resume_from=first / 'last.pt')
                    a = torch.load(full / 'last.pt', weights_only=True)
                    b = torch.load(resumed / 'last.pt', weights_only=True)
                    for key in a['model']:
                        self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
                    self.assertEqual(a['scheduler'], b['scheduler'])
                    self.assertEqual([r['train_loss'] for r in a['history']],
                                     [r['train_loss'] for r in b['history']])
                    best = torch.load(resumed / 'best.pt', weights_only=True)
                    self.assertEqual(best['epoch'], 1)

    def test_min_delta_does_not_discard_the_highest_scoring_checkpoint(self):
        model, loaders = model_and_loaders()
        scores = iter([0.4, 0.41, 0.42])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'run'
            summary = fit(model, loaders['train'], lambda _: {'NDCG@10': next(scores)},
                           settings(patience=2, min_delta=0.05), out, metadata={})
            self.assertEqual(summary['best_epoch'], 3)
            self.assertEqual(summary['best_validation_NDCG@10'], 0.42)
            self.assertTrue(summary['early_stopped'])

    def test_nonfinite_training_logits_are_rejected(self):
        model, loaders = model_and_loaders()
        model.training_logits = lambda x, y: (torch.full((int(y.ne(0).sum()), 11), float('nan')),
                                               y[y.ne(0)])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with self.assertRaisesRegex(ValueError, 'logits'):
            train_epoch(model, loaders['train'], optimizer, 'cpu', 1.0)

    def test_empty_targets_cannot_produce_a_pass(self):
        model, _ = model_and_loaders()
        loader = [{'input_seq': torch.zeros((2, 5), dtype=torch.long),
                   'target_seq': torch.zeros((2, 5), dtype=torch.long)}]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        with self.assertRaisesRegex(ValueError, 'No nonpadding'):
            train_epoch(model, loader, optimizer, 'cpu', 1.0)

    def test_reject_invalid_metrics_and_records_failure(self):
        for value in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(value=value):
                model, loaders = model_and_loaders()
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary) / 'run'
                    with self.assertRaisesRegex(ValueError, 'metrics'):
                        fit(model, loaders['train'], lambda _: {'NDCG@10': value}, settings(),
                             out, metadata={})
                    self.assertEqual(json.loads((out / 'failure.json').read_text())['status'], 'FAILED')
                    self.assertFalse((out / 'best.pt').exists())

    def test_refuses_to_overwrite_a_run(self):
        model, loaders = model_and_loaders()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            evidence = out / 'existing.txt'
            evidence.write_text('keep')
            with self.assertRaises(FileExistsError):
                fit(model, loaders['train'], lambda _: {'NDCG@10': 0.5}, settings(), out, metadata={})
            self.assertEqual(evidence.read_text(), 'keep')

    def test_empty_existing_directory_is_also_rejected(self):
        model, loaders = model_and_loaders()
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                fit(model, loaders['train'], lambda _: {'NDCG@10': 0.5}, settings(),
                     temporary, metadata={})

    def test_resume_rejects_changed_provenance(self):
        model, loaders = model_and_loaders()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'first'
            fit(model, loaders['train'], lambda _: {'NDCG@10': 0.5}, settings(epochs=1), out,
                 metadata={'data_hash': 'original'})
            with self.assertRaisesRegex(ValueError, 'provenance'):
                fit(model, loaders['train'], lambda _: {'NDCG@10': 0.5}, settings(),
                     Path(temporary) / 'resume', metadata={'data_hash': 'changed'},
                     resume_from=out / 'last.pt')

    def test_shared_validator_is_validation_only_and_rejects_nan(self):
        class Evaluator:
            def evaluate(self, score_fn, data, **kwargs):
                self.kwargs = kwargs
                score_fn(np.array([[0, 0, 1, 2, 3]]), np.array([0]))
                return {'ranks': np.array([1.]), 'metrics': {'NDCG@10': 1., 'MRR': 1.}}
        evaluator = Evaluator()
        model, _ = model_and_loaders()
        validate = make_shared_validator(evaluator, {'n_items': 10})
        self.assertEqual(validate(model)['NDCG@10'], 1.)
        self.assertEqual(evaluator.kwargs['split'], 'val')
        self.assertFalse(evaluator.kwargs['sampled'])
        model.next_item_scores = lambda x: torch.full((len(x), 11), float('nan'))
        with self.assertRaisesRegex(ValueError, 'score matrix'):
            validate(model)

    def test_rejects_bad_settings_and_missing_metric(self):
        for changes in ({'epochs': 0}, {'patience': True}, {'optimizer': 'SGD'},
                         {'learning_rate': float('nan')}, {'lr_step_size': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                settings(**changes)
        with self.assertRaisesRegex(ValueError, 'NDCG@10'):
            checked_metrics({'MRR': 0.5})


class RunConfigTests(unittest.TestCase):
    def test_unconfirmed_team_config_cannot_run(self):
        with self.assertRaisesRegex(ValueError, 'Unconfirmed'):
            resolve_config(ROOT / 'configs/config.json', 0, 42)

    def test_resolves_one_grid_config_or_expanded_config(self):
        config = load_config(ROOT / 'configs/config.json')
        config['training'].update(batch_size=2, epochs=3, num_layers=1, num_heads=2,
                                  weight_decay=0.0, early_stopping_patience=2, seeds=[42])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps(config))
            actual = resolve_config(path, 3, 42)
            self.assertEqual(actual, expand_grid(config)[3])
            path.write_text(json.dumps(actual))
            self.assertEqual(resolve_config(path, None, 42), actual)
            with self.assertRaisesRegex(ValueError, 'seed'):
                resolve_config(path, None, 43)

    def test_cli_to_parquet_adapter_and_frozen_evaluator(self):
        """All input records here are invented; execute the real CLI/shared code."""
        from src.training.integration_check import REQUIRED
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / 'invented_processed'
            for folder in (processed / 'eval', processed / 'groups'):
                folder.mkdir(parents=True)
            seq = fixture()
            seq['n_train'] = seq.train_items.map(len)
            seq.to_parquet(processed / 'sequences.parquet', index=False)
            (processed / 'dataset_info.json').write_text(json.dumps({
                'dataset': 'Amazon Reviews 2023 - Beauty_and_Personal_Care',
                'subset': {'value': 'Skin Care'}, 'max_seq_len': 50, 'padding_idx': 0,
                'n_users': 4, 'n_items': 10, 'n_interactions': 20,
                'test_fixture_only': True,
            }))
            (processed / 'eval/eval_protocol.json').write_text(json.dumps({
                'main': 'full_catalog_ranking', 'mask_history': True, 'ks': [5, 10, 20],
            }))
            for split in ('val', 'test'):
                np.save(processed / f'eval/neg_{split}.npy', np.ones((4, 2), dtype=np.int64))
            for name in ('eval/eval_cases.parquet', 'groups/item_groups.parquet',
                         'groups/user_groups.parquet'):
                pd.DataFrame({'fixture_only': [True]}).to_parquet(processed / name, index=False)
            notebook = json.loads((ROOT / 'notebooks/data_prep/07_freeze_and_sanity_check.ipynb').read_text())
            source = None
            for cell in notebook['cells']:
                text = ''.join(cell.get('source', []))
                if cell['cell_type'] == 'code' and 'MODULE_SRC =' in text:
                    for node in ast.parse(text).body:
                        if isinstance(node, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == 'MODULE_SRC' for t in node.targets):
                            source = ast.literal_eval(node.value)
            self.assertIsNotNone(source)
            (processed / 'rec_eval.py').write_text(source)
            (processed / 'manifest.json').write_text(json.dumps({'processed_files': {
                name: {'sha256': hashlib.sha256((processed / name).read_bytes()).hexdigest()}
                for name in REQUIRED}}))
            config = load_config(ROOT / 'configs/config.json')
            config['data']['processed_dir'] = str(processed)
            config['training'].update(batch_size=2, epochs=1, num_layers=1, num_heads=2,
                                      weight_decay=0., early_stopping_patience=1, seeds=[42])
            config_path = root / 'fixture_config.json'
            config_path.write_text(json.dumps(config))
            output = root / 'training_output'
            run = subprocess.run([
                sys.executable, '-B', '-m', 'src.training.train', '--config', str(config_path),
                '--grid-index', '0', '--seed', '42', '--optimizer', 'Adam',
                '--allow-unapproved-model', '--allow-legacy-tie-metrics',
                '--output-dir', str(output),
            ], cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['epochs_completed'], 1)
            self.assertFalse(report['test_evaluated'])
            self.assertEqual(len(report['metadata']['verified_files']), 9)
            self.assertEqual(report['metadata']['selection_split'], 'val')
            self.assertTrue(report['metadata']['legacy_tie_policy_unresolved'])


if __name__ == '__main__':
    unittest.main()
