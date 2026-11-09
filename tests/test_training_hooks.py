"""Part C hook/stage-selection regression tests on tiny synthetic sequences."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from src.data.sequence_dataset import make_dataloaders
from src.models.sasrec import SASRec
from src.training.trainer import TrainingSettings, fit, seed_everything, train_epoch


def fixture():
    seed_everything(42)
    model = SASRec(8, max_len=4, hidden_size=8, num_heads=2, num_layers=1, dropout=0.2)
    sequences = pd.DataFrame({'user_idx': [0, 1], 'train_items': [[1, 2, 3], [2, 4, 5]],
                              'val_item': [6, 7], 'test_item': [7, 8]})
    loader = make_dataloaders(sequences, max_len=4, batch_size=2, seed=42,
                              pin_memory=False)['train']
    return model, loader


def settings(**changes):
    return replace(TrainingSettings(epochs=6, patience=2, optimizer='Adam',
                                     learning_rate=0.001, weight_decay=0.0, seed=42), **changes)


def plain_epoch(epoch, model, loader, optimizer, device, grad_clip_norm):
    return train_epoch(model, loader, optimizer, device, grad_clip_norm)


class TrainingHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_custom_epoch_replaces_default_and_receives_absolute_epoch(self):
        model, loader = fixture()
        before = model.item_embedding.weight.detach().clone()
        seen = []

        def epoch_fn(epoch, *args):
            seen.append(epoch)
            return plain_epoch(epoch, *args)

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'run'
            with patch('src.training.trainer.train_epoch', side_effect=AssertionError('default called')):
                fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(epochs=3), out,
                    metadata={}, epoch_fn=epoch_fn)
            self.assertEqual(seen, [1, 2, 3])
            self.assertFalse(torch.equal(before, model.item_embedding.weight))
            run_config = json.loads((out / 'run_config.json').read_text())
            recorded = run_config['trainer_options']
            self.assertEqual(recorded['epoch_fn'], epoch_fn.__module__ + '.' + epoch_fn.__qualname__)
            self.assertEqual(recorded['selection_start_epoch'], 1)
            checkpoint = torch.load(out / 'last.pt', weights_only=True)
            self.assertEqual(checkpoint['trainer_options'], recorded)
            # Part F reconstructs this dataclass directly from run_config.training.
            self.assertEqual(TrainingSettings(**run_config['training']), settings(epochs=3))

    def test_stage1_neither_selects_checkpoint_nor_consumes_patience(self):
        model, loader = fixture()
        scores = iter([0.99, 0.98, 0.4, 0.4, 0.4, 0.9])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'run'

            def epoch_fn(epoch, *args):
                if epoch in (2, 3):
                    self.assertFalse((out / 'best.pt').exists())
                    last = torch.load(out / 'last.pt', weights_only=True)
                    self.assertEqual(last['bad_epochs'], 0)
                    self.assertEqual(last['best_epoch'], 0)
                    self.assertIsNone(last['best_model'])
                return plain_epoch(epoch, *args)

            summary = fit(model, loader, lambda _: {'NDCG@10': next(scores)}, settings(), out,
                          metadata={}, epoch_fn=epoch_fn, selection_start_epoch=3)
            self.assertEqual(summary['epochs_completed'], 5)
            self.assertEqual(summary['best_epoch'], 3)
            self.assertEqual(summary['best_validation_NDCG@10'], 0.4)
            self.assertTrue(summary['early_stopped'])
            history = pd.read_csv(out / 'history.csv')
            self.assertEqual(history['val_NDCG@10'].tolist(), [0.99, 0.98, 0.4, 0.4, 0.4])
            self.assertEqual(history['improved'].tolist(), [False, False, True, False, False])
            self.assertEqual(torch.load(out / 'best.pt', weights_only=True)['epoch'], 3)

    def test_default_and_plain_hook_have_identical_model0_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, kwargs in [('default', {}), ('explicit', {'epoch_fn': plain_epoch,
                                                              'selection_start_epoch': 1})]:
                model, loader = fixture()
                scores = iter([0.4, 0.3, 0.2])
                fit(model, loader, lambda _: {'NDCG@10': next(scores)}, settings(), root / name,
                    metadata={}, **kwargs)
            a, b = [torch.load(root / name / 'last.pt', weights_only=True)
                    for name in ('default', 'explicit')]
            self.assertEqual(a['epoch'], b['epoch'])
            self.assertEqual(a['bad_epochs'], b['bad_epochs'])
            self.assertEqual(a['best_epoch'], b['best_epoch'])
            self.assertEqual([r['train_loss'] for r in a['history']],
                             [r['train_loss'] for r in b['history']])
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
                self.assertTrue(torch.equal(a['best_model'][key], b['best_model'][key]), key)

    def test_resume_from_stage1_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fail_on_second = False
            seen = []

            def epoch_fn(epoch, *args):
                seen.append(epoch)
                if fail_on_second and epoch == 2:
                    raise RuntimeError('simulated interruption')
                return plain_epoch(epoch, *args)

            def run(name, resume=None):
                model, loader = fixture()
                return fit(model, loader, lambda _: {'NDCG@10': 0.4},
                           settings(epochs=4, lr_step_size=1, lr_gamma=0.5), root / name,
                           metadata={'schedule': {'stage1_epochs': 2}}, epoch_fn=epoch_fn,
                           selection_start_epoch=3, resume_from=resume)

            run('full')
            fail_on_second = True
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                run('first')
            self.assertFalse((root / 'first/best.pt').exists())
            fail_on_second = False
            seen.clear()
            summary = run('resumed', root / 'first/last.pt')
            self.assertEqual(seen, [2, 3, 4])
            self.assertEqual(summary['best_epoch'], 3)
            a, b = [torch.load(root / name / 'last.pt', weights_only=True)
                    for name in ('full', 'resumed')]
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertEqual([r['train_loss'] for r in a['history']],
                             [r['train_loss'] for r in b['history']])

    def test_resume_rejects_changed_hook_or_selection_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, loader = fixture()
            fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(epochs=3), root / 'first',
                metadata={}, epoch_fn=plain_epoch, selection_start_epoch=2)
            for name, kwargs in [('hook', {'epoch_fn': None, 'selection_start_epoch': 2}),
                                 ('selection', {'epoch_fn': plain_epoch, 'selection_start_epoch': 3})]:
                with self.subTest(name=name):
                    model, loader = fixture()
                    with self.assertRaisesRegex(ValueError, 'Checkpoint trainer_options mismatch'):
                        fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(epochs=4),
                            root / name, metadata={}, resume_from=root / 'first/last.pt', **kwargs)
                    self.assertFalse((root / name).exists())

    def test_old_model0_checkpoint_can_resume_with_default_hook_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, loader = fixture()
            fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(epochs=1), root / 'first',
                metadata={})
            legacy = torch.load(root / 'first/last.pt', weights_only=True)
            del legacy['trainer_options']
            torch.save(legacy, root / 'old.pt')
            model, loader = fixture()
            summary = fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(epochs=2),
                          root / 'resumed', metadata={}, resume_from=root / 'old.pt')
            self.assertEqual(summary['epochs_completed'], 2)
            self.assertEqual(summary['best_epoch'], 1)
            self.assertTrue((root / 'resumed/best.pt').exists())

    def test_invalid_hook_arguments_fail_before_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, value in enumerate([True, 0, -1, 1.5, 7]):
                with self.subTest(selection_start_epoch=value):
                    model, loader = fixture()
                    with self.assertRaisesRegex(ValueError, 'selection_start_epoch'):
                        fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(), root / str(i),
                            metadata={}, selection_start_epoch=value)
                    self.assertFalse((root / str(i)).exists())
            model, loader = fixture()
            with self.assertRaisesRegex(ValueError, 'epoch_fn'):
                fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(), root / 'invalid',
                    metadata={}, epoch_fn=42)
            self.assertFalse((root / 'invalid').exists())

    def test_invalid_hook_output_records_failure_without_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            for i, result in enumerate([(float('nan'), 4), (0.5, 0)]):
                with self.subTest(result=result):
                    model, loader = fixture()
                    out = Path(temporary) / str(i)
                    with self.assertRaisesRegex(ValueError, 'epoch_fn'):
                        fit(model, loader, lambda _: {'NDCG@10': 0.4}, settings(), out,
                            metadata={}, epoch_fn=lambda *args: result)
                    self.assertFalse((out / 'best.pt').exists())
                    self.assertEqual(json.loads((out / 'failure.json').read_text())['status'], 'FAILED')


if __name__ == '__main__':
    unittest.main()
