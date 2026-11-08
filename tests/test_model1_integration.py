"""Model 1 through the real trainer: distillation epoch hook + stage-2 checkpoint selection.

Needs the trainer hook (fit(..., epoch_fn=..., selection_start_epoch=...)); skipped until that is on the branch.
"""
import inspect
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from src.data.sequence_dataset import make_dataloaders
from src.models.distill import DistillSchedule, ProfileTargets, make_epoch_fn
from src.models.sasrec import SASRec
from src.training.trainer import TrainingSettings, fit, seed_everything

HAS_HOOK = 'epoch_fn' in inspect.signature(fit).parameters and 'selection_start_epoch' in inspect.signature(fit).parameters


def fixture():
    seed_everything(42)
    model = SASRec(8, max_len=4, hidden_size=8, num_heads=2, num_layers=1, dropout=0.2)
    sequences = pd.DataFrame({'user_idx': [0, 1], 'train_items': [[1, 2, 3], [2, 4, 5]],
                              'val_item': [6, 7], 'test_item': [7, 8]})
    loader = make_dataloaders(sequences, max_len=4, batch_size=2, seed=42, pin_memory=False)['train']
    targets = ProfileTargets.from_arrays(2, [0, 1], torch.randn(2, 8).numpy())
    return model, loader, targets


def settings(**changes):
    return replace(TrainingSettings(epochs=5, patience=2, optimizer='Adam', learning_rate=0.001,
                                    weight_decay=0.0, seed=42), **changes)


@unittest.skipUnless(HAS_HOOK, 'needs fit(..., epoch_fn, selection_start_epoch)')
class Model1IntegrationTests(unittest.TestCase):
    def test_stage1_is_distilled_but_never_selected_and_stage2_selects(self):
        model, loader, targets = fixture()
        schedule = DistillSchedule(alpha=0.6, stage1_epochs=2)
        hook = make_epoch_fn(targets, schedule)
        scores = iter([0.99, 0.98, 0.2, 0.3, 0.4])           # the best scores come in stage 1, and must be ignored
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'run'
            metadata = {'distill_schedule': {'alpha': 0.6, 'stage1_epochs': 2, 'use_beta': True}}
            summary = fit(model, loader, lambda _: {'NDCG@10': next(scores)}, settings(), out, metadata=metadata,
                          epoch_fn=hook, selection_start_epoch=schedule.selection_starts_at())
            history = pd.read_csv(out / 'history.csv')
            self.assertEqual(history['improved'].tolist(), [False, False, True, True, True])
            self.assertEqual(summary['best_epoch'], 5)
            self.assertAlmostEqual(summary['best_validation_NDCG@10'], 0.4)
            self.assertEqual(torch.load(out / 'best.pt', weights_only=True)['epoch'], 5)
            options = json.loads((out / 'run_config.json').read_text())['trainer_options']
            self.assertEqual(options['selection_start_epoch'], 3)
            self.assertIn('make_epoch_fn', options['epoch_fn'])
        self.assertEqual([row['alpha'] for row in hook.log], [0.6, 0.6, 0.0, 0.0, 0.0])
        self.assertTrue(all(row['distill_loss'] > 0 for row in hook.log[:2]))
        self.assertTrue(all(row['distill_loss'] == 0.0 for row in hook.log[2:]))

    def test_stage1_longer_than_the_epoch_budget_is_rejected(self):
        model, loader, targets = fixture()
        schedule = DistillSchedule(alpha=0.6, stage1_epochs=5)       # selection would start at epoch 6 of 5
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                fit(model, loader, lambda _: {'NDCG@10': 0.3}, settings(), Path(temporary) / 'run', metadata={},
                    epoch_fn=make_epoch_fn(targets, schedule), selection_start_epoch=schedule.selection_starts_at())

    def test_recommendation_loss_is_what_the_trainer_logs(self):
        model, loader, targets = fixture()
        hook = make_epoch_fn(targets, DistillSchedule(alpha=0.6, stage1_epochs=1))
        scores = iter([0.1, 0.2, 0.3])
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'run'
            fit(model, loader, lambda _: {'NDCG@10': next(scores)}, settings(epochs=3), out, metadata={},
                epoch_fn=hook, selection_start_epoch=2)
            logged = pd.read_csv(out / 'history.csv')['train_loss'].tolist()
        self.assertEqual(len(logged), len(hook.log))
        for seen, row in zip(logged, hook.log):
            self.assertAlmostEqual(seen, row['recommendation_loss'], places=6)


if __name__ == '__main__':
    unittest.main()
