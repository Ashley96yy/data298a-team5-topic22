import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.profile_file import write_profile_file, read_profile_file
from src.models.distill import DistillLog
from src.training.train_model1 import (build_targets, check_optimizer, make_schedule, synthetic_model1_smoke,
                                      targets_sha256, write_synthetic_profile)


class ScheduleTests(unittest.TestCase):
    def test_stage1_must_leave_room_for_stage2(self):
        self.assertEqual(make_schedule(0.6, 3, 10).selection_starts_at(), 4)
        for bad in (0, 10, 11, -1):
            with self.assertRaises(ValueError):
                make_schedule(0.6, bad, 10)


class OptimizerGuardTests(unittest.TestCase):
    def test_cli_optimizer_must_match_the_configured_one(self):
        check_optimizer({'optimizer': 'Adam'}, 'Adam')
        check_optimizer({}, 'AdamW')                                 # older configs do not declare an optimizer
        with self.assertRaises(ValueError):
            check_optimizer({'optimizer': 'Adam'}, 'AdamW')


class TargetTests(unittest.TestCase):
    def profile(self, n=30, d=12, seed=0):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p.npz'
            write_profile_file(path, np.arange(n), np.random.default_rng(seed).normal(size=(n, d)), 'llm_profile', True)
            return read_profile_file(path)

    def test_targets_have_the_hidden_size_and_cover_the_profile_users(self):
        targets, _ = build_targets(self.profile(), n_users=40, n_components=8)
        self.assertEqual(tuple(targets.values.shape), (40, 8))
        self.assertEqual(int(targets.valid.sum()), 30)                               # users 30..39 have no profile

    def test_subset_of_users_only_keeps_those_rows(self):
        targets, _ = build_targets(self.profile(), 40, 8, users=np.arange(10))
        self.assertEqual(int(targets.valid.sum()), 10)

    def test_profile_user_outside_the_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            build_targets(self.profile(n=30), n_users=20, n_components=8)

    def test_hash_is_stable_and_changes_with_the_values(self):
        a, _ = build_targets(self.profile(seed=0), 40, 8)
        b, _ = build_targets(self.profile(seed=0), 40, 8)
        c, _ = build_targets(self.profile(seed=1), 40, 8)
        self.assertEqual(targets_sha256(a), targets_sha256(b))
        self.assertNotEqual(targets_sha256(a), targets_sha256(c))


class SmokeTests(unittest.TestCase):
    def test_synthetic_smoke_runs_end_to_end_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            summary = synthetic_model1_smoke(out)
            self.assertGreaterEqual(summary['best_epoch'], 3)                        # stage 1 is epochs 1-2: never selected
            meta = json.loads((out / 'run_config.json').read_text())['metadata']
            self.assertEqual(meta['distill_schedule']['stage1_epochs'], 2)
            self.assertEqual(meta['profile_source'], 'synthetic')
            self.assertEqual(len(meta['reduced_targets_sha256']), 64)
            self.assertEqual(meta['reduction']['fit_on'], 'training users only')
            options = json.loads((out / 'run_config.json').read_text())['trainer_options']
            self.assertEqual(options['selection_start_epoch'], 3)
            log = pd.read_csv(out / 'distill_log.csv')
            self.assertEqual(log['alpha'].tolist()[:2], [0.6, 0.6])
            self.assertTrue((log['alpha'].iloc[2:] == 0.0).all())
            self.assertTrue((out / 'best.pt').exists())


class InterruptedRunTests(unittest.TestCase):
    """A failed or interrupted run must not lose the auxiliary (distillation) history of the epochs it finished."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.profile, self.first, self.second = root / 'profiles.npz', root / 'first', root / 'second'
        write_synthetic_profile(self.profile)

    def tearDown(self):
        self.tmp.cleanup()

    def fail_at(self, epoch):
        with self.assertRaises(Exception):
            synthetic_model1_smoke(self.first, fail_at_epoch=epoch, profile_path=self.profile)

    def test_a_failed_run_keeps_the_history_of_the_epochs_it_finished(self):
        self.fail_at(2)
        self.assertEqual(pd.read_csv(self.first / 'distill_log.csv')['epoch'].tolist(), [1, 2])   # epoch 2 trained, then validation failed
        self.assertEqual(torch.load(self.first / 'last.pt', weights_only=True)['epoch'], 1)     # the checkpoint is one epoch behind
        self.assertEqual(sorted(p.name for p in self.first.iterdir() if p.name.endswith('.tmp')), [])

    def test_a_resumed_run_has_the_complete_history_from_epoch_1(self):
        self.fail_at(2)
        first_log = pd.read_csv(self.first / 'distill_log.csv')
        summary = synthetic_model1_smoke(self.second, resume_from=self.first / 'last.pt', profile_path=self.profile)
        log = pd.read_csv(self.second / 'distill_log.csv')
        history = pd.read_csv(self.second / 'history.csv')
        self.assertEqual(log['epoch'].tolist(), [1, 2, 3, 4])
        self.assertEqual(log['epoch'].tolist(), history['epoch'].tolist())                       # same coverage as the trainer
        self.assertEqual(log['alpha'].tolist(), [0.6, 0.6, 0.0, 0.0])
        self.assertAlmostEqual(log.loc[0, 'recommendation_loss'], first_log.loc[0, 'recommendation_loss'], places=9)
        self.assertAlmostEqual(log.loc[0, 'distill_loss'], first_log.loc[0, 'distill_loss'], places=9)
        self.assertFalse((self.second / 'distill_log_resumed.csv').exists())
        self.assertGreaterEqual(summary['best_epoch'], 3)

    def test_the_epoch_that_will_be_trained_again_is_not_carried_over_twice(self):
        self.fail_at(3)                                                                          # checkpoint at epoch 2, log has 1, 2, 3
        self.assertEqual(pd.read_csv(self.first / 'distill_log.csv')['epoch'].tolist(), [1, 2, 3])
        synthetic_model1_smoke(self.second, resume_from=self.first / 'last.pt', profile_path=self.profile)
        self.assertEqual(pd.read_csv(self.second / 'distill_log.csv')['epoch'].tolist(), [1, 2, 3, 4])

    def test_resume_is_refused_without_the_previous_log(self):
        self.fail_at(2)
        (self.first / 'distill_log.csv').unlink()
        with self.assertRaises(ValueError):
            synthetic_model1_smoke(self.second, resume_from=self.first / 'last.pt', profile_path=self.profile)
        self.assertFalse((self.second / 'distill_log.csv').exists())

    def test_resume_is_refused_when_the_previous_log_does_not_cover_the_finished_epochs(self):
        self.fail_at(3)
        incomplete = DistillLog(self.first / 'distill_log.csv')
        incomplete.append({'epoch': 2, 'alpha': 0.6, 'recommendation_loss': 1.0, 'distill_loss': 1.0})   # epoch 1 missing
        with self.assertRaises(ValueError):
            synthetic_model1_smoke(self.second, resume_from=self.first / 'last.pt', profile_path=self.profile)


if __name__ == '__main__':
    unittest.main()
