import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.profile_file import write_profile_file, read_profile_file
from src.training.train_model1 import (build_targets, make_schedule, synthetic_model1_smoke, targets_sha256)


class ScheduleTests(unittest.TestCase):
    def test_stage1_must_leave_room_for_stage2(self):
        self.assertEqual(make_schedule(0.6, 3, 10).selection_starts_at(), 4)
        for bad in (0, 10, 11, -1):
            with self.assertRaises(ValueError):
                make_schedule(0.6, bad, 10)


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


if __name__ == '__main__':
    unittest.main()
