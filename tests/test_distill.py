import unittest

import numpy as np
import torch

from src.models.distill import (DistillSchedule, PCAReducer, ProfileTargets, combine_losses, distill_loss,
                                make_epoch_fn, pooled_representation, train_epoch_distill)
from src.models.sasrec import SASRec


def toy_loader(n_users=8, length=6, n_items=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = torch.randint(1, n_items + 1, (n_users, length + 1), generator=g)
    x, y = items[:, :-1].clone(), items[:, 1:].clone()
    x[:, :2] = 0                        # left padding, aligned in x and y
    y[:, :2] = 0
    batches = [{'user_idx': torch.arange(i, i + 4), 'input_seq': x[i:i + 4], 'target_seq': y[i:i + 4]}
               for i in range(0, n_users, 4)]
    return batches


class PoolingTests(unittest.TestCase):
    def test_padding_positions_are_excluded(self):
        hidden = torch.tensor([[[9.0, 9.0], [1.0, 3.0], [3.0, 5.0]]])
        tokens = torch.tensor([[0, 4, 7]])
        self.assertTrue(torch.allclose(pooled_representation(hidden, tokens), torch.tensor([[2.0, 4.0]])))

    def test_all_padding_row_is_zero_not_nan(self):
        out = pooled_representation(torch.ones(1, 3, 2), torch.zeros(1, 3, dtype=torch.long))
        self.assertTrue(torch.equal(out, torch.zeros(1, 2)))


class LossTests(unittest.TestCase):
    def test_distill_loss_uses_only_valid_rows(self):
        pooled = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
        target = torch.tensor([[0.0, 0.0], [100.0, 100.0]])
        valid = torch.tensor([True, False])
        self.assertAlmostEqual(distill_loss(pooled, target, valid).item(), 1.0)

    def test_distill_loss_with_no_valid_rows_is_zero(self):
        pooled = torch.ones(2, 3, requires_grad=True)
        loss = distill_loss(pooled, torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()                                   # graph must still be differentiable

    def test_alpha_zero_returns_model_loss_exactly(self):
        l_model, l_dist = torch.tensor(2.5, requires_grad=True), torch.tensor(0.3, requires_grad=True)
        self.assertTrue(combine_losses(l_model, l_dist, 0.0) is l_model)

    def test_beta_carries_no_gradient(self):
        l_model = torch.tensor(2.0, requires_grad=True)
        l_dist = torch.tensor(0.5, requires_grad=True)
        combine_losses(l_model, l_dist, 0.6).backward()
        beta = 2.0 / 0.5
        self.assertAlmostEqual(l_dist.grad.item(), 0.6 * beta, places=6)        # d/dl_dist = alpha * beta
        self.assertAlmostEqual(l_model.grad.item(), 1.0 - 0.6, places=6)        # beta is detached

    def test_beta_off_uses_unit_weight(self):
        l_model, l_dist = torch.tensor(2.0), torch.tensor(0.5)
        self.assertAlmostEqual(combine_losses(l_model, l_dist, 0.5, use_beta=False).item(), 0.5 * 0.5 + 0.5 * 2.0)

    def test_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            combine_losses(torch.tensor(1.0), torch.tensor(1.0), 1.5)


class ScheduleTests(unittest.TestCase):
    def test_stage_switch(self):
        s = DistillSchedule(alpha=0.6, stage1_epochs=3)
        self.assertEqual([s.alpha_for_epoch(e) for e in (1, 2, 3, 4, 5)], [0.6, 0.6, 0.6, 0.0, 0.0])
        self.assertEqual(s.selection_starts_at(), 4)


class TargetTests(unittest.TestCase):
    def test_lookup_and_validity(self):
        t = ProfileTargets.from_arrays(5, [1, 3], np.array([[1, 2], [3, 4]], dtype=np.float32))
        v, m = t.lookup(torch.tensor([3, 0, 1]))
        self.assertEqual(m.tolist(), [True, False, True])
        self.assertEqual(v[0].tolist(), [3.0, 4.0])

    def test_rejects_duplicates_nonfinite_and_out_of_range(self):
        with self.assertRaises(ValueError):
            ProfileTargets.from_arrays(5, [1, 1], np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            ProfileTargets.from_arrays(5, [1], np.array([[np.nan, 0.0]]))
        with self.assertRaises(ValueError):
            ProfileTargets.from_arrays(5, [7], np.zeros((1, 2)))

    def test_pca_is_deterministic_standardised_and_train_fit_only(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 10)) * np.arange(1, 11)
        a, b = PCAReducer(4).fit(X).transform(X), PCAReducer(4).fit(X).transform(X)
        self.assertTrue(np.allclose(a, b))
        self.assertTrue(np.allclose(a.std(axis=0), 1.0, atol=1e-3))
        new = PCAReducer(4).fit(X).transform(rng.normal(size=(5, 10)))     # transform needs no refit
        self.assertEqual(new.shape, (5, 4))

    def test_pca_rejects_nonfinite_input(self):
        X = np.random.default_rng(0).normal(size=(20, 5))
        with self.assertRaises(ValueError):
            PCAReducer(2).fit(np.where(np.arange(100).reshape(20, 5) == 7, np.nan, X))
        reducer = PCAReducer(2).fit(X)
        with self.assertRaises(ValueError):
            reducer.transform(np.full((2, 5), np.inf))

    def test_pca_rejects_too_many_components(self):
        with self.assertRaises(ValueError):
            PCAReducer(8).fit(np.random.default_rng(0).normal(size=(50, 5)))


class EpochTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SASRec(20, max_len=6, hidden_size=16, num_heads=1, num_layers=1, dropout=0.0)
        self.loader = toy_loader()
        self.targets = ProfileTargets.from_arrays(8, list(range(8)), torch.randn(8, 16).numpy())

    def test_alpha_zero_matches_plain_recommendation_step(self):
        import copy
        plain, dist = copy.deepcopy(self.model), copy.deepcopy(self.model)
        opt_a = torch.optim.SGD(plain.parameters(), lr=0.1)
        opt_b = torch.optim.SGD(dist.parameters(), lr=0.1)
        for batch in self.loader:
            x, y = batch['input_seq'], batch['target_seq']
            opt_a.zero_grad()
            logits, labels = plain.training_logits(x, y)
            torch.nn.functional.cross_entropy(logits, labels).backward()
            torch.nn.utils.clip_grad_norm_(plain.parameters(), 1.0)
            opt_a.step()
        train_epoch_distill(dist, self.loader, opt_b, 'cpu', 1.0, self.targets, alpha=0.0)
        for pa, pb in zip(plain.parameters(), dist.parameters()):
            self.assertTrue(torch.allclose(pa, pb, atol=1e-6))

    def test_distillation_moves_pooled_representation_toward_target(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=0.01)
        first = last = None
        for epoch in range(60):
            _, _, d = train_epoch_distill(self.model, self.loader, opt, 'cpu', 1.0, self.targets, alpha=0.9)
            first = d if first is None else first
            last = d
        self.assertLess(last, first)

    def test_same_seed_same_losses(self):
        import copy
        runs = []
        for _ in range(2):
            torch.manual_seed(1)
            m = copy.deepcopy(self.model)
            opt = torch.optim.SGD(m.parameters(), lr=0.1)
            runs.append(train_epoch_distill(m, self.loader, opt, 'cpu', 1.0, self.targets, alpha=0.5))
        self.assertEqual(runs[0], runs[1])

    def test_rejects_misaligned_padding(self):
        bad = [{'user_idx': torch.arange(2), 'input_seq': torch.tensor([[0, 1], [2, 3]]),
                'target_seq': torch.tensor([[1, 0], [3, 4]])}]
        with self.assertRaises(ValueError):
            train_epoch_distill(self.model, bad, torch.optim.SGD(self.model.parameters(), lr=0.1), 'cpu', 1.0,
                                self.targets, alpha=0.5)

    def test_epoch_fn_follows_the_schedule(self):
        schedule = DistillSchedule(alpha=0.6, stage1_epochs=1)
        fn = make_epoch_fn(self.targets, schedule)
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss1, n1 = fn(1, self.model, self.loader, opt, 'cpu', 1.0)       # stage 1: distillation on
        loss2, n2 = fn(2, self.model, self.loader, opt, 'cpu', 1.0)       # stage 2: recommendation loss only
        self.assertEqual(n1, n2)
        self.assertTrue(isinstance(loss1, float) and isinstance(loss2, float))


if __name__ == '__main__':
    unittest.main()
