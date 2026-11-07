"""Extra checks for src/models/sasrec.py: padding behaviour and train/eval consistency.

Complements tests/test_sasrec.py (finite gradients with padding, future tokens don't leak).
Run from the repository root (same command the team already uses):

    python -m unittest discover -s tests -p "test_sasrec*.py" -v
"""
import copy
import unittest

import torch

from src.models.sasrec import SASRec


class PaddingAndConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.n_items = 30
        self.model = SASRec(self.n_items, max_len=50, hidden_size=16, num_heads=4,
                            num_layers=2, dropout=0.2).eval()

    def _batch(self, lengths, length=12):
        """Left-padded inputs x and next-item targets y; n real inputs -> n real targets."""
        x = torch.zeros(len(lengths), length, dtype=torch.long)
        y = torch.zeros_like(x)
        for i, n in enumerate(lengths):
            items = torch.randint(1, self.n_items + 1, (n + 1,))
            x[i, length - n:] = items[:-1]
            y[i, length - n:] = items[1:]
        return x, y

    def test_left_padding_does_not_change_real_item_representations(self):
        # Zero the position table on a copy so a padded and an unpadded copy of the same
        # items are comparable; any remaining difference would be caused by padding.
        probe = copy.deepcopy(self.model).eval()
        probe.position_embedding.weight.data.zero_()
        real = torch.randint(1, self.n_items + 1, (3, 30))
        padded = torch.cat([torch.zeros(3, 20, dtype=torch.long), real], dim=1)
        with torch.no_grad():
            h_real = probe.hidden(real)
            h_padded = probe.hidden(padded)[:, 20:, :]
        self.assertTrue(torch.allclose(h_real, h_padded, atol=1e-4))

    def test_padded_positions_are_exactly_zero(self):
        x = torch.tensor([[0, 0, 0, 4, 5], [0, 1, 2, 3, 4]])
        with torch.no_grad():
            h = self.model.hidden(x)
        self.assertTrue(torch.equal(h[x == 0], torch.zeros_like(h[x == 0])))

    def test_all_padding_user_gives_finite_scores(self):
        x = torch.zeros(2, 10, dtype=torch.long)
        x[1, -3:] = torch.tensor([3, 4, 5])
        with torch.no_grad():
            scores = self.model.next_item_scores(x)
        self.assertEqual(scores.shape, (2, self.n_items + 1))
        self.assertTrue(torch.isfinite(scores).all())

    def test_next_item_scores_match_forward_last_position(self):
        x, _ = self._batch([12, 7, 3, 1])
        with torch.no_grad():
            a = self.model.next_item_scores(x)
            b = self.model(x)[:, -1]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_training_logits_match_forward_on_nonpadding_targets(self):
        x, y = self._batch([12, 7, 3, 5])
        keep = y != 0
        with torch.no_grad():
            logits, labels = self.model.training_logits(x, y)
            full = self.model(x)
        self.assertEqual(logits.shape, (int(keep.sum()), self.n_items + 1))
        self.assertTrue(torch.equal(labels, y[keep]))
        self.assertTrue(torch.allclose(logits, full[keep], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
