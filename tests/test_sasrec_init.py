import math
import unittest

import torch

from src.models.sasrec import SASRec


class InitTests(unittest.TestCase):
    def test_initial_loss_is_near_chance(self):
        torch.manual_seed(0)
        n_items = 1000
        model = SASRec(n_items, max_len=50, hidden_size=64, num_heads=1, num_layers=2)
        model.eval()
        tokens = torch.randint(1, n_items + 1, (32, 50))
        targets = torch.randint(1, n_items + 1, (32, 50))
        with torch.no_grad():
            logits, labels = model.training_logits(tokens, targets)
            loss = torch.nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(loss, math.log(n_items), delta=0.5)

    def test_padding_embedding_is_zero(self):
        model = SASRec(30, max_len=5, hidden_size=16, num_heads=4)
        self.assertEqual(torch.count_nonzero(model.item_embedding.weight[0]).item(), 0)
