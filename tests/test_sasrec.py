import unittest
import torch
from src.models.sasrec import SASRec

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model = SASRec(30, max_len=5, hidden_size=16, num_heads=4, dropout=0.2)

    def test_padding_has_finite_forward_and_gradients(self):
        x = torch.tensor([[0,0,0,1,2], [0,0,3,4,5]])
        targets = torch.tensor([[0,0,0,2,3], [0,0,4,5,6]])
        logits, labels = self.model.training_logits(x, targets)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.model.eval()
        with torch.no_grad():
            scores = self.model.next_item_scores(x)
            self.assertEqual(scores.shape, (2,31))
            self.assertTrue(torch.isfinite(scores).all())

    def test_future_tokens_do_not_change_past_representation(self):
        self.model.eval()
        a = torch.tensor([[0,0,1,2,3]])
        b = torch.tensor([[0,0,1,4,5]])
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.model.hidden(a)[:,2], self.model.hidden(b)[:,2], atol=1e-6))

if __name__ == '__main__':
    unittest.main()
