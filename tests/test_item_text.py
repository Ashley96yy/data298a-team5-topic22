"""Item alignment, frozen inputs, pooling, fusion and checkpoint integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from src.data.item_embeddings import (read_items, pool_tokens, encode_items, save_embeddings,
                                       load_embeddings, digest, HFEncoder)
from src.data.sequence_dataset import make_dataloaders
from src.models.sasrec import SASRec
from src.models.text_sasrec import TextSASRec, load_text_checkpoint
from src.models.distill import DistillSchedule, ProfileTargets, make_epoch_fn
from src.training.trainer import TrainingSettings, fit, seed_everything


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.items = pd.DataFrame({'item_idx': [3, 1, 2], 'item_text': ['third', 'first', 'second']})
        self.write_fixture()

    def tearDown(self):
        self.tmp.cleanup()

    def write_fixture(self):
        self.items.to_parquet(self.root / 'item_text.parquet')
        (self.root / 'dataset_info.json').write_text(json.dumps({'n_items': 3}))
        (self.root / 'manifest.json').write_text(json.dumps({'processed_files': {
            f: {'sha256': digest(self.root / f)} for f in ['item_text.parquet', 'dataset_info.json']}}))

    def test_frozen_input_sorted_and_encoded_by_id_not_file_position(self):
        items, meta = read_items(self.root)
        def encode(texts):
            return np.array([[{'first': 1, 'second': 2, 'third': 3}[t], 1] for t in texts])
        matrix, count = encode_items(items, 3, encode, 2, batch_size=2)
        self.assertEqual(count, 3)
        np.testing.assert_array_equal(matrix[:, 0], [0, 1, 2, 3])
        output = self.root / 'out'
        save_embeddings(output, 'minilm', matrix, {**meta, 'complete_catalog': True})
        loaded, _ = load_embeddings(output, manifest_sha256=meta['manifest_sha256'], n_items=3)
        np.testing.assert_array_equal(loaded, matrix)
        with self.assertRaises(FileExistsError):
            save_embeddings(output, 'minilm', matrix, meta)

    def test_tampered_frozen_input_rejected(self):
        self.items.to_parquet(self.root / 'item_text.parquet', compression=None)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            read_items(self.root)

    def test_duplicate_missing_and_blank_items_rejected(self):
        for bad in [pd.DataFrame({'item_idx': [1, 1, 3], 'item_text': ['a', 'b', 'c']}),
                    pd.DataFrame({'item_idx': [1, 2], 'item_text': ['a', 'b']}),
                    pd.DataFrame({'item_idx': [1, 2, 3], 'item_text': ['a', ' ', 'c']})]:
            self.items = bad
            self.write_fixture()
            with self.assertRaises(ValueError):
                read_items(self.root)

    def test_mean_pooling_excludes_padding_and_cls_is_separate(self):
        hidden = torch.tensor([[[1., 0.], [0., 1.], [99., 99.]]])
        mask = torch.tensor([[1, 1, 0]])
        torch.testing.assert_close(pool_tokens(hidden, mask, 'mean'), torch.tensor([[2**-0.5, 2**-0.5]]))
        torch.testing.assert_close(pool_tokens(hidden, mask, 'cls'), torch.tensor([[1., 0.]]))

    def test_partial_embeddings_and_wrong_dataset_rejected(self):
        items, meta = read_items(self.root)
        matrix, _ = encode_items(items, 3, lambda t: np.ones((len(t), 2)), 2, limit=1)
        out = self.root / 'partial'
        save_embeddings(out, 'minilm', matrix, {**meta, 'complete_catalog': False})
        with self.assertRaisesRegex(ValueError, 'complete'):
            load_embeddings(out, manifest_sha256=meta['manifest_sha256'], n_items=3)
        save_embeddings(self.root / 'wrong', 'minilm', np.array([[0,0],[1,1],[1,1],[1,1]], dtype='float32'),
                        {**meta, 'complete_catalog': True})
        with self.assertRaisesRegex(ValueError, 'different frozen'):
            load_embeddings(self.root / 'wrong', manifest_sha256='wrong', n_items=3)

    def test_tampered_embedding_rejected(self):
        _, meta = read_items(self.root)
        out = self.root / 'out'
        path = save_embeddings(out, 'blair', np.ones((4, 2), dtype='float32'), {**meta, 'complete_catalog': True})
        path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_embeddings(out, manifest_sha256=meta['manifest_sha256'], n_items=3)

    def test_invalid_encoder_vectors_rejected(self):
        items, _ = read_items(self.root)
        for encoder in [lambda t: np.zeros((len(t), 2)), lambda t: np.full((len(t), 2), np.nan),
                        lambda t: np.ones((len(t), 3))]:
            with self.assertRaises(ValueError):
                encode_items(items, 3, encoder, 2)

    def test_unpinned_revision_rejected_before_network(self):
        with self.assertRaisesRegex(ValueError, 'immutable'):
            HFEncoder('example/model', 'main', 'minilm', 'cpu', None)


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(42)
        self.text = np.random.default_rng(42).normal(size=(9, 6)).astype('float32')
        self.text[0] = 0
        self.kw = dict(max_len=5, hidden_size=8, num_heads=2, num_layers=1, dropout=0.)
        self.model = TextSASRec(8, self.text, **self.kw).eval()

    def test_zero_projection_starts_at_model0_and_then_learns(self):
        seed_everything(42)
        original = SASRec(8, **self.kw).eval()
        tokens = torch.tensor([[0, 0, 1, 2, 3]])
        torch.testing.assert_close(self.model(tokens), original(tokens), rtol=0, atol=0)
        frozen = self.model.item_embedding.text.clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss = torch.nn.functional.cross_entropy(self.model.next_item_scores(tokens), torch.tensor([4]))
        loss.backward()
        self.assertGreater(self.model.item_embedding.projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        torch.testing.assert_close(self.model.item_embedding.text, frozen, rtol=0, atol=0)
        self.assertIsNone(self.model.item_embedding.text.grad)
        self.assertFalse(torch.equal(self.model(tokens), original(tokens)))
        self.assertEqual(self.model.item_embedding.weight[0].abs().sum(), 0)

    def test_causal_padding_and_all_padding_remain_safe_with_text(self):
        with torch.no_grad():
            self.model.item_embedding.projection.weight.fill_(0.05)
        a = torch.tensor([[0, 0, 1, 2, 3]])
        b = torch.tensor([[0, 0, 1, 2, 8]])
        torch.testing.assert_close(self.model.hidden(a)[:, :4], self.model.hidden(b)[:, :4])
        self.assertEqual(self.model.hidden(a)[:, :2].abs().sum(), 0)
        empty = self.model(torch.zeros((1, 5), dtype=torch.long))
        self.assertTrue(torch.isfinite(empty).all())
        self.assertEqual(empty.abs().sum(), 0)
        logits, targets = self.model.training_logits(a, torch.tensor([[0, 0, 2, 3, 4]]))
        torch.testing.assert_close(logits, self.model(a)[torch.tensor([[False,False,True,True,True]])])
        self.assertEqual(targets.tolist(), [2, 3, 4])

    def test_invalid_text_rejected(self):
        for text in [np.ones((9, 6)), np.zeros((8, 6)), np.full((9, 6), np.nan)]:
            with self.assertRaises(ValueError):
                TextSASRec(8, text, **self.kw)

    def test_trainer_distillation_and_checkpoint_roundtrip(self):
        sequences = pd.DataFrame({'user_idx': [0,1], 'train_items': [[1,2,3],[2,4,5]],
                                  'val_item':[6,7], 'test_item':[7,8]})
        loader = make_dataloaders(sequences, max_len=5, batch_size=2, seed=42, pin_memory=False)['train']
        targets = ProfileTargets.from_arrays(2, [0,1], np.ones((2,8), dtype='float32'))
        schedule = DistillSchedule(alpha=0.6, stage1_epochs=1)
        scores = iter([0.99, 0.2, 0.3])
        metadata = {'model_class':'TextSASRec','text_architecture':self.model.architecture,
                    'purpose':'synthetic_smoke_only_not_Amazon_results'}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            settings = TrainingSettings(epochs=3, patience=2, optimizer='Adam', learning_rate=0.001,
                                        weight_decay=0., seed=42)
            summary = fit(self.model, loader, lambda _: {'NDCG@10': next(scores)}, settings, out,
                          metadata=metadata, epoch_fn=make_epoch_fn(targets,schedule), selection_start_epoch=2)
            self.assertEqual(summary['best_epoch'], 3)
            restored, checkpoint = load_text_checkpoint(out/'best.pt')
            self.model.eval()
            tokens = torch.tensor([[0,0,1,2,3]])
            torch.testing.assert_close(restored.next_item_scores(tokens), self.model.next_item_scores(tokens), rtol=0, atol=0)
            self.assertEqual(checkpoint['epoch'],3)
            self.assertTrue(torch.isfinite(restored.next_item_scores(tokens)).all())


    def test_item_only_entrypoint_checks_sources_and_uses_validation(self):
        from src.training.train_item_text import run
        sequences = pd.DataFrame({'user_idx':[0,1], 'train_items':[[1,2,3],[2,4,5]],
                                  'val_item':[6,7], 'test_item':[7,8]})
        data = {'n_items':8, 'max_seq_len':50, 'sequences':sequences}
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = json.loads((root/'configs/config.json').read_text())
            config['data']['processed_dir'] = str(tmp)
            (tmp/'config.json').write_text(json.dumps(config))
            (tmp/'dataset_info.json').write_text('{}')
            (tmp/'item_text.parquet').write_bytes(b'synthetic source fixture')
            hashes = {n:digest(tmp/n) for n in ['dataset_info.json','item_text.parquet']}
            (tmp/'manifest.json').write_text(json.dumps({'processed_files':{n:{'sha256':h} for n,h in hashes.items()}}))
            meta = {'n_items':8,'manifest_sha256':digest(tmp/'manifest.json'),
                    'verified_files':hashes,'complete_catalog':True}
            save_embeddings(tmp/'vectors','minilm',self.text,meta)
            with patch('src.training.train_item_text.load_verified_data',return_value=(object(),data,hashes)), \
                 patch('src.training.train_item_text.make_shared_validator',return_value=lambda m:{'NDCG@10':0.2}) as validator:
                summary = run(str(tmp/'config.json'),7,42,tmp/'vectors',tmp/'run',smoke=True)
                self.assertFalse(summary['test_evaluated'])
                self.assertEqual(summary['metadata']['selection_split'],'val')
                self.assertEqual(summary['epochs_completed'],2)
                self.assertEqual(summary['metadata']['purpose'],'item_text_smoke_only')
                self.assertEqual(validator.call_args.kwargs['users'].tolist(),[0,1])
                model,_ = load_text_checkpoint(tmp/'run/best.pt')
                self.assertTrue(torch.isfinite(model.next_item_scores(torch.tensor([[0,0,1,2,3]]))).all())
                (tmp/'item_text.parquet').write_bytes(b'changed source')
                with self.assertRaisesRegex(ValueError,'Embedding source differs'):
                    run(str(tmp/'config.json'),7,42,tmp/'vectors',tmp/'rejected',smoke=True)
                self.assertFalse((tmp/'rejected').exists())


if __name__ == '__main__':
    unittest.main()
