"""Inference adapter and provenance guards; tiny synthetic data/checkpoints only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.models.sasrec import SASRec
from src.serving.recommend import Recommender, build_artifacts, file_sha256
from src.serving.sasrec import SASRecScorer, register_sasrec


class SASRecServingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.processed, self.artifacts = self.root / 'processed', self.root / 'artifacts'
        self.processed.mkdir()
        pd.DataFrame({'item_idx': np.arange(1, 9), 'title': [f'product {i}' for i in range(8)],
                      'brand': ['b'] * 8, 'cat_l2': ['face'] * 8,
                      'price_num': [1.] * 8}).to_parquet(self.processed / 'items.parquet')
        pd.DataFrame({'user_idx': [0, 1], 'train_items': [[1, 2, 3], [2, 4, 5]]}
                     ).to_parquet(self.processed / 'sequences.parquet')
        hashes = {n: file_sha256(self.processed / n) for n in ['items.parquet', 'sequences.parquet']}
        (self.processed / 'manifest.json').write_text(json.dumps({
            'processed_files': {n: {'sha256': value} for n, value in hashes.items()}}))
        build_artifacts(self.processed, self.artifacts)
        self.model = SASRec(8, max_len=5, hidden_size=8, num_heads=2, num_layers=1, dropout=0.2).eval()
        self.checkpoint = self.root / 'best.pt'
        self.payload = {'schema_version': 1, 'model': self.model.state_dict(), 'epoch': 2,
                        'validation_NDCG@10': 0.4, 'metadata': {
                            'manifest_sha256': file_sha256(self.processed / 'manifest.json'),
                            'verified_files': {'sequences.parquet': hashes['sequences.parquet']},
                            'max_seq_len': 5, 'model_layers': 1, 'model_heads': 2,
                            'model_hyperparameters': {'hidden_size': 8, 'dropout': 0.2}}}
        torch.save(self.payload, self.checkpoint)

    def tearDown(self):
        self.temp.cleanup()

    def scorer(self):
        return SASRecScorer(self.checkpoint, self.artifacts, file_sha256(self.checkpoint))

    def test_left_padding_truncation_and_scores_match_training_model(self):
        scorer = self.scorer()
        for history in [[], [1, 2], [1, 2, 3, 4, 5, 6, 7]]:
            with self.subTest(history=history):
                tokens = torch.zeros((1, 5), dtype=torch.long)
                if history:
                    tokens[0, -min(len(history), 5):] = torch.tensor(history[-5:])
                with torch.inference_mode():
                    expected = self.model.next_item_scores(tokens)[0].numpy()
                np.testing.assert_array_equal(scorer(history), expected)
        self.assertFalse(scorer.model.training)

    def test_registration_and_seen_item_masking(self):
        rec = Recommender(self.artifacts)
        register_sasrec(rec, self.checkpoint, file_sha256(self.checkpoint), self.artifacts)
        self.assertIn('sasrec', rec.methods)
        result = rec.recommend([1, 2, 3], 'sasrec', k=3)
        self.assertFalse(set(result.item_idx) & {0, 1, 2, 3})
        self.assertEqual(len(result), 3)

    def test_wrong_checkpoint_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'checksum'):
            SASRecScorer(self.checkpoint, self.artifacts, '0' * 64)

    def test_catalog_or_source_mismatch_is_rejected(self):
        self.payload['metadata']['manifest_sha256'] = 'wrong'
        torch.save(self.payload, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'different frozen data'):
            self.scorer()

    def test_changed_display_artifact_is_rejected(self):
        (self.artifacts / 'items.parquet').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Demo artifact checksum'):
            self.scorer()

    def test_missing_frozen_manifest_is_rejected(self):
        manifest = self.artifacts / 'artifact_manifest.json'
        record = json.loads(manifest.read_text())
        record['source_manifest_sha256'] = None
        manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'frozen processed manifest'):
            self.scorer()

    def test_invalid_history_ids_and_nonfinite_weights_are_rejected(self):
        scorer = self.scorer()
        for history in [[0], [9], [1.5], [True]]:
            with self.subTest(history=history), self.assertRaises(ValueError):
                scorer(history)
        self.payload['model']['item_embedding.weight'][1, 0] = float('nan')
        torch.save(self.payload, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.scorer()

    def test_builder_rejects_modified_frozen_input_before_writing_output(self):
        (self.processed / 'sequences.parquet').write_bytes(b'changed')
        out = self.root / 'never'
        with self.assertRaisesRegex(ValueError, 'Frozen input checksum'):
            build_artifacts(self.processed, out)
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
