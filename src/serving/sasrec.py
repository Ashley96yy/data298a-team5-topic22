"""Optional, inference-only SASRec adapter for the demo's register(name, fn) API."""
import json
from pathlib import Path

import numpy as np

from src.serving.recommend import ARTIFACT_FILES, file_sha256


class SASRecScorer:
    """Load a trusted checkpoint only after verifying its hash and frozen catalog.

    No LLM, training or test evaluation is performed. PyTorch is imported only
    when this optional adapter is used, keeping the baseline demo lightweight.
    """

    def __init__(self, checkpoint_path, artifacts_dir, expected_sha256, *, device='cpu'):
        import torch
        from src.models.sasrec import SASRec

        checkpoint_path, artifacts_dir = Path(checkpoint_path), Path(artifacts_dir)
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha256)):
            raise ValueError('Provide the independently verified checkpoint SHA-256')
        if file_sha256(checkpoint_path) != expected_sha256:
            raise ValueError('SASRec checkpoint checksum mismatch')
        manifest = json.loads((artifacts_dir / 'artifact_manifest.json').read_text())
        if manifest.get('schema_version') != 1 or not manifest.get('source_manifest_sha256'):
            raise ValueError('Rebuild demo artifacts from the frozen processed manifest')
        for name in ARTIFACT_FILES:
            if file_sha256(artifacts_dir / name) != manifest['artifact_files'].get(name):
                raise ValueError(f'Demo artifact checksum mismatch: {name}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        metadata = checkpoint['metadata']
        if (metadata.get('manifest_sha256') != manifest['source_manifest_sha256']
                or any(metadata.get('verified_files', {}).get(name) != value
                       for name, value in manifest['source_files'].items()
                       if name == 'sequences.parquet')):
            raise ValueError('Checkpoint and demo artifacts use different frozen data')
        if checkpoint.get('schema_version') != 1 or checkpoint.get('epoch', 0) < 1:
            raise ValueError('Expected a completed project best.pt checkpoint')
        if not np.isfinite(checkpoint.get('validation_NDCG@10', float('nan'))):
            raise ValueError('Checkpoint is missing its validation score')
        self.n_items, self.max_len = manifest['n_items'], metadata['max_seq_len']
        if type(self.n_items) is not int or self.n_items < 1 or type(self.max_len) is not int or self.max_len < 1:
            raise ValueError('Invalid catalog size or sequence length')
        hyper = metadata['model_hyperparameters']
        self.device = torch.device(device)
        self.model = SASRec(self.n_items, max_len=self.max_len, hidden_size=hyper['hidden_size'],
                            dropout=hyper['dropout'], num_heads=metadata['model_heads'],
                            num_layers=metadata['model_layers'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.to(self.device).eval()
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('Nonfinite checkpoint parameters')
        self.checkpoint_sha256 = expected_sha256
        self.best_epoch = checkpoint['epoch']

    def __call__(self, history):
        import torch

        if any(not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_))
               or not 1 <= i <= self.n_items for i in history):
            raise ValueError('History must contain integer item IDs from this catalog')
        history = list(history)[-self.max_len:]
        tokens = torch.zeros((1, self.max_len), dtype=torch.long, device=self.device)
        if history:
            tokens[0, -len(history):] = torch.tensor(history, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            scores = self.model.next_item_scores(tokens)[0]
        if scores.shape != (self.n_items + 1,) or not torch.isfinite(scores).all():
            raise ValueError('Invalid SASRec score vector')
        return scores.cpu().numpy()


def register_sasrec(recommender, checkpoint_path, expected_sha256, artifacts_dir, *, device='cpu'):
    """Register a scorer after all loading and provenance checks succeed."""
    scorer = SASRecScorer(checkpoint_path, artifacts_dir, expected_sha256, device=device)
    if scorer.n_items != recommender.n_items:
        raise ValueError('SASRec catalog size does not match the recommender')
    recommender.register('sasrec', scorer)
    return scorer
