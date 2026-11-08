"""File format for per-user profile embeddings (the distillation targets' raw material).

A profile file is a NumPy .npz written with write_profile_file():
    user_idx          int64 (n,)     unique user ids, 0 <= user_idx < n_users
    embeddings        float (n, d)   one text embedding per user, finite
    source            str            'llm_profile' | 'item_text_centroid' | 'synthetic'
    train_items_only  bool           the producer attests that every profile was built from TRAINING items only
                                     (no validation or test item, no review written after the user's cut-off)
Users without a row simply have no distillation term (subset ablation). NumPy only, so it is easy to produce on Colab.
"""
import hashlib
from pathlib import Path

import numpy as np

SOURCES = ('llm_profile', 'item_text_centroid', 'synthetic')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_profile_file(path, user_idx, embeddings, source, train_items_only):
    user_idx = np.asarray(user_idx, dtype=np.int64)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.savez(path, user_idx=user_idx, embeddings=embeddings, source=np.array(source),
             train_items_only=np.array(bool(train_items_only)))


def read_profile_file(path, *, allow_synthetic=False):
    """Load and validate a profile file; raises ValueError with a specific reason instead of returning bad data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'profile file not found: {path}')
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in ('user_idx', 'embeddings', 'source', 'train_items_only') if k not in z.files]
        if missing:
            raise ValueError(f'profile file is missing {missing}')
        user_idx, embeddings = z['user_idx'], z['embeddings']
        source, train_only = str(z['source']), bool(z['train_items_only'])
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}, got {source!r}')
    if source == 'synthetic' and not allow_synthetic:
        raise ValueError('synthetic profiles are for smoke tests only and cannot be used for a real run')
    if not train_only:
        raise ValueError('profiles must be built from training items only (train_items_only is not True)')
    if user_idx.ndim != 1 or not np.issubdtype(user_idx.dtype, np.integer):
        raise ValueError('user_idx must be a 1-D integer array')
    if embeddings.ndim != 2 or len(embeddings) != len(user_idx):
        raise ValueError('embeddings must be (n, d) with one row per user_idx')
    if len(user_idx) == 0:
        raise ValueError('profile file has no rows')
    if user_idx.min() < 0 or len(np.unique(user_idx)) != len(user_idx):
        raise ValueError('user_idx must be unique and non-negative')
    if not np.isfinite(embeddings).all():
        raise ValueError('embeddings contain non-finite values')
    return {'user_idx': user_idx.astype(np.int64), 'embeddings': embeddings.astype(np.float64), 'source': source,
            'sha256': file_sha256(path)}
