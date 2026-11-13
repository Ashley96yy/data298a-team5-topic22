"""No-LLM control profiles for Model 1: the centroid of each user's TRAINING items' text embeddings.

For every user the profile vector is the mean of the item-text embeddings of the user's last `window` training items
(the same 50-item window as the model input), L2-normalised. Only `train_items` is read: never the validation or test item.
The result is written in the profile-file format of src.data.profile_file with source 'item_text_centroid', so it goes through the
same distillation code as LLM profiles. This is the control arm that shows how much of any gain needs an LLM at all.

    python -m src.data.centroid_profiles --processed data/processed \
        --item-embeddings minilm/item_text_emb_minilm.npy --item-metadata minilm/metadata.json --output profiles_centroid_minilm.npz
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.profile_file import file_sha256, write_profile_file


def check_item_embeddings(embeddings, n_items):
    """Row 0 is padding (all zeros); rows 1..n_items are finite, non-zero item vectors."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != n_items + 1:
        raise ValueError(f'item embeddings must have shape ({n_items + 1}, d), got {embeddings.shape}')
    if not np.isfinite(embeddings).all():
        raise ValueError('item embeddings contain non-finite values')
    if np.any(embeddings[0] != 0):
        raise ValueError('row 0 (padding) of the item embeddings must be all zeros')
    if np.any(np.linalg.norm(embeddings[1:], axis=1) == 0):
        raise ValueError('some item embeddings are all zeros')
    return embeddings


def build_centroid_profiles(sequences, item_embeddings, window=50):
    """Return (user_idx, profiles) with one L2-normalised mean vector per user, from train_items only."""
    n_items = item_embeddings.shape[0] - 1
    user_idx, profiles = [], np.zeros((len(sequences), item_embeddings.shape[1]), dtype=np.float32)
    for row, (user, items) in enumerate(zip(sequences['user_idx'].to_numpy(), sequences['train_items'])):
        items = np.asarray(items, dtype=np.int64)[-window:]
        if len(items) == 0:
            raise ValueError(f'user {user} has no training items')
        if items.min() < 1 or items.max() > n_items:
            raise ValueError(f'user {user} has a training item outside 1..{n_items}')
        centroid = item_embeddings[items].mean(axis=0)
        norm = np.linalg.norm(centroid)
        profiles[row] = centroid / norm if norm > 0 else centroid
        user_idx.append(int(user))
    return np.asarray(user_idx, dtype=np.int64), profiles


def shuffle_profiles(profiles, seed):
    """Control: give every user the profile of ANOTHER user (no user keeps its own). Same vectors, wrong owners."""
    n = len(profiles)
    if n < 2:
        raise ValueError('need at least two users to shuffle profiles')
    order = np.random.default_rng(seed).permutation(n)
    shuffled = np.empty_like(profiles)
    shuffled[order] = profiles[np.roll(order, 1)]
    return shuffled


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', default='data/processed')
    ap.add_argument('--item-embeddings', required=True, help='item_text_emb_<encoder>.npy, shape (n_items + 1, d), row 0 = padding')
    ap.add_argument('--item-metadata', required=True, help='metadata.json written with the item embeddings')
    ap.add_argument('--output', required=True, help='profile file (.npz); a profiles_meta JSON is written next to it')
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--shuffle-seed', type=int, help='CONTROL: give each user another user\'s profile; the file is then labelled shuffled_control')
    args = ap.parse_args(argv)
    processed = Path(args.processed)
    try:
        manifest = file_sha256(processed / 'manifest.json')
        meta = json.loads(Path(args.item_metadata).read_text())
        if meta.get('manifest_sha256') != manifest:
            raise ValueError('the item embeddings were built from a different frozen dataset (manifest hash differs)')
        sequences = pd.read_parquet(processed / 'sequences.parquet', columns=['user_idx', 'train_items'])
        embeddings = check_item_embeddings(np.load(args.item_embeddings, allow_pickle=False), int(meta['n_items']))
        user_idx, profiles = build_centroid_profiles(sequences, embeddings, args.window)
        if args.shuffle_seed is not None:
            profiles = shuffle_profiles(profiles, args.shuffle_seed)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f'Profiles not written: {error}')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    source = 'item_text_centroid' if args.shuffle_seed is None else 'shuffled_control'
    write_profile_file(out, user_idx, profiles, source, True)
    out.with_suffix('.meta.json').write_text(json.dumps({
        'source': source, 'shuffle_seed': args.shuffle_seed, 'train_items_only': True, 'window': args.window, 'normalization': 'mean of unit item vectors, then L2',
        'n_users': int(len(user_idx)), 'dimension': int(profiles.shape[1]), 'item_encoder': meta.get('model_id'),
        'item_encoder_revision': meta.get('revision'), 'item_embedding_sha256': file_sha256(args.item_embeddings),
        'manifest_sha256': manifest, 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}, indent=2))
    print(f'wrote {out} ({len(user_idx)} users, {profiles.shape[1]} numbers each) and {out.with_suffix(".meta.json")}')


if __name__ == '__main__':
    main()
