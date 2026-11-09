"""Live recommendation logic for the demo app (no notebooks, no hard-coded outputs).

Two parts:
  * build_artifacts(): turns the frozen processed dataset into small files the app can ship
    (item table, popularity counts, Markov transition counts, a few real example histories).
  * Recommender: scores the whole catalog for a user history picked in the interface and returns the top-k.

Today the scorers are the reference baselines (popularity, last-item Markov). A SASRec / Model 1 scorer plugs in
through `Recommender.register(name, fn)` where fn(history_item_ids) -> scores of length n_items + 1.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

TIE_BREAK = 1e-6
ITEM_COLUMNS = ['item_idx', 'title', 'brand', 'cat_l2', 'price_num']
ARTIFACT_FILES = ['items.parquet', 'popularity.npy', 'markov.npz', 'example_histories.parquet']


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_artifacts(processed_dir, out_dir, n_example_users=200, seed=42):
    processed, out = Path(processed_dir), Path(out_dir)
    missing = [f for f in ('items.parquet', 'sequences.parquet') if not (processed / f).exists()]
    if missing:
        raise FileNotFoundError(f'{processed} is missing {missing}; copy them from the shared Drive folder processed/')
    provenance = {'source_manifest_sha256': None, 'source_files': {}}
    if (processed / 'manifest.json').exists():
        manifest = json.loads((processed / 'manifest.json').read_text())
        for name in ('items.parquet', 'sequences.parquet'):
            actual = file_sha256(processed / name)
            if actual != manifest['processed_files'][name]['sha256']:
                raise ValueError(f'Frozen input checksum mismatch: {name}')
            provenance['source_files'][name] = actual
        provenance['source_manifest_sha256'] = file_sha256(processed / 'manifest.json')
    items = pd.read_parquet(processed / 'items.parquet')[ITEM_COLUMNS].sort_values('item_idx').reset_index(drop=True)
    seq = pd.read_parquet(processed / 'sequences.parquet').sort_values('user_idx').reset_index(drop=True)
    if not len(items) or not np.array_equal(items.item_idx, np.arange(1, len(items) + 1)):
        raise ValueError('Item IDs must cover 1..n_items exactly')
    out.mkdir(parents=True, exist_ok=True)                 # only after the inputs were read, so a failure leaves nothing behind
    n_items = int(items['item_idx'].max())
    pop = np.zeros(n_items + 1)
    rows, cols = [], []
    for train in seq['train_items']:
        train = [int(i) for i in train]
        for i in train:
            pop[i] += 1
        rows += train[:-1]
        cols += train[1:]
    transitions = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_items + 1, n_items + 1))
    items.to_parquet(out / 'items.parquet', index=False)
    np.save(out / 'popularity.npy', pop)
    sp.save_npz(out / 'markov.npz', transitions)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(seq), size=min(n_example_users, len(seq)), replace=False)
    examples = seq.iloc[np.sort(pick)][['user_idx', 'train_items']].copy()
    examples['train_items'] = examples['train_items'].map(lambda t: [int(i) for i in t])
    examples.to_parquet(out / 'example_histories.parquet', index=False)
    (out / 'artifact_manifest.json').write_text(json.dumps({
        'schema_version': 1, 'n_items': n_items, **provenance,
        'artifact_files': {name: file_sha256(out / name) for name in ARTIFACT_FILES},
    }, indent=2) + '\n')
    return {'n_items': n_items, 'n_examples': len(examples), 'transitions': int(transitions.nnz)}


class Recommender:
    def __init__(self, artifacts_dir):
        a = Path(artifacts_dir)
        self.items = pd.read_parquet(a / 'items.parquet').set_index('item_idx', drop=False)
        self.popularity = np.load(a / 'popularity.npy')
        self.markov = sp.load_npz(a / 'markov.npz').tocsr()
        self.examples = pd.read_parquet(a / 'example_histories.parquet')
        self.n_items = len(self.popularity) - 1
        pop = self.popularity / max(self.popularity.max(), 1.0)
        self._pop_scaled = pop
        self._scorers = {
            'popularity': lambda history: pop.copy(),
            'markov': self._markov_scores,
        }

    def register(self, name, fn):
        """Plug in another scorer (for example SASRec): fn(history_item_ids) -> array of n_items + 1 scores."""
        self._scorers[name] = fn

    @property
    def methods(self):
        return list(self._scorers)

    def _markov_scores(self, history):
        if not history:
            return self._pop_scaled.copy()                     # no history: fall back to popularity
        return self.markov[history[-1]].toarray().ravel() + TIE_BREAK * self._pop_scaled

    def search(self, query, limit=20):
        q = query.strip().lower()
        if not q:
            return self.items.iloc[0:0]
        hay = (self.items['title'].fillna('') + ' ' + self.items['brand'].fillna('')).str.lower()
        return self.items[hay.str.contains(q, regex=False)].head(limit)

    def recommend(self, history, method='markov', k=10):
        history = [int(i) for i in history]
        if any(i < 1 or i > self.n_items for i in history):
            raise ValueError('history contains an item id outside the catalog')
        if method not in self._scorers:
            raise ValueError(f'unknown method {method!r}; choose from {self.methods}')
        scores = np.asarray(self._scorers[method](history), dtype=np.float64).copy()
        if scores.shape != (self.n_items + 1,):
            raise ValueError('scorer must return n_items + 1 scores')
        scores[0] = -np.inf                                    # padding
        scores[history] = -np.inf                              # never recommend what the user already has
        k = min(k, self.n_items - len(set(history)))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind='stable')]
        out = self.items.loc[top, ['item_idx', 'title', 'brand', 'cat_l2', 'price_num']].copy()
        out.insert(0, 'rank', range(1, len(out) + 1))
        out['score'] = scores[top]
        return out.reset_index(drop=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Build the small artifacts the demo app ships with.')
    ap.add_argument('--processed', default='data/processed')
    ap.add_argument('--out', default='app/artifacts')
    args = ap.parse_args(argv)
    print(build_artifacts(args.processed, args.out))


if __name__ == '__main__':
    main()
