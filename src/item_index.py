"""Item retrieval index for the prototype: exact cosine search over the MiniLM item-text embeddings,
with price, category and brand filters. CPU only; no extra index library is needed for 25,903 items.

The index holds only catalog metadata text (title, brand, category, features, description, price band),
never review or interaction text, so it carries no validation or test information.

Usage:
    from src.retrieval.item_index import ItemIndex
    index = ItemIndex.load('<298A>/retrieval_index/item_index_minilm_v1')
    hits = index.search('gentle moisturizer for dry skin', k=10, price_max=25, categories=['Face'])
    hits = index.search_vector(profile_vector, k=10, exclude_items=train_items)   # e.g. a MiniLM user profile
"""
import difflib
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_FORMAT = 'item_index_v1'
FILES = ('embeddings.npy', 'items.parquet', 'index_meta.json')
ITEM_COLUMNS = ['item_idx', 'title', 'brand', 'category_path', 'cat_l2', 'cat_l3', 'price_num', 'price_band']


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def category_levels(category_path):
    # 'Skin Care > Face > Creams & Moisturizers' -> ['skin care', 'face', 'creams & moisturizers']; missing -> []
    if category_path is None or (not isinstance(category_path, str) and pd.isna(category_path)):
        return []
    return [p.strip().lower() for p in str(category_path).split(' > ') if p.strip()]


class ItemIndex:
    def __init__(self, embeddings, items, meta, encoder=None):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(items):
            raise ValueError('embeddings must be a 2-D matrix with one row per item')
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
            raise ValueError('embeddings must be finite and L2-normalized')
        missing = [c for c in ITEM_COLUMNS if c not in items.columns]
        if missing:
            raise ValueError(f'items table is missing columns {missing}')
        if not items.item_idx.is_unique:
            raise ValueError('item_idx must be unique')
        self.embeddings = embeddings
        self.items = items.reset_index(drop=True)
        self.meta = meta
        self.item_ids = self.items.item_idx.to_numpy(dtype=np.int64)
        self.row_of = {int(i): r for r, i in enumerate(self.item_ids)}
        self.price = self.items.price_num.to_numpy(dtype=np.float64)
        self.brand_lower = self.items.brand.fillna('').astype(str).str.strip().str.lower().to_numpy()
        self.levels = [category_levels(p) for p in self.items.category_path]
        self._encoder = encoder

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, directory, verify=True, encoder=None):
        """Load an index folder. encoder: an already loaded SentenceTransformer to reuse for text queries
        (it must be the model and commit in index_meta.json); by default it is loaded on the first text query."""
        directory = Path(directory)
        meta = json.loads((directory / 'index_meta.json').read_text())
        if meta.get('format') != INDEX_FORMAT:
            raise ValueError(f'not an {INDEX_FORMAT} folder: {directory}')
        if verify:
            for name in ('embeddings.npy', 'items.parquet'):
                if sha256_file(directory / name) != meta['sha256'][name]:
                    raise ValueError(f'{name} does not match its checksum in index_meta.json')
        emb = np.load(directory / 'embeddings.npy', allow_pickle=False)
        items = pd.read_parquet(directory / 'items.parquet')
        if emb.shape != (meta['n_items'], meta['dim']) or len(items) != meta['n_items']:
            raise ValueError('index files do not match index_meta.json')
        return cls(emb, items, meta, encoder=encoder)

    # ------------------------------------------------------------------ helpers
    def categories(self, level=None):
        """Category names the filter accepts, with item counts (level 2 = e.g. Face, Body; level 3 below that)."""
        col = {2: 'cat_l2', 3: 'cat_l3'}.get(level)
        if col:
            return self.items[col].dropna().value_counts().rename_axis(col).reset_index(name='n_items')
        names = pd.Series([lvl for levels in self.levels for lvl in levels[1:]])
        return names.value_counts().rename_axis('category').reset_index(name='n_items')

    def _encoder_model(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            enc = self.meta['encoder']
            self._encoder = SentenceTransformer(enc['model'], revision=enc.get('revision'), device='cpu')
        return self._encoder

    def encode_query(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('query text must be a non-empty string')
        prefix = self.meta['encoder'].get('query_prefix', '')
        v = self._encoder_model().encode([prefix + text.strip()], normalize_embeddings=True, convert_to_numpy=True)
        return v[0].astype(np.float32)

    def filter_mask(self, price_min=None, price_max=None, categories=None, brands=None, exclude_items=None,
                    include_unpriced=False):
        """Boolean mask over items. Price bounds are inclusive. Items without a price are dropped when a price bound
        is given, unless include_unpriced=True. A category matches any level of the item's category path, or a
        path prefix such as 'Face > Creams & Moisturizers' (case-insensitive). Unknown category or brand names
        raise ValueError with close matches, so a caller never filters on a misspelled name by accident."""
        mask = np.ones(len(self.items), dtype=bool)
        if price_min is not None or price_max is not None:
            if price_min is not None and price_max is not None and price_min > price_max:
                raise ValueError('price_min is larger than price_max')
            has = ~np.isnan(self.price)
            ok = has.copy()
            if price_min is not None:
                ok &= np.where(has, self.price >= price_min, False)
            if price_max is not None:
                ok &= np.where(has, self.price <= price_max, False)
            mask &= ok | (~has if include_unpriced else False)
        if categories:
            wanted = [category_levels(c) for c in ([categories] if isinstance(categories, str) else categories)]
            known = {lvl for levels in self.levels for lvl in levels}
            for w in wanted:
                if not w:
                    raise ValueError('empty category name')
                if len(w) == 1 and w[0] not in known:
                    close = difflib.get_close_matches(w[0], sorted(known), n=5, cutoff=0.5)
                    raise ValueError(f'unknown category {w[0]!r}; close matches: {close}')
            def match(levels):
                for w in wanted:
                    if len(w) == 1 and w[0] in levels:
                        return True
                    if len(w) > 1 and any(levels[s:s + len(w)] == w for s in range(len(levels) - len(w) + 1)):
                        return True
                return False
            mask &= np.fromiter((match(levels) for levels in self.levels), dtype=bool, count=len(self.levels))
        if brands:
            wanted = {b.strip().lower() for b in ([brands] if isinstance(brands, str) else brands)}
            known = set(self.brand_lower)
            unknown = sorted(wanted - known)
            if unknown:
                close = {u: difflib.get_close_matches(u, sorted(known), n=3, cutoff=0.6) for u in unknown}
                raise ValueError(f'unknown brand(s); close matches: {close}')
            mask &= np.isin(self.brand_lower, list(wanted))
        if exclude_items is not None:
            rows = [self.row_of[int(i)] for i in exclude_items if int(i) in self.row_of]
            mask[rows] = False
        return mask

    # ------------------------------------------------------------------ search
    def search_vector(self, vector, k=10, min_score=None, **filters):
        """Top-k items by cosine similarity to a vector in the index's embedding space, after the filters.
        Returns a DataFrame (rank, item_idx, score and item metadata), possibly empty: an empty result means
        "no suitable match" for these filters or this min_score."""
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.embeddings.shape[1] or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            raise ValueError(f'query vector must be finite, non-zero and of length {self.embeddings.shape[1]}')
        v = v / np.linalg.norm(v)
        rows = np.flatnonzero(self.filter_mask(**filters))
        if len(rows) == 0:
            return self._result(rows, np.array([], dtype=np.float32))
        scores = self.embeddings[rows] @ v
        top = np.argsort(-scores, kind='stable')[:k]
        rows, scores = rows[top], scores[top]
        if min_score is not None:
            keep = scores >= min_score
            rows, scores = rows[keep], scores[keep]
        return self._result(rows, scores)

    def search(self, query, k=10, min_score=None, **filters):
        """Top-k items for a text query (encoded with the same pinned MiniLM as the items), after the filters."""
        return self.search_vector(self.encode_query(query), k=k, min_score=min_score, **filters)

    def _result(self, rows, scores):
        out = self.items.iloc[rows][ITEM_COLUMNS].copy().reset_index(drop=True)
        out.insert(0, 'rank', np.arange(1, len(out) + 1))
        out.insert(2, 'score', np.round(scores.astype(np.float64), 6))
        return out


def build_index(embeddings, items, meta, out_dir):
    """Write an index folder (embeddings.npy, items.parquet, index_meta.json) and return its checksums."""
    ItemIndex(embeddings, items, meta)                          # validates before anything is written
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'embeddings.npy', np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
    items[ITEM_COLUMNS].reset_index(drop=True).to_parquet(out_dir / 'items.parquet', index=False)
    meta = dict(meta, format=INDEX_FORMAT, n_items=int(len(items)), dim=int(np.asarray(embeddings).shape[1]),
                sha256={n: sha256_file(out_dir / n) for n in ('embeddings.npy', 'items.parquet')})
    (out_dir / 'index_meta.json').write_text(json.dumps(meta, indent=2))
    return meta['sha256']
