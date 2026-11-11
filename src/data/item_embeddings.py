"""Verified item-text encoding; no validation/test interaction text is read.

BLaIR uses normalized CLS; MiniLM uses normalized attention-masked mean pooling.
Exact model ID and immutable Hugging Face revision are supplied by the caller.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ENCODERS = {'blair': {'pooling': 'cls', 'dimension': 768, 'max_length': 512},
            'minilm': {'pooling': 'mean', 'dimension': 384, 'max_length': 256}}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_items(processed):
    processed = Path(processed)
    manifest = json.loads((processed / 'manifest.json').read_text())
    hashes = {}
    for name in ['dataset_info.json', 'item_text.parquet']:
        hashes[name] = digest(processed / name)
        if hashes[name] != manifest['processed_files'][name]['sha256']:
            raise ValueError('Frozen input checksum mismatch: ' + name)
    info = json.loads((processed / 'dataset_info.json').read_text())
    n_items = info['n_items']
    # Metadata-only text avoids the need to regenerate review snippets for the
    # separately retrained cold-start variant. Do not concatenate other fields.
    items = pd.read_parquet(processed / 'item_text.parquet', columns=['item_idx', 'item_text'])
    items = items.sort_values('item_idx').reset_index(drop=True)
    if not np.array_equal(items.item_idx.to_numpy(), np.arange(1, n_items + 1)):
        raise ValueError('Item IDs must cover 1..n_items exactly once')
    if not items.item_text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Every item must have nonempty item_text')
    return items, {'n_items': n_items, 'manifest_sha256': digest(processed / 'manifest.json'),
                   'verified_files': hashes, 'text_column': 'item_text'}


def pool_tokens(hidden, mask, method):
    if method == 'cls':
        vectors = hidden[:, 0]
    elif method == 'mean':
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        vectors = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
    else:
        raise ValueError('Unknown pooling method')
    return F.normalize(vectors.float(), p=2, dim=1)


class HFEncoder:
    def __init__(self, model_id, revision, encoder, device, cache_dir):
        if not re.fullmatch(r'[0-9a-fA-F]{40}', revision or ''):
            raise ValueError('revision must be an immutable 40-character Hugging Face commit SHA')
        if not model_id or encoder not in ENCODERS:
            raise ValueError('Specify the agreed model ID and encoder family')
        from transformers import AutoModel, AutoTokenizer
        self.spec = ENCODERS[encoder]
        self.device = device
        kwargs = dict(revision=revision, trust_remote_code=False, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        self.model = AutoModel.from_pretrained(model_id, use_safetensors=True, **kwargs).to(device).eval()
        if self.model.config.hidden_size != self.spec['dimension']:
            raise ValueError('Unexpected hidden size for this encoder family')

    @torch.inference_mode()
    def __call__(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.spec['max_length'], return_tensors='pt').to(self.device)
        hidden = self.model(**inputs).last_hidden_state
        return pool_tokens(hidden, inputs['attention_mask'], self.spec['pooling']).cpu().numpy()


def encode_items(items, n_items, encode, dimension, batch_size=32, limit=None):
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError('Batch size and optional smoke limit must be positive')
    chosen = items.iloc[:limit] if limit is not None else items
    matrix = np.zeros((n_items + 1, dimension), dtype=np.float32)
    for start in range(0, len(chosen), batch_size):
        batch = chosen.iloc[start:start + batch_size]
        values = np.asarray(encode(batch.item_text.tolist()), dtype=np.float32)
        if values.shape != (len(batch), dimension) or not np.isfinite(values).all():
            raise ValueError('Encoder returned wrong shape or nonfinite values')
        if np.any(np.linalg.norm(values, axis=1) == 0):
            raise ValueError('Encoder returned zero vectors for real items')
        matrix[batch.item_idx.to_numpy()] = values
    return matrix, len(chosen)


def save_embeddings(output, encoder, matrix, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    name = f'item_text_emb_{encoder}.npy'
    np.save(output / name, matrix, allow_pickle=False)
    metadata = {**metadata, 'schema_version': 1, 'file': name,
                'embedding_sha256': digest(output / name), 'shape': list(matrix.shape),
                'dtype': str(matrix.dtype), 'padding_idx': 0}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return output / name


def load_embeddings(directory, *, manifest_sha256, n_items):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('schema_version') != 1 or metadata.get('complete_catalog') is not True:
        raise ValueError('Training requires a complete, versioned catalog embedding artifact')
    if metadata.get('manifest_sha256') != manifest_sha256 or metadata.get('n_items') != n_items:
        raise ValueError('Embedding artifact belongs to a different frozen dataset')
    name = metadata['file']
    if name not in ('item_text_emb_blair.npy', 'item_text_emb_minilm.npy'):
        raise ValueError('Invalid embedding filename')
    if digest(directory / name) != metadata['embedding_sha256']:
        raise ValueError('Embedding checksum mismatch')
    matrix = np.load(directory / name, allow_pickle=False)
    if (matrix.dtype != np.float32 or list(matrix.shape) != metadata['shape'] or matrix.ndim != 2
            or matrix.shape[0] != n_items + 1 or matrix.shape[1] < 1
            or not np.isfinite(matrix).all() or np.count_nonzero(matrix[0])
            or np.any(np.linalg.norm(matrix[1:], axis=1) == 0)):
        raise ValueError('Invalid embedding shape, dtype, padding or values')
    return matrix, metadata


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--processed', default='data/processed')
    ap.add_argument('--encoder', choices=ENCODERS, required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--revision', required=True, help='Immutable 40-character model commit SHA')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--limit-items', type=int, help='Smoke only; partial artifacts cannot train a model')
    ap.add_argument('--cache-dir', default='results/encoder_cache')
    args = ap.parse_args()
    if Path(args.output_dir).exists():
        ap.error('Output directory already exists; choose a new directory')
    if args.batch_size < 1 or (args.limit_items is not None and args.limit_items < 1):
        ap.error('Batch size and smoke limit must be positive')
    items, provenance = read_items(args.processed)
    encode = HFEncoder(args.model_id, args.revision, args.encoder, args.device, args.cache_dir)
    started = time.perf_counter()
    matrix, count = encode_items(items, provenance['n_items'], encode, ENCODERS[args.encoder]['dimension'],
                                 args.batch_size, args.limit_items)
    metadata = {**provenance, **ENCODERS[args.encoder], 'encoder': args.encoder,
                'model_id': args.model_id, 'revision': args.revision, 'normalize': True,
                'encoded_items': count, 'complete_catalog': count == provenance['n_items'],
                'batch_size': args.batch_size, 'device': args.device,
                'seconds_encoding': time.perf_counter() - started,
                'source_sha256': digest(__file__),
                'versions': {'python': platform.python_version(),
                             **{p: importlib.metadata.version(p) for p in
                                ['torch', 'transformers', 'huggingface-hub', 'numpy', 'pandas', 'pyarrow']}}}
    print(save_embeddings(args.output_dir, args.encoder, matrix, metadata))


if __name__ == '__main__':
    main()
