"""Shared helpers for Parts E and F that do not need PyTorch.

`load_evaluator` repeats the manifest check of src/training/integration_check.py
because that module imports torch at import time; the popularity and chance
references in Part F must run in an environment without it.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Same nine files as integration_check.REQUIRED, plus what Parts E and F read.
EVALUATOR_FILES = ['dataset_info.json', 'rec_eval.py', 'sequences.parquet',
                   'eval/eval_protocol.json', 'eval/neg_val.npy', 'eval/neg_test.npy',
                   'eval/eval_cases.parquet', 'groups/item_groups.parquet', 'groups/user_groups.parquet']
CATEGORY_FILE = 'item_text.parquet'
COLD_FILES = ['cold_start/cold_items.npy', 'cold_start/cold_sequences.parquet',
              'cold_start/cold_start_info.json']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_files(processed_dir, names):
    """Check each file against the frozen manifest and return its SHA-256."""
    path = Path(processed_dir)
    manifest = json.loads((path / 'manifest.json').read_text())
    checked = {}
    for name in names:
        actual = digest(path / name)
        if actual != manifest['processed_files'][name]['sha256']:
            raise ValueError(f'Manifest checksum mismatch: {name}')
        checked[name] = actual
    return checked


def load_evaluator(processed_dir, extra_files=()):
    """Verify, then import the frozen rec_eval.py and load the dataset with it."""
    hashes = verify_files(processed_dir, [*EVALUATOR_FILES, *extra_files])
    spec = importlib.util.spec_from_file_location('shared_rec_eval', Path(processed_dir) / 'rec_eval.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator, evaluator.load_dataset(processed_dir), hashes


def item_categories(processed_dir, data):
    """Category per item_idx (index 0 = padding), built exactly as notebook 07 does."""
    items = pd.read_parquet(Path(processed_dir) / CATEGORY_FILE).sort_values('item_idx')
    if not np.array_equal(items.item_idx.to_numpy(), np.arange(1, data['n_items'] + 1)):
        raise ValueError('item_text.parquet does not cover item_idx 1..n_items exactly')
    column = 'cat_l2' if data['info']['subset']['kind'] == 'category' else 'cat_l1'
    return np.array(['padding'] + items[column].fillna('unknown').tolist(), dtype=object)


def json_ready(value):
    """Convert NumPy scalars/arrays so json.dumps(allow_nan=False) accepts them."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_run(run_dir):
    """Settings, metadata and summary of one completed Part C run directory."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text())
    if summary.get('status') != 'PASS':
        raise ValueError(f'{run_dir} did not complete')
    run_config = json.loads((run_dir / 'run_config.json').read_text())
    return run_config['training'], run_config['metadata'], summary


def write_new(path, text):
    """Write a file that must not exist yet (results are never silently overwritten)."""
    with Path(path).open('x') as stream:
        stream.write(text)
