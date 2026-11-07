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

from src.training.config import GRID, expand_grid, validate_config

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


def expected_configuration(config_path, grid_index=None):
    """The team-confirmed settings every final run must have used.

    Mirrors src.training.train.resolve_config, which cannot be imported without PyTorch:
    a grid config needs --grid-index, an expanded config (one dict from expand_grid) does not.
    Empty (null) settings are rejected.
    """
    config = json.loads(Path(config_path).read_text())
    if 'search' in config:
        validate_config(config, require_ready=True)
        runs = expand_grid(config)
        if grid_index is None or not 0 <= grid_index < len(runs):
            raise ValueError(f'Select the configuration with --grid-index 0..{len(runs) - 1}')
        return runs[grid_index]
    if grid_index is not None:
        raise ValueError('An expanded configuration does not take --grid-index')
    validate_config({**config, 'search': GRID}, require_ready=True)
    parameters = config.get('model_hyperparameters', {})
    if set(parameters) != set(GRID) or any(parameters[k] not in GRID[k] for k in GRID):
        raise ValueError('Invalid expanded model_hyperparameters')
    return config


def configuration_mismatches(training, metadata, expected):
    """Every difference between one run's recorded settings and the expected configuration."""
    settings, hyper = expected['training'], expected['model_hyperparameters']
    pairs = {'model_hyperparameters': (metadata.get('model_hyperparameters'), hyper),
             'learning_rate': (training.get('learning_rate'), hyper['learning_rate']),
             'batch_size': (metadata.get('batch_size'), settings['batch_size']),
             'epochs': (training.get('epochs'), settings['epochs']),
             'num_layers': (metadata.get('model_layers'), settings['num_layers']),
             'num_heads': (metadata.get('model_heads'), settings['num_heads']),
             'weight_decay': (training.get('weight_decay'), settings['weight_decay']),
             'early_stopping_patience': (training.get('patience'),
                                         settings['early_stopping_patience']),
             'max_seq_len': (metadata.get('max_seq_len'), expected['data']['max_seq_len'])}
    # Older development configs did not declare these settings. Enforce them
    # whenever the team configuration does, including explicit scheduler absence.
    if 'optimizer' in settings:
        pairs['optimizer'] = (training.get('optimizer'), settings['optimizer'])
    if settings.get('scheduler') == 'none':
        for name in ('lr_step_size', 'lr_gamma'):
            pairs[name] = (training.get(name, '<missing>'), None)
    found = [f'{name}: run {run!r}, config {wanted!r}'
             for name, (run, wanted) in pairs.items() if run != wanted]
    if training.get('seed') not in settings['seeds']:
        found.append(f"seed {training.get('seed')!r} is not in the config seeds {settings['seeds']}")
    return found


def partial_reasons(metadata, n_train_users, n_validation_users):
    """Why a run counts as partial: trained or validated on fewer users than the full split."""
    reasons = []
    if metadata['train_users'] != n_train_users:
        reasons.append(f"trained on {metadata['train_users']:,} of {n_train_users:,} users")
    if metadata['validation_users'] != n_validation_users:
        reasons.append(f"validated on {metadata['validation_users']:,} of {n_validation_users:,} users")
    return reasons


def write_new(path, text):
    """Write a file that must not exist yet (results are never silently overwritten)."""
    with Path(path).open('x') as stream:
        stream.write(text)
