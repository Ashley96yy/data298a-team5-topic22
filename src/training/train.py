"""Part C entry point: run ONE configuration and ONE seed, not a grid search."""
import argparse
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import torch

from src.data.sequence_dataset import make_dataloaders
from src.models.sasrec import SASRec
from src.training.config import GRID, expand_grid, validate_config
from src.training.integration_check import digest, load_verified_data
from src.training.trainer import (TrainingSettings, fit, load_best,
                                  make_shared_validator, seed_everything)


def resolve_config(path, grid_index, seed):
    """Accept either Part D's grid config or one dict from expand_grid()."""
    config = json.loads(Path(path).read_text())
    if 'search' in config:
        validate_config(config, require_ready=True)
        if grid_index is None or not 0 <= grid_index < 8:
            raise ValueError('Select exactly one configuration with --grid-index 0..7')
        config = expand_grid(config)[grid_index]
    else:
        if grid_index is not None:
            raise ValueError('Expanded configurations do not take --grid-index')
        validate_config({**config, 'search': GRID}, require_ready=True)
        parameters = config.get('model_hyperparameters', {})
        if set(parameters) != set(GRID) or any(parameters[k] not in GRID[k] for k in GRID):
            raise ValueError('Invalid expanded model_hyperparameters')
    if seed not in config['training']['seeds']:
        raise ValueError('--seed must be one of the explicitly configured seeds')
    return config


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    files = ['src/data/sequence_dataset.py', 'src/models/sasrec.py',
             'src/training/trainer.py', 'src/training/train.py',
             'src/training/config.py', 'src/training/integration_check.py']
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'commit': commit, 'source_sha256': {name: digest(root / name) for name in files},
            'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                         'numpy': np.__version__, 'pandas': pd.__version__}}


def synthetic_smoke(output_dir):
    """A tiny CPU fixture, not Amazon data or the project's research evaluator."""
    torch.set_num_threads(2)
    seed_everything(42)
    sequences = pd.DataFrame({
        'user_idx': list(range(8)),
        'train_items': [[u + 1, u + 9, u + 17] for u in range(8)],
        'val_item': [25 + u % 3 for u in range(8)],
        'test_item': [28 + u % 3 for u in range(8)],
    })
    loaders = make_dataloaders(sequences, max_len=50, batch_size=4,
                               eval_batch_size=4, seed=42, pin_memory=False)
    model = SASRec(30, max_len=50, hidden_size=16, num_heads=2, num_layers=1, dropout=0.1)

    @torch.no_grad()
    def validate(current):
        # Synthetic-only deterministic item-ID tie breaking. This is NOT a new
        # evaluator for the Amazon benchmark; real runs call frozen rec_eval.py.
        ranks = []
        for batch in loaders['val']:
            scores = current.next_item_scores(batch['input_seq']).clone()
            if not torch.isfinite(scores).all():
                raise ValueError('Nonfinite synthetic scores')
            scores[:, 0] = -torch.inf
            scores.scatter_(1, batch['history'], -torch.inf)
            targets = batch['target']
            target_scores = scores.gather(1, targets[:, None])
            item_ids = torch.arange(scores.shape[1])[None, :]
            rank = 1 + (scores > target_scores).sum(1) + (
                (scores == target_scores) & (item_ids < targets[:, None])).sum(1)
            ranks.extend(rank.tolist())
        ranks = np.asarray(ranks, dtype=float)
        return {'NDCG@10': float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean()),
                'Recall@10': float((ranks <= 10).mean()), 'MRR': float((1 / ranks).mean())}

    settings = TrainingSettings(epochs=3, patience=2, optimizer='Adam', learning_rate=0.001,
                                 weight_decay=0.0, seed=42)
    metadata = {'purpose': 'synthetic_smoke_only_not_Amazon_results', 'users': 8,
                'items': 30, 'selection_split': 'val', **code_provenance()}
    summary = fit(model, loaders['train'], validate, settings, output_dir, metadata=metadata)
    restored = SASRec(30, max_len=50, hidden_size=16, num_heads=2, num_layers=1, dropout=0.1)
    checkpoint = load_best(restored, Path(output_dir) / 'best.pt')
    if checkpoint['epoch'] != summary['best_epoch']:
        raise ValueError('Checkpoint epoch mismatch')
    restored_metrics = validate(restored)
    if restored_metrics['NDCG@10'] != summary['best_validation_NDCG@10']:
        raise ValueError('Reloaded checkpoint validation mismatch')
    print('Synthetic checkpoint reload: PASS (not formal Model 0 results)')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--grid-index', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--optimizer', choices=['Adam', 'AdamW'])
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--output-dir', required=True, help='New run directory; never overwrite a run')
    parser.add_argument('--resume', help='Trusted last.pt, resuming into a new output directory')
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--limit-train-users', type=int)
    parser.add_argument('--limit-val-users', type=int)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--lr-step-size', type=int)
    parser.add_argument('--lr-gamma', type=float)
    parser.add_argument('--synthetic-smoke', action='store_true')
    parser.add_argument('--allow-unapproved-model', action='store_true')
    parser.add_argument('--allow-legacy-tie-metrics', action='store_true')
    args = parser.parse_args()
    if args.synthetic_smoke:
        if args.resume:
            parser.error('The synthetic CLI smoke is standalone; resume is covered by unit tests')
        synthetic_smoke(args.output_dir)
        return
    try:
        config = resolve_config(args.config, args.grid_index, args.seed)
        if not args.allow_unapproved_model:
            raise ValueError('Model 0 adoption is pending; explicit --allow-unapproved-model '
                             'is required for provisional development runs')
        if not args.allow_legacy_tie_metrics:
            raise ValueError('Shared evaluator tie handling needs review. For development only, '
                             'explicitly acknowledge --allow-legacy-tie-metrics; not final results')
        if args.optimizer is None:
            raise ValueError('Choose --optimizer explicitly; Adam and AdamW are not interchangeable')
        if args.eval_batch_size < 1 or any(v is not None and v < 1 for v in
                                         (args.limit_train_users, args.limit_val_users)):
            raise ValueError('Batch size and optional user limits must be positive')
        training, hyper = config['training'], config['model_hyperparameters']
        settings = TrainingSettings(
            epochs=training['epochs'], patience=training['early_stopping_patience'],
            optimizer=args.optimizer, learning_rate=hyper['learning_rate'],
            weight_decay=training['weight_decay'], seed=args.seed, device=args.device,
            min_delta=args.min_delta, lr_step_size=args.lr_step_size, lr_gamma=args.lr_gamma)
        evaluator, data, hashes = load_verified_data(config['data']['processed_dir'], config)
        sequences = data['sequences'].iloc[:args.limit_train_users] if args.limit_train_users else data['sequences']
        users = data['sequences'].user_idx.iloc[:args.limit_val_users].to_numpy()
        seed_everything(args.seed)
        model = SASRec(data['n_items'], max_len=data['max_seq_len'], hidden_size=hyper['hidden_size'],
                       dropout=hyper['dropout'], num_heads=training['num_heads'],
                       num_layers=training['num_layers'])
        loaders = make_dataloaders(sequences, max_len=data['max_seq_len'],
                                   batch_size=training['batch_size'], seed=args.seed,
                                   sliding_window=False, pin_memory=args.device == 'cuda')
        validate = make_shared_validator(evaluator, data, device=args.device,
                                          batch_size=args.eval_batch_size, users=users)
        metadata = {'purpose': 'provisional_development_not_final_baseline',
                    'selection_split': 'val', 'selection_metric': 'NDCG@10', 'sampled': False,
                    'legacy_tie_policy_unresolved': True, 'test_evaluated': False,
                    'model_hyperparameters': hyper,
                    'model_layers': training['num_layers'], 'model_heads': training['num_heads'],
                    'max_seq_len': data['max_seq_len'], 'batch_size': training['batch_size'],
                    'train_users': len(sequences), 'validation_users': len(users),
                    'eval_batch_size': args.eval_batch_size, 'sliding_window': False,
                    'manifest_sha256': digest(Path(config['data']['processed_dir']) / 'manifest.json'),
                    'verified_files': hashes, **code_provenance()}
        fit(model, loaders['train'], validate, settings, args.output_dir,
             metadata=metadata, resume_from=args.resume)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'Training not started/completed: {error}\n')


if __name__ == '__main__':
    main()
