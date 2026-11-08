"""Load a trusted Part D winner with configuration and checksum verification.

Part C owns checkpoint writing and epoch-boundary resume. This adapter restores
an inference model from the three portable selection artifacts, without data access.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from src.models.sasrec import SASRec
from src.training.config import GRID, validate_config
from src.training.integration_check import digest


def load_selected_model(directory, device='cpu'):
    """Return (eval-mode model, checkpoint, selection) from a trusted run directory."""
    directory = Path(directory)
    selection = json.loads((directory / 'selection.json').read_text())
    config = json.loads((directory / 'best_config.json').read_text())
    checkpoint_path = directory / 'best_checkpoint.pt'
    if (selection.get('status') != 'PASS' or selection.get('completed_runs') != 8
            or selection.get('selection_split') != 'val'
            or selection.get('selection_metric') != 'NDCG@10'
            or selection.get('sampled') is not False or selection.get('test_evaluated') is not False):
        raise ValueError('Expected a complete validation-only grid selection')
    if digest(checkpoint_path) != selection.get('checkpoint_sha256'):
        raise ValueError('Selected checkpoint checksum mismatch')
    validate_config({**config, 'search': GRID}, require_ready=True)
    hyper = config['model_hyperparameters']
    if set(hyper) != set(GRID) or any(hyper[k] not in GRID[k] for k in GRID):
        raise ValueError('Invalid selected hyperparameters')
    if config['training']['seeds'] != [selection['seed']]:
        raise ValueError('Selected seed mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint schema')
    score = checkpoint['validation_NDCG@10']
    if (not math.isfinite(score) or not 0 <= score <= 1
            or score != selection['best_validation_NDCG@10']
            or checkpoint['epoch'] != selection['best_epoch']):
        raise ValueError('Selected epoch/validation score mismatch')
    metadata = checkpoint['metadata']
    expected = {'model_hyperparameters': hyper,
                'model_layers': config['training']['num_layers'],
                'model_heads': config['training']['num_heads'],
                'max_seq_len': config['data']['max_seq_len'],
                'selection_split': 'val', 'selection_metric': 'NDCG@10', 'sampled': False,
                'tie_policy': selection['tie_policy']}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint metadata does not match selected configuration/protocol')
    n_items = checkpoint['model']['item_embedding.weight'].shape[0] - 1
    model = SASRec(n_items, max_len=metadata['max_seq_len'], hidden_size=hyper['hidden_size'],
                   dropout=hyper['dropout'], num_heads=metadata['model_heads'],
                   num_layers=metadata['model_layers'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    return model, checkpoint, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    args = parser.parse_args()
    model, checkpoint, selection = load_selected_model(args.run_dir, args.device)
    print(json.dumps({'status': 'PASS', 'epoch': checkpoint['epoch'],
                      'validation_NDCG@10': selection['best_validation_NDCG@10'],
                      'model_in_eval_mode': not model.training, 'test_evaluated': False}))


if __name__ == '__main__':
    main()
