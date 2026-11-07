"""Part D: run all eight configurations using Part C, on one explicit tuning seed."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from src.training.config import expand_grid, load_config
from src.training.integration_check import digest
from src.training.trainer import _json
from src.training.train import TIE_POLICY

ROOT = Path(__file__).resolve().parents[2]


def run_grid(config_path, output_dir, *, seed, optimizer, device='cpu',
             eval_batch_size=128, limit_train_users=None, limit_val_users=None,
             discard_last_checkpoint=False):
    config = load_config(config_path, require_ready=True)
    if config['training'].get('optimizer', optimizer) != optimizer:
        raise ValueError('Optimizer differs from the configured optimizer')
    if seed not in config['training']['seeds']:
        raise ValueError('Tuning seed must be in training.seeds')
    if optimizer not in ('Adam', 'AdamW') or device not in ('cpu', 'cuda', 'mps'):
        raise ValueError('Choose a supported optimizer and device explicitly')
    if any(type(v) is not int or v < 1 for v in
           [eval_batch_size] + [v for v in (limit_train_users, limit_val_users) if v is not None]):
        raise ValueError('Batch size and user limits must be positive integers')
    # Resolve the data path before child processes change to the repository root.
    config['data']['processed_dir'] = str(Path(config['data']['processed_dir']).resolve())
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / 'search_config.json'
    _json(snapshot, config)
    plan = {'purpose': 'model0_grid_search',
            'user_limited': limit_train_users is not None or limit_val_users is not None,
            'tie_policy': TIE_POLICY.copy(),
            'selection_split': 'val', 'selection_metric': 'NDCG@10', 'sampled': False,
            'seed': seed, 'seed_policy': 'one_explicit_tuning_seed',
            'optimizer': optimizer, 'device': device, 'eval_batch_size': eval_batch_size,
            'discard_last_checkpoint': discard_last_checkpoint,
            'limit_train_users': limit_train_users, 'limit_val_users': limit_val_users,
            'tie_break': 'first_configuration_in_config_order', 'test_evaluated': False,
            'runner_sha256': digest(Path(__file__)), 'configuration_count': 8}
    _json(output / 'search_plan.json', plan)
    rows = []
    try:
        for index, expanded in enumerate(expand_grid(config)):
            run_dir = output / f'run_{index:02d}_seed_{seed}'
            command = [sys.executable, '-B', '-m', 'src.training.train',
                       '--config', str(snapshot), '--grid-index', str(index),
                       '--seed', str(seed), '--optimizer', optimizer, '--device', device,
                       '--eval-batch-size', str(eval_batch_size), '--output-dir', str(run_dir)]
            for flag, value in (('--limit-train-users', limit_train_users),
                                ('--limit-val-users', limit_val_users)):
                if value is not None:
                    command.extend([flag, str(value)])
            row = {'grid_index': index, 'seed': seed, **expanded['model_hyperparameters'],
                   'status': 'FAILED', 'validation_NDCG@10': '', 'best_epoch': '',
                   'seconds': 0, 'run_dir': run_dir.name}
            start = time.perf_counter()
            print(f'Running configuration {index + 1}/8: {expanded["model_hyperparameters"]}', flush=True)
            try:
                with (output / f'run_{index:02d}.log').open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                summary = json.loads((run_dir / 'summary.json').read_text())
                score = summary['best_validation_NDCG@10']
                metadata = summary['metadata']
                if (summary['status'] != 'PASS' or summary['test_evaluated'] is not False
                    or metadata['selection_split'] != 'val' or metadata['sampled'] is not False
                    or metadata.get('tie_policy') != TIE_POLICY
                    or metadata['model_hyperparameters'] != expanded['model_hyperparameters']
                    or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                    or not (run_dir / 'best.pt').is_file()):
                    raise ValueError('Invalid validation result or missing checkpoint')
                row.update({'status': 'PASS', 'validation_NDCG@10': score})
                row['best_epoch'] = summary['best_epoch']
                if discard_last_checkpoint:
                    (run_dir / 'last.pt').unlink(missing_ok=True)
            finally:
                row['seconds'] = time.perf_counter() - start
                rows.append(row)
                with (output / 'grid_results.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
        # max preserves configuration order on equal scores; never select partial grids.
        best = max(rows, key=lambda r: r['validation_NDCG@10'])
        selected = expand_grid(config)[best['grid_index']]
        selected['training']['seeds'] = [seed]
        selected['output_dir'] = str(output / best['run_dir'])
        shutil.copyfile(output / best['run_dir'] / 'best.pt', output / 'best_checkpoint.pt')
        _json(output / 'best_config.json', selected)
        result = {**plan, 'status': 'PASS', 'completed_runs': len(rows),
                  'best_grid_index': best['grid_index'],
                  'best_validation_NDCG@10': best['validation_NDCG@10'],
                  'best_epoch': best['best_epoch'], 'source_checkpoint': best['run_dir'] + '/best.pt',
                  'checkpoint_sha256': digest(output / 'best_checkpoint.pt')}
        _json(output / 'selection.json', result)
        return result
    except Exception as error:
        _json(output / 'failure.json', {'status': 'FAILED', 'error': str(error),
                                       'completed_runs': sum(r['status'] == 'PASS' for r in rows),
                                       'test_evaluated': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--output-dir')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--optimizer', choices=['Adam', 'AdamW'])
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--limit-train-users', type=int)
    parser.add_argument('--limit-val-users', type=int)
    parser.add_argument('--discard-last-checkpoint', action='store_true',
                        help='Delete each successful run resume checkpoint; keep best.pt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        if args.dry_run:
            print(json.dumps(expand_grid(load_config(args.config)), indent=2))
            return
        if args.seed is None or args.optimizer is None or args.output_dir is None:
            parser.error('--seed, --optimizer and --output-dir are required for execution')
        options = vars(args).copy()
        options.pop('dry_run')
        options['config_path'] = options.pop('config')
        print(json.dumps(run_grid(**options), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f'Grid search failed: {error}\n')


if __name__ == '__main__':
    main()
