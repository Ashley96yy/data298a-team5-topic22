"""Part E: repeat ONE selected configuration over several seeds, then score test once per seed.

`train` runs Part C (src.training.train) unchanged, one subprocess per seed.
`evaluate` reloads each best.pt, checks it reproduces its own validation score,
then calls the frozen rec_eval.py on test: full catalog (main protocol), 100
sampled negatives (secondary), user/item group breakdown and category diversity.
Each run's test result is written once next to its checkpoint; rerunning
`evaluate` reuses it instead of scoring the test split again.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from src.evaluation.common import (CATEGORY_FILE, digest, item_categories, json_ready,
                                   load_run, verify_files, write_new)
from src.evaluation.seed_stats import aggregate_groups, aggregate_metrics, markdown_table, mean_std
from src.models.sasrec import SASRec
from src.training.config import load_config
from src.training.integration_check import load_verified_data
from src.training.trainer import load_best

ROOT = Path(__file__).resolve().parents[2]
# Reloaded best.pt must reproduce the validation NDCG@10 recorded during training.
VALIDATION_TOLERANCE = 1e-4
GROUP_METRICS = ['Recall@10', 'NDCG@10', 'MRR']


def train_seeds(runs_dir, seeds, forwarded):
    """Call Part C once per seed. Completed seeds are skipped, failed ones are reported."""
    if any(flag in forwarded for flag in ('--seed', '--output-dir', '--resume')):
        raise ValueError('--seed, --output-dir and --resume are set per seed by this runner')
    runs_dir = Path(runs_dir)
    for seed in seeds:
        output = runs_dir / f'seed_{seed}'
        if (output / 'summary.json').exists():
            print(f'seed {seed}: already trained, skipping {output}')
            continue
        if output.exists():
            raise FileExistsError(f'{output} exists without summary.json. Resume it with '
                                  'src.training.train --resume into a new directory, then pass '
                                  'that directory to evaluate.')
        command = [sys.executable, '-m', 'src.training.train', *forwarded,
                   '--seed', str(seed), '--output-dir', str(output)]
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


def build_model(metadata, n_items):
    hyper = metadata['model_hyperparameters']
    return SASRec(n_items, max_len=metadata['max_seq_len'], hidden_size=hyper['hidden_size'],
                  dropout=hyper['dropout'], num_heads=metadata['model_heads'],
                  num_layers=metadata['model_layers'])


def scorer(model, n_items, device):
    """score_fn in the rec_eval contract: (inputs, users) -> (batch, n_items + 1) NumPy."""
    @torch.no_grad()
    def score_fn(inputs, users):
        del users
        model.eval()
        scores = model.next_item_scores(torch.as_tensor(inputs, dtype=torch.long, device=device))
        if scores.shape != (len(inputs), n_items + 1) or not torch.isfinite(scores).all():
            raise ValueError('Invalid/nonfinite score matrix')
        return scores.float().cpu().numpy()
    return score_fn


def comparable(training, metadata):
    """Everything that must match across seeds: all settings except seed/device, all
    provenance except the Git commit (a README-only commit between seeds is harmless;
    source file hashes are still compared)."""
    return ({k: v for k, v in training.items() if k not in ('seed', 'device')},
            {k: v for k, v in metadata.items() if k != 'commit'})


def check_runs(run_dirs, data, hashes, manifest_sha256, allow_partial):
    runs = [load_run(d) for d in run_dirs]
    seeds = [training['seed'] for training, _, _ in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'Duplicate seeds: {seeds}')
    reference = comparable(*runs[0][:2])
    for run_dir, (training, metadata, _) in zip(run_dirs, runs):
        if comparable(training, metadata) != reference:
            raise ValueError(f'{run_dir} differs from {run_dirs[0]} in more than seed/device')
        if metadata['manifest_sha256'] != manifest_sha256 or metadata['verified_files'] != hashes:
            raise ValueError(f'{run_dir} was trained on different frozen data')
        full = metadata['train_users'] == data['n_users'] == metadata['validation_users']
        if not (full or allow_partial):
            raise ValueError(f'{run_dir} used a user limit; only --allow-partial-runs accepts it')
    return runs


def evaluate_run(run_dir, run, evaluator, data, item_cat, device, batch_size):
    run_dir = Path(run_dir)
    training, metadata, summary = run
    record_path, ranks_path = run_dir / 'test_evaluation.json', run_dir / 'test_ranks.npz'
    checkpoint_sha256 = digest(run_dir / 'best.pt')
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record['checkpoint_sha256'] != checkpoint_sha256:
            raise ValueError(f'{run_dir}/best.pt changed after its test evaluation')
        print(f'seed {training["seed"]}: reusing existing test evaluation')
        return record
    if ranks_path.exists():
        raise FileExistsError(f'{ranks_path} exists without test_evaluation.json; inspect it first')
    model = build_model(metadata, data['n_items'])
    checkpoint = load_best(model, run_dir / 'best.pt', device=device)
    if checkpoint['metadata'] != metadata or checkpoint['epoch'] != summary['best_epoch']:
        raise ValueError(f'{run_dir}/best.pt does not belong to this run')
    score_fn = scorer(model, data['n_items'], device)
    # Same validation users and batch size as during training, then compare scores.
    users = data['sequences'].user_idx.iloc[:metadata['validation_users']].to_numpy()
    val = evaluator.evaluate(score_fn, data, split='val', sampled=False, users=users,
                             batch_size=metadata['eval_batch_size'], return_topk=False)
    gap = abs(val['metrics']['NDCG@10'] - summary['best_validation_NDCG@10'])
    if gap > VALIDATION_TOLERANCE:
        raise ValueError(f'{run_dir}: reloaded validation NDCG@10 differs by {gap:.2e}')
    test = evaluator.evaluate(score_fn, data, split='test', sampled=False, batch_size=batch_size)
    sampled = evaluator.evaluate(score_fn, data, split='test', sampled=True, batch_size=batch_size)
    full_metrics = {**test['metrics'],
                    'ILD@10': evaluator.category_diversity(test['topk'][:, :10], item_cat)}
    groups = evaluator.group_breakdown(test, data)
    np.savez_compressed(ranks_path, user_idx=test['user_idx'], ranks=test['ranks'], topk=test['topk'])
    record = {'seed': training['seed'], 'run_dir': str(run_dir), 'checkpoint_sha256': checkpoint_sha256,
              'best_epoch': summary['best_epoch'], 'purpose': metadata['purpose'],
              'legacy_tie_policy_unresolved': metadata.get('legacy_tie_policy_unresolved'),
              'validation_check': {'saved_NDCG@10': summary['best_validation_NDCG@10'],
                                   'recomputed_NDCG@10': val['metrics']['NDCG@10'], 'gap': gap},
              'n_test_cases': test['n_cases'], 'test_full': full_metrics,
              'test_sampled': sampled['metrics'], 'groups': groups.to_dict('records'),
              'ranks_file': ranks_path.name, 'device': device,
              'evaluated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'numpy': np.__version__, 'pandas': pd.__version__}}
    write_new(record_path, json.dumps(json_ready(record), indent=2, allow_nan=False) + '\n')
    print(f'seed {training["seed"]}: test NDCG@10 = {full_metrics["NDCG@10"]:.4f}', flush=True)
    return record


def write_aggregate(records, output_dir, label):
    full = aggregate_metrics({r['seed']: r['test_full'] for r in records})
    sampled = aggregate_metrics({r['seed']: r['test_sampled'] for r in records})
    groups = aggregate_groups({r['seed']: pd.DataFrame(r['groups']) for r in records})
    seeds = sorted(r['seed'] for r in records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'aggregate.json', json.dumps(json_ready({
        'label': label, 'seeds': seeds, 'n_test_cases': records[0]['n_test_cases'],
        'std': 'sample standard deviation (ddof=1) across seeds',
        'purpose': sorted({r['purpose'] for r in records}),
        'runs': {str(r['seed']): r['run_dir'] for r in records},
        'test_full': full, 'test_sampled': sampled}), indent=2, allow_nan=False) + '\n')
    groups.to_csv(output / 'group_breakdown.csv', index=False)
    lines = [f'# {label}: test results over seeds {", ".join(map(str, seeds))}', '',
             f'Mean ± sample standard deviation over {len(seeds)} seeds, '
             f'{records[0]["n_test_cases"]:,} test users. Selection used validation NDCG@10 only.', '',
             markdown_table([(f'{label}, full catalog', full),
                             (f'{label}, 100 sampled negatives', sampled)],
                            metrics=['Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20', 'MRR']),
             markdown_table([(f'{label}, full catalog', full)],
                            metrics=['Coverage@10', 'Coverage@20', 'ILD@10']),
             '| Grouping | Group | Cases | ' + ' | '.join(GROUP_METRICS) + ' |',
             '|---|---|---|' + '---|' * len(GROUP_METRICS)]
    for row in groups.to_dict('records'):
        cells = [mean_std({'mean': row[f'{m}_mean'], 'std': row[f'{m}_std']}) for m in GROUP_METRICS]
        lines.append(f"| {row['grouping']} | {row['group']} | {row['n_cases']:,} | "
                     + ' | '.join(cells) + ' |')
    write_new(output / 'evaluation_table.md', '\n'.join(lines) + '\n')
    print((output / 'evaluation_table.md').read_text())


def evaluate(args):
    config = load_config(args.config)
    processed = Path(config['data']['processed_dir'])
    evaluator, data, hashes = load_verified_data(processed, config)
    verify_files(processed, [CATEGORY_FILE])
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory exists; choose a new one')
    runs = check_runs(args.run_dirs, data, hashes, digest(processed / 'manifest.json'),
                      args.allow_partial_runs)
    item_cat = item_categories(processed, data)
    records = [evaluate_run(d, run, evaluator, data, item_cat, args.device, args.eval_batch_size)
               for d, run in zip(args.run_dirs, runs)]
    write_aggregate(records, args.output_dir, args.label)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    # allow_abbrev=False: a forwarded --seed must not be read as an abbreviation of --seeds.
    train = commands.add_parser('train', allow_abbrev=False,
                                help='Run src.training.train once per seed; other arguments '
                                'are forwarded to it unchanged')
    train.add_argument('--runs-dir', required=True, help='Each seed writes RUNS_DIR/seed_<seed>')
    train.add_argument('--seeds', type=int, nargs='+', required=True)
    score = commands.add_parser('evaluate', allow_abbrev=False,
                                help='Score each run on test once, then aggregate')
    score.add_argument('--run-dirs', nargs='+', required=True)
    score.add_argument('--config', default='configs/config.json')
    score.add_argument('--output-dir', required=True)
    score.add_argument('--label', default='Model 0 (SASRec)')
    score.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    score.add_argument('--eval-batch-size', type=int, default=256)
    score.add_argument('--allow-partial-runs', action='store_true',
                       help='Accept development runs trained or validated on a user subset')
    args, forwarded = parser.parse_known_args()
    try:
        if args.command == 'train':
            train_seeds(args.runs_dir, args.seeds, forwarded)
        else:
            if forwarded:
                parser.error(f'Unknown arguments: {forwarded}')
            evaluate(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'Part E stopped: {error}\n')
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, f'Part E stopped: training failed ({error})\n')


if __name__ == '__main__':
    main()
