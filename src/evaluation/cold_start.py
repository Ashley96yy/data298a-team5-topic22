"""Part F: can Model 0 recommend products it never saw in training?

Subcommands:
  reference  popularity and chance level on cold and warm test cases (no PyTorch needed)
  train      train Model 0 on the cold-variant sequences, with the settings of a main run
  evaluate   score cold and warm test cases for one or more cold-variant runs (seeds)

Cold cases are the 1,142 users whose test item is a cold product. Warm cases are
every other user, scored by the SAME cold-variant model, as the control group.
The main Model 0 checkpoint cannot be used here: cold items are in main training data.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.evaluation import cold_start_data as cold
from src.evaluation.common import (CATEGORY_FILE, COLD_FILES, configuration_mismatches, digest,
                                   expected_configuration, item_categories, json_ready,
                                   load_evaluator, load_run, partial_reasons, verify_files,
                                   write_new)
from src.evaluation.seed_stats import aggregate_metrics, markdown_table, single_run

TABLE = ['Recall@10', 'NDCG@10', 'MRR', 'median_rank', 'mean_relative_rank']
DIGITS = {'median_rank': 0, 'MRR': 5, 'mean_relative_rank': 3}
SOURCES = ['src/evaluation/cold_start.py', 'src/evaluation/cold_start_data.py',
           'src/evaluation/common.py', 'src/evaluation/multi_seed.py']
ROOT = Path(__file__).resolve().parents[2]


def processed_dir(config_path):
    # Read directly: the planning template may still hold nulls in training settings.
    return Path(json.loads(Path(config_path).read_text())['data']['processed_dir'])


def flat(summary):
    """Metrics plus rank statistics, as one dict for the tables."""
    return {**summary['metrics'], 'median_rank': summary['median_rank'],
            'mean_relative_rank': summary['mean_relative_rank']}


def reference_results(evaluator, data, variant, item_cat):
    frame = variant['frame']
    view = cold.test_view(frame)
    cold_users, warm_users = cold.case_users(frame)
    popularity = cold.popularity_scores(frame, data['n_items'])
    out = {}
    for name, users in (('cold', cold_users), ('warm', warm_users)):
        summary, _ = cold.evaluate_cases(evaluator, data, view, popularity, users,
                                         variant['removed'], item_cat)
        out[name] = {'popularity': summary, 'chance': summary.pop('chance_metrics')}
    return out


def reference_rows(reference):
    return [('Popularity, new products', single_run(flat(reference['cold']['popularity']))),
            ('Chance level, new products', single_run(reference['cold']['chance'])),
            ('Popularity, warm control', single_run(flat(reference['warm']['popularity'])))]


def write_report(output_dir, payload, rows, title, note):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'cold_start.json', json.dumps(json_ready(payload), indent=2, allow_nan=False) + '\n')
    text = f'# {title}\n\n{note}\n\n' + markdown_table(rows, TABLE, label='Scorer, cases', digits=DIGITS)
    write_new(output / 'cold_start_table.md', text)
    print(text)


def run_reference(args):
    processed = processed_dir(args.config)
    evaluator, data, hashes = load_evaluator(processed, [*COLD_FILES, CATEGORY_FILE])
    variant = cold.load_cold_variant(processed, data)
    reference = reference_results(evaluator, data, variant, item_categories(processed, data))
    n_cold, n_warm = (reference[k]['popularity']['n_cases'] for k in ('cold', 'warm'))
    write_report(args.output_dir, {'reference': reference, 'cold_start_info': variant['info'],
                                   'verified_files': hashes}, reference_rows(reference),
                 'Cold-start references (no model)',
                 f'{n_cold:,} new-product test cases, {n_warm:,} warm control cases. Relative rank '
                 '0.5 = chance, 1.0 = last. Popularity uses cold-variant training counts.')


def train_variant(args):
    # PyTorch imports stay inside the Model 0 paths so `reference` runs without it.
    from src.data.sequence_dataset import make_dataloaders
    from src.evaluation.multi_seed import build_model
    from src.training.config import load_config
    from src.training.integration_check import load_verified_data
    from src.training.train import code_provenance
    from src.training.trainer import TrainingSettings, fit, make_shared_validator, seed_everything

    limits = [flag for flag, value in (('--limit-train-users', args.limit_train_users),
                                       ('--limit-val-users', args.limit_val_users)) if value]
    if limits and not args.allow_partial_runs:
        raise ValueError(f'{", ".join(limits)} trains or validates on a user subset; '
                         'add --allow-partial-runs for development runs')
    training, reference, _ = load_run(args.reference_run)
    if reference.get('variant') == 'cold_start':
        raise ValueError('--reference-run must be a main-split Model 0 run')
    seed = reference_seed = training['seed']
    if args.seed is not None:
        seed = args.seed
    # The cold model copies the reference run's settings, so the reference must be the
    # team configuration, and the new seed must be one of its seeds.
    mismatches = configuration_mismatches({**training, 'seed': seed}, reference,
                                          expected_configuration(args.config, args.grid_index))
    if mismatches:
        raise ValueError('Reference run or seed does not match the team configuration: '
                         + '; '.join(mismatches))
    settings = TrainingSettings(**{**training, 'seed': seed, 'device': args.device})
    config = load_config(args.config)
    processed = Path(config['data']['processed_dir'])
    evaluator, data, hashes = load_verified_data(processed, config)
    partial = partial_reasons(reference, data['n_users'], data['n_users'])
    if partial and not args.allow_partial_runs:
        raise ValueError(f'Reference run is partial ({"; ".join(partial)}); '
                         'only --allow-partial-runs accepts it')
    if (reference['manifest_sha256'] != digest(processed / 'manifest.json')
            or reference['verified_files'] != hashes):
        raise ValueError('Reference run was trained on different frozen data')
    cold_hashes = verify_files(processed, COLD_FILES)
    variant = cold.load_cold_variant(processed, data)
    frame = variant['frame'][cold.COLUMNS]
    train_frame = frame.iloc[:args.limit_train_users] if args.limit_train_users else frame
    val_users = cold.validation_users(frame)[:args.limit_val_users]
    seed_everything(seed)
    model = build_model(reference, data['n_items'])
    # Same loader as main training. Its val/test datasets are built but unused here.
    loader = make_dataloaders(train_frame, max_len=data['max_seq_len'],
                              batch_size=reference['batch_size'], seed=seed,
                              sliding_window=False, pin_memory=args.device == 'cuda')['train']
    # Early stopping on cold-variant validation (users whose val item was not cold).
    validate = make_shared_validator(evaluator, {**data, 'sequences': frame}, device=args.device,
                                     batch_size=reference['eval_batch_size'], users=val_users)
    metadata = {**{k: reference[k] for k in ('selection_split', 'selection_metric', 'sampled',
                                              'model_hyperparameters', 'model_layers', 'model_heads',
                                              'max_seq_len', 'batch_size', 'eval_batch_size',
                                              'sliding_window', 'manifest_sha256', 'verified_files')},
                'purpose': 'cold_start_variant_' + reference['purpose'], 'variant': 'cold_start',
                'legacy_tie_policy_unresolved': reference.get('legacy_tie_policy_unresolved'),
                'tie_policy': reference.get('tie_policy'),
                'reference_run': str(args.reference_run), 'reference_seed': reference_seed,
                'reference_run_config_sha256': digest(Path(args.reference_run) / 'run_config.json'),
                'cold_files': cold_hashes, 'test_evaluated': False,
                'train_users': len(train_frame), 'validation_users': len(val_users),
                'cold_source_sha256': {name: digest(ROOT / name) for name in SOURCES},
                **code_provenance()}
    fit(model, loader, validate, settings, args.output_dir, metadata=metadata)


def evaluate_variant(run_dir, run, evaluator, data, variant, item_cat, device):
    from src.evaluation.multi_seed import VALIDATION_TOLERANCE, build_model, scorer
    from src.training.trainer import load_best

    run_dir = Path(run_dir)
    training, metadata, summary = run
    record_path = run_dir / 'cold_evaluation.json'
    checkpoint_sha256 = digest(run_dir / 'best.pt')
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record['checkpoint_sha256'] != checkpoint_sha256:
            raise ValueError(f'{run_dir}/best.pt changed after its cold-start evaluation')
        print(f'seed {training["seed"]}: reusing existing cold-start evaluation')
        return record
    model = build_model(metadata, data['n_items'])
    checkpoint = load_best(model, run_dir / 'best.pt', device=device)
    if checkpoint['metadata'] != metadata or checkpoint['epoch'] != summary['best_epoch']:
        raise ValueError(f'{run_dir}/best.pt does not belong to this run')
    score_fn = scorer(model, data['n_items'], device)
    frame = variant['frame']
    val_users = cold.validation_users(frame)[:metadata['validation_users']]
    val = evaluator.evaluate(score_fn, {**data, 'sequences': frame[cold.COLUMNS]}, split='val',
                             users=val_users, batch_size=metadata['eval_batch_size'], return_topk=False)
    gap = abs(val['metrics']['NDCG@10'] - summary['best_validation_NDCG@10'])
    if gap > VALIDATION_TOLERANCE:
        raise ValueError(f'{run_dir}: reloaded validation NDCG@10 differs by {gap:.2e}')
    view = cold.test_view(frame)
    cold_users, warm_users = cold.case_users(frame)
    results = {}
    for name, users in (('cold', cold_users), ('warm', warm_users)):
        cases, result = cold.evaluate_cases(evaluator, data, view, score_fn, users,
                                            variant['removed'], item_cat)
        cases.pop('chance_metrics')
        results[name] = cases
        if name == 'cold':
            cold_ranks = {'user_idx': result['user_idx'], 'ranks': result['ranks']}
    record = {'seed': training['seed'], 'run_dir': str(run_dir), 'checkpoint_sha256': checkpoint_sha256,
              'best_epoch': summary['best_epoch'], 'purpose': metadata['purpose'],
              'validation_check': {'saved_NDCG@10': summary['best_validation_NDCG@10'],
                                   'recomputed_NDCG@10': val['metrics']['NDCG@10'], 'gap': gap},
              'cold': results['cold'], 'warm': results['warm'], 'cold_ranks': cold_ranks,
              'device': device, 'evaluated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    write_new(record_path, json.dumps(json_ready(record), indent=2, allow_nan=False) + '\n')
    print(f'seed {training["seed"]}: cold NDCG@10 = {results["cold"]["metrics"]["NDCG@10"]:.5f}, '
          f'warm NDCG@10 = {results["warm"]["metrics"]["NDCG@10"]:.4f}', flush=True)
    return record


def check_cold_runs(run_dirs, runs, hashes, cold_hashes, n_train_users, n_validation_users,
                    allow_partial, expected):
    """Refuse cold-variant runs that are partial, off-config, or on different frozen data."""
    seeds = [training['seed'] for training, _, _ in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'Duplicate seeds: {seeds}')
    for run_dir, (training, metadata, _) in zip(run_dirs, runs):
        if metadata.get('variant') != 'cold_start':
            raise ValueError(f'{run_dir} is not a cold-variant run (use `train` first)')
        if metadata['verified_files'] != hashes or metadata['cold_files'] != {
                k: cold_hashes[k] for k in COLD_FILES}:
            raise ValueError(f'{run_dir} was trained on different frozen data')
        partial = partial_reasons(metadata, n_train_users, n_validation_users)
        if partial and not allow_partial:
            raise ValueError(f'{run_dir} is partial ({"; ".join(partial)}); '
                             'only --allow-partial-runs accepts it')
        mismatches = configuration_mismatches(training, metadata, expected)
        if mismatches:
            raise ValueError(f'{run_dir} does not match the team configuration: '
                             + '; '.join(mismatches))
    return seeds


def run_evaluate(args):
    from src.training.config import load_config
    from src.training.integration_check import load_verified_data

    config = load_config(args.config)
    processed = Path(config['data']['processed_dir'])
    evaluator, data, hashes = load_verified_data(processed, config)
    cold_hashes = verify_files(processed, [*COLD_FILES, CATEGORY_FILE])
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory exists; choose a new one')
    expected = expected_configuration(args.config, args.grid_index)
    variant = cold.load_cold_variant(processed, data)
    runs = [load_run(d) for d in args.run_dirs]
    seeds = check_cold_runs(args.run_dirs, runs, hashes, cold_hashes, data['n_users'],
                            len(cold.validation_users(variant['frame'])),
                            args.allow_partial_runs, expected)
    item_cat = item_categories(processed, data)
    records = [evaluate_variant(d, run, evaluator, data, variant, item_cat, args.device)
               for d, run in zip(args.run_dirs, runs)]
    reference = reference_results(evaluator, data, variant, item_cat)
    model_cold = aggregate_metrics({r['seed']: flat(r['cold']) for r in records})
    model_warm = aggregate_metrics({r['seed']: flat(r['warm']) for r in records})
    seeds = sorted(seeds)
    rows = [(f'{args.label}, new products', model_cold), *reference_rows(reference)[:2],
            (f'{args.label}, warm control', model_warm), reference_rows(reference)[2]]
    n_cold, n_warm = records[0]['cold']['n_cases'], records[0]['warm']['n_cases']
    write_report(args.output_dir,
                 {'label': args.label, 'seeds': seeds, 'model_cold': model_cold,
                  'model_warm': model_warm, 'reference': reference,
                  'runs': {str(r['seed']): r['run_dir'] for r in records},
                  'purpose': sorted({r['purpose'] for r in records}),
                  'cold_start_info': variant['info']},
                 rows, f'Cold-start test: {args.label}, seeds {", ".join(map(str, seeds))}',
                 f'Model trained on the cold variant (new products removed from training). '
                 f'{n_cold:,} new-product test cases, {n_warm:,} warm control cases scored by the same '
                 'model. Mean ± sample std over seeds. Relative rank 0.5 = chance, 1.0 = last.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    reference = commands.add_parser('reference', help='Popularity and chance level, no model')
    reference.add_argument('--config', default='configs/config.json')
    reference.add_argument('--output-dir', required=True)
    train = commands.add_parser('train', help='Train Model 0 on the cold-variant sequences')
    train.add_argument('--reference-run', required=True,
                       help='Main-split run directory whose settings are reused unchanged')
    train.add_argument('--seed', type=int, help='Defaults to the reference run seed')
    train.add_argument('--config', default='configs/config.json',
                       help='Team configuration the reference run must match')
    train.add_argument('--grid-index', type=int, help='Selected grid configuration (grid configs)')
    train.add_argument('--output-dir', required=True)
    train.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    train.add_argument('--limit-train-users', type=int, help='Development only')
    train.add_argument('--limit-val-users', type=int, help='Development only')
    train.add_argument('--allow-partial-runs', action='store_true',
                       help='Allow user limits and a partial reference run (development only)')
    score = commands.add_parser('evaluate', help='Score cold and warm cases, aggregate seeds')
    score.add_argument('--run-dirs', nargs='+', required=True)
    score.add_argument('--config', default='configs/config.json',
                       help='Team configuration every run must match')
    score.add_argument('--grid-index', type=int, help='Selected grid configuration (grid configs)')
    score.add_argument('--output-dir', required=True)
    score.add_argument('--label', default='Model 0 (SASRec)')
    score.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    score.add_argument('--allow-partial-runs', action='store_true',
                       help='Accept development runs trained or validated on a user subset')
    args = parser.parse_args()
    try:
        {'reference': run_reference, 'train': train_variant, 'evaluate': run_evaluate}[args.command](args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'Part F stopped: {error}\n')


if __name__ == '__main__':
    main()
