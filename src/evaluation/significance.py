"""Paired significance tests on per-user test metrics.

Implements the significance row of docs/metrics_commitment.md (amendment of 2026-11-22):

* Unit: per-user NDCG@10 on the full-catalog test split (all users). For a model trained with seeds 42, 43 and 44,
  each user's NDCG@10 is the mean over the three seeds. Recall@10 and MRR are reported with an interval only.
* Test: two-sided paired randomization (sign-flip) test on the mean per-user difference, 10,000 permutations.
  Most users score 0 under both models, so the Wilcoxon signed-rank test (which drops tied pairs) is not used.
* Interval: paired bootstrap 95% percentile interval of the mean difference, 10,000 resamples of users.
* Random seed 2026. Each comparison draws its own streams from SeedSequence(2026), so results do not depend on
  the order of comparisons. Holm correction across the comparisons in one results table, level 0.05.

Per-user metrics use the same formulas as rec_eval.metrics_from_ranks (tie-aware average rank, the team's frozen
convention). Nothing here trains a model or rescores a model on the test split: model ranks come from the saved
test_ranks.npz of each seed, and each file must reproduce the NDCG@10 recorded in its test_evaluation.json. The
untrained baselines are rescored once through the frozen evaluator and must reproduce docs/baselines/baselines.json.

Usage, from the repository root:
    python -m src.evaluation.significance baselines --processed data/processed \\
        --output results/significance/baseline_ranks
    python -m src.evaluation.significance compare \\
        --model "Model 0=runs/seed_42,runs/seed_43,runs/seed_44" \\
        --baseline-ranks results/significance/baseline_ranks \\
        --output results/significance/model0_vs_baselines
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from src.evaluation.common import digest, json_ready, write_new

N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000
SEED = 2026
ALPHA = 0.05
K = 10
SEEDS = (42, 43, 44)
BASELINES = ('popularity', 'markov_last_item', 'itemknn_cosine')
# Committed comparisons, written as (A, B) and tested as A minus B.
COMMITTED = [('Model 1', 'Model 0'),
             ('Model 0', 'popularity'), ('Model 0', 'markov_last_item'), ('Model 0', 'itemknn_cosine'),
             ('Model 1', 'popularity'), ('Model 1', 'markov_last_item'), ('Model 1', 'itemknn_cosine')]
TIE_POLICY = 'average_rank_then_metrics'
MATCH_TOLERANCE = 1e-12
ROOT = Path(__file__).resolve().parents[2]


def per_user_metrics(ranks, k=K):
    """Per-user NDCG@k, Recall@k and reciprocal rank, with rec_eval's formulas."""
    ranks = np.asarray(ranks, dtype=float)
    if ranks.ndim != 1 or not np.isfinite(ranks).all() or (ranks < 1).any():
        raise ValueError('ranks must be a 1-D array of finite values >= 1')
    hit = ranks <= k
    return {f'NDCG@{k}': np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0),
            f'Recall@{k}': hit.astype(float),
            'MRR': 1.0 / ranks}


def _streams(seed=SEED):
    permutation, bootstrap = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(permutation), np.random.default_rng(bootstrap)


def randomization_test(diff, n_permutations=N_PERMUTATIONS, rng=None, chunk=200):
    """Two-sided paired sign-flip test of mean(diff) = 0. Returns (p, observed mean)."""
    diff = np.asarray(diff, dtype=float)
    if diff.ndim != 1 or not len(diff) or not np.isfinite(diff).all():
        raise ValueError('diff must be a non-empty 1-D array of finite values')
    rng = rng if rng is not None else _streams()[0]
    n = len(diff)
    observed = float(diff.mean())
    threshold = abs(observed) * (1 - 1e-9)          # the identity permutation must count despite rounding
    extreme = 0
    for start in range(0, n_permutations, chunk):
        m = min(chunk, n_permutations - start)
        signs = rng.integers(0, 2, size=(m, n)).astype(np.float32) * 2 - 1
        means = (signs @ diff) / n
        extreme += int(np.count_nonzero(np.abs(means) >= threshold))
    return (extreme + 1) / (n_permutations + 1), observed


def bootstrap_interval(diff, n_resamples=N_BOOTSTRAP, rng=None, alpha=ALPHA, chunk=200):
    """Paired bootstrap percentile interval for mean(diff), resampling users with replacement."""
    diff = np.asarray(diff, dtype=float)
    if diff.ndim != 1 or not len(diff) or not np.isfinite(diff).all():
        raise ValueError('diff must be a non-empty 1-D array of finite values')
    rng = rng if rng is not None else _streams()[1]
    n = len(diff)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - start)
        means[start:start + m] = diff[rng.integers(0, n, size=(m, n))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def holm(p_values):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty(len(p))
    running = 0.0
    for position, index in enumerate(np.argsort(p, kind='stable')):
        running = max(running, (len(p) - position) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compare(a, b, seed=SEED, n_permutations=N_PERMUTATIONS, n_resamples=N_BOOTSTRAP):
    """Compare per-user metric dicts a and b (same users, same order): A minus B."""
    permutation_rng, bootstrap_rng = _streams(seed)
    primary = f'NDCG@{K}'
    diff = a[primary] - b[primary]
    p, observed = randomization_test(diff, n_permutations, permutation_rng)
    low, high = bootstrap_interval(diff, n_resamples, bootstrap_rng)
    mean_b = float(b[primary].mean())
    result = {'metric': primary, 'mean_a': float(a[primary].mean()), 'mean_b': mean_b,
              'difference': observed, 'relative_change': observed / mean_b if mean_b else None,
              'ci95': [low, high], 'p_value': p, 'secondary': {}}
    for name in (f'Recall@{K}', 'MRR'):                 # reported with an interval, not used for decisions
        secondary_rng = _streams(seed)[1]
        d = a[name] - b[name]
        result['secondary'][name] = {'mean_a': float(a[name].mean()), 'mean_b': float(b[name].mean()),
                                     'difference': float(d.mean()),
                                     'ci95': list(bootstrap_interval(d, n_resamples, secondary_rng))}
    return result


def load_model(run_dirs, seeds=SEEDS):
    """Per-user metrics for one model, averaged over its seed runs, after integrity checks."""
    by_seed, users, files = {}, None, {}
    for run_dir in map(Path, run_dirs):
        record = json.loads((run_dir / 'test_evaluation.json').read_text())
        seed = int(record['seed'])
        if seed in by_seed:
            raise ValueError(f'seed {seed} is given twice')
        if record.get('tie_policy', {}).get('name') != TIE_POLICY:
            raise ValueError(f'{run_dir}: tie policy is not {TIE_POLICY}')
        ranks_path = run_dir / record['ranks_file']
        with np.load(ranks_path) as saved:
            order = np.argsort(saved['user_idx'], kind='stable')
            run_users, ranks = saved['user_idx'][order], saved['ranks'][order]
        if len(np.unique(run_users)) != len(run_users):
            raise ValueError(f'{ranks_path}: duplicate users')
        if users is None:
            users = run_users
        elif not np.array_equal(users, run_users):
            raise ValueError(f'{ranks_path}: users differ from the other seeds')
        metrics = per_user_metrics(ranks)
        recorded = record['test_full'][f'NDCG@{K}']
        if abs(metrics[f'NDCG@{K}'].mean() - recorded) > MATCH_TOLERANCE:
            raise ValueError(f'{ranks_path}: ranks do not reproduce the recorded test NDCG@{K}')
        if len(run_users) != record['n_test_cases']:
            raise ValueError(f'{ranks_path}: {len(run_users)} users, record says {record["n_test_cases"]}')
        by_seed[seed] = metrics
        files[seed] = {'ranks_sha256': digest(ranks_path), 'record_sha256': digest(run_dir / 'test_evaluation.json')}
    if tuple(sorted(by_seed)) != tuple(sorted(seeds)):
        raise ValueError(f'seeds {sorted(by_seed)} do not match the committed seeds {sorted(seeds)}')
    averaged = {name: np.mean([by_seed[s][name] for s in sorted(by_seed)], axis=0) for name in by_seed[seeds[0]]}
    return users, averaged, {'seeds': sorted(by_seed), 'files': files}


def load_baselines(directory):
    """Per-user metrics for the baselines saved by the `baselines` command."""
    directory = Path(directory)
    path = directory / 'baseline_test_ranks.npz'
    record = json.loads((directory / 'baseline_ranks.json').read_text())
    if record['ranks_sha256'] != digest(path):
        raise ValueError(f'{path} changed after it was written')
    out = {}
    with np.load(path) as saved:
        users = saved['user_idx']
        for name in BASELINES:
            out[name] = per_user_metrics(saved[name])
    return users, out, {'ranks_sha256': record['ranks_sha256'], 'verified_inputs': record['provenance']['verified_files']}


def align(users, metrics, reference):
    """Reorder per-user metrics to the reference user order; the user sets must be identical."""
    if len(users) != len(reference) or not np.array_equal(np.sort(users), np.sort(reference)):
        raise ValueError('models were evaluated on different users')
    position = np.argsort(users, kind='stable')[np.searchsorted(np.sort(users), reference)]
    return {name: values[position] for name, values in metrics.items()}


def _versions():
    out = {'python': platform.python_version()}
    for package in ('numpy', 'scipy', 'pandas'):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def run_baselines(args):
    from src.baselines.simple import build_stats, itemknn_scorer, markov_scorer, popularity_scorer
    from src.evaluation.common import load_evaluator

    published = json.loads((ROOT / 'docs' / 'baselines' / 'baselines.json').read_text())['metrics']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    evaluator, data, hashes = load_evaluator(args.processed)
    pop, transitions, similarity = build_stats(data['sequences']['train_items'], data['n_items'])
    scorers = {'popularity': popularity_scorer(pop), 'markov_last_item': markov_scorer(transitions, pop),
               'itemknn_cosine': itemknn_scorer(similarity, pop)}
    arrays, metrics, users = {}, {}, None
    for name in BASELINES:
        result = evaluator.evaluate(scorers[name], data, split='test', return_topk=False)
        if users is None:
            users = result['user_idx']
        elif not np.array_equal(users, result['user_idx']):
            raise ValueError('baselines were scored on different users')
        arrays[name] = result['ranks']
        metrics[name] = {key: float(result['metrics'][key]) for key in (f'NDCG@{K}', f'Recall@{K}', 'MRR')}
        for key, value in metrics[name].items():
            if abs(value - published[name]['test'][key]) > MATCH_TOLERANCE:
                raise ValueError(f'{name} test {key} = {value} does not reproduce docs/baselines/baselines.json')
        print(name, {key: round(value, 4) for key, value in metrics[name].items()}, flush=True)
    path = output / 'baseline_test_ranks.npz'
    np.savez(path, user_idx=users, **arrays)
    record = {'purpose': 'per-user test ranks of the untrained baselines for the significance tests',
              'metrics': metrics, 'reproduces': 'docs/baselines/baselines.json (test)',
              'ranks_sha256': digest(path),
              'provenance': {'verified_files': hashes, 'manifest_sha256': digest(Path(args.processed) / 'manifest.json'),
                             'code_sha256': {'significance.py': digest(Path(__file__)),
                                             'simple.py': digest(ROOT / 'src' / 'baselines' / 'simple.py')},
                             'versions': _versions()},
              'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    write_new(output / 'baseline_ranks.json', json.dumps(json_ready(record), indent=2, allow_nan=False) + '\n')
    print(f'wrote {path}')


def parse_model(text):
    name, separator, dirs = text.partition('=')
    if not separator or not name.strip() or not dirs.strip():
        raise argparse.ArgumentTypeError('use "Model name=run_dir,run_dir,run_dir"')
    return name.strip(), [d.strip() for d in dirs.split(',') if d.strip()]


def _p_text(p):
    # 1 / (permutations + 1) is the smallest p-value a randomization test with this many permutations can give.
    return '< 0.0001' if p < 1e-4 else f'{p:.4f}'


def table(rows):
    lines = ['| Comparison (A minus B) | NDCG@10 A | NDCG@10 B | Difference | Relative | 95% CI | p | Holm p | Significant |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        rel = f"{row['relative_change'] * 100:+.1f}%" if row['relative_change'] is not None else 'n/a'
        lines.append(f"| {row['a']} vs {row['b']} | {row['mean_a']:.4f} | {row['mean_b']:.4f} | {row['difference']:+.4f} "
                     f"| {rel} | [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] | {_p_text(row['p_value'])} "
                     f"| {_p_text(row['p_holm'])} | {'yes' if row['significant'] else 'no'} |")
    return '\n'.join(lines)


def run_compare(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} exists; results are never overwritten')
    models, sources, reference = {}, {}, None
    for name, dirs in args.model:
        users, values, source = load_model(dirs)
        reference = users if reference is None else reference
        models[name], sources[name] = align(users, values, reference), source
    if args.baseline_ranks:
        users, values, source = load_baselines(args.baseline_ranks)
        reference = users if reference is None else reference
        for name in BASELINES:
            models[name] = align(users, values[name], reference)
        sources['baselines'] = source
    pairs = [pair for pair in COMMITTED if pair[0] in models and pair[1] in models]
    if not pairs:
        raise ValueError('no committed comparison can be made with the given inputs')
    rows = []
    for a, b in pairs:
        rows.append({'a': a, 'b': b, **compare(models[a], models[b])})
        print(f'{a} vs {b}: difference {rows[-1]["difference"]:+.5f}, p = {rows[-1]["p_value"]:.4g}', flush=True)
    for row, adjusted in zip(rows, holm([row['p_value'] for row in rows])):
        row['p_holm'] = float(adjusted)
        row['significant'] = bool(adjusted < ALPHA)
    missing = [f'{a} vs {b}' for a, b in COMMITTED if (a, b) not in pairs]
    payload = {'commitment': 'docs/metrics_commitment.md, amendment of 2026-11-22',
               'settings': {'metric': f'NDCG@{K}', 'test': 'two-sided paired randomization (sign-flip)',
                            'permutations': N_PERMUTATIONS, 'bootstrap_resamples': N_BOOTSTRAP, 'seed': SEED,
                            'alpha': ALPHA, 'correction': 'Holm, across the comparisons in this table',
                            'seed_averaging': 'per-user mean over seeds 42, 43 and 44', 'n_users': int(len(reference))},
               'comparisons': rows, 'not_yet_available': missing, 'sources': sources,
               'code_sha256': digest(Path(__file__)), 'versions': _versions(),
               'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    output.mkdir(parents=True)
    write_new(output / 'significance.json', json.dumps(json_ready(payload), indent=2, allow_nan=False) + '\n')
    note = ('\nHolm correction covers only the comparisons in this table. Committed comparisons not yet available: '
            + ', '.join(missing) + '.\n') if missing else '\n'
    write_new(output / 'significance.md', '# Paired significance tests, full-catalog test split\n\n' + table(rows) + '\n' + note)
    print(f'wrote {output / "significance.md"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    base = commands.add_parser('baselines', help='rescore the untrained baselines once and save per-user test ranks')
    base.add_argument('--processed', default='data/processed')
    base.add_argument('--output', required=True, help='new directory')
    comp = commands.add_parser('compare', help='run the committed comparisons that the inputs allow')
    comp.add_argument('--model', action='append', type=parse_model, default=[],
                      help='"Model 0=run_dir,run_dir,run_dir" (repeat for Model 1)')
    comp.add_argument('--baseline-ranks', help='directory written by the baselines command')
    comp.add_argument('--output', required=True, help='new directory')
    args = parser.parse_args(argv)
    try:
        run_baselines(args) if args.command == 'baselines' else run_compare(args)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        sys.exit(f'Not run: {error}')


if __name__ == '__main__':
    main()
