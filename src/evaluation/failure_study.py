"""Failure study: why a model misses the true next item (docs/metrics_commitment.md, failure-taxonomy row).

A failure is a test case whose true item is ranked outside the top 10 in every seed of the model (seeds 42, 43 and
44 for a final model), so a case missed by only one seed is not counted. A fixed random sample of failures (at least
50, seed 2026) is assigned ONE category each by the first rule below that applies, so the counts sum to the sample:

1. near_miss         the true item reached ranks 11 to 20 in at least one seed
2. unpopular_target  the true item is outside the head of the training popularity (item group mid or tail)
3. new_brand         the true item has a known brand that is not in the user's history
4. category_switch   the true item's level-2 category differs from the user's most recent item
5. long_gap          more than 180 days passed between the user's last known interaction and the test interaction
6. other             none of the above: a popular item of a familiar brand and category, bought soon after

The categories and their order were chosen after looking at the Model 0 failures (an exploratory analysis, not a
pre-registered test). The same rules are then applied unchanged to every model. Besides the exclusive taxonomy, the
study reports how common each condition is among all failures and among cases the model gets right in every seed.

Inputs: the frozen dataset (sequences, items and item groups, verified against the manifest) and each seed's saved
test_ranks.npz. Nothing is trained or rescored. The case file (titles and histories of the sampled users) is for
manual review and stays outside Git; only aggregate tables are meant to be committed.

    python -m src.evaluation.failure_study --processed data/processed \\
        --model "Model 0=<seed_42 dir>,<seed_43 dir>,<seed_44 dir>" --output results/failure_study/model0
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.common import digest, json_ready, verify_files, write_new
from src.evaluation.significance import SEEDS, load_model, parse_model

K = 10
NEAR_MISS_RANK = 20
LONG_GAP_DAYS = 180
MIN_SAMPLE = 50
SAMPLE_SIZE = 100
SEED = 2026
FILES = ['sequences.parquet', 'items.parquet', 'groups/item_groups.parquet']
CATEGORIES = ['near_miss', 'unpopular_target', 'new_brand', 'category_switch', 'long_gap', 'other']
DESCRIPTIONS = {
    'near_miss': 'True item ranked 11 to 20 in at least one seed',
    'unpopular_target': 'True item outside the head of training popularity (mid or tail)',
    'new_brand': "True item's brand not in the user's history",
    'category_switch': "True item's category differs from the user's most recent item",
    'long_gap': f'More than {LONG_GAP_DAYS} days since the last known interaction',
    'other': 'Popular item of a familiar brand and category, bought soon after',
}
UNKNOWN_BRANDS = {'', 'unknown', 'none', 'n/a', 'na', 'generic'}


def _known(brand):
    return isinstance(brand, str) and brand.strip().lower() not in UNKNOWN_BRANDS


def case_features(sequences, items, item_groups, ranks):
    """One row per test user: the conditions used by the rules, from the frozen tables and the per-seed ranks.

    sequences: user_idx, train_items, val_item, test_item, val_ts, test_ts, n_train (sorted by user_idx)
    items: indexed by item_idx with brand_key and cat_l2; item_groups: indexed by item_idx with item_group, train_count
    ranks: array (n_seeds, n_users) in the same user order.
    """
    ranks = np.asarray(ranks, dtype=float)
    if ranks.ndim != 2 or ranks.shape[1] != len(sequences):
        raise ValueError('ranks must have shape (n_seeds, n_users) in the sequences order')
    brand, category = items['brand_key'], items['cat_l2']
    target = sequences['test_item'].to_numpy()
    histories = [list(t) + [int(v)] for t, v in zip(sequences['train_items'], sequences['val_item'])]
    target_brand = brand.reindex(target).to_numpy()
    history_brands = [{b for b in brand.reindex(h).to_numpy() if _known(b)} for h in histories]
    last_category = category.reindex([h[-1] for h in histories]).to_numpy()
    out = pd.DataFrame({
        'user_idx': sequences['user_idx'].to_numpy(),
        'n_train': sequences['n_train'].to_numpy(),
        'test_item': target,
        'best_rank': ranks.min(axis=0),
        'mean_rank': ranks.mean(axis=0),
        'fail_all': (ranks > K).all(axis=0),
        'hit_all': (ranks <= K).all(axis=0),
        'target_group': item_groups['item_group'].reindex(target).to_numpy(),
        'target_train_count': item_groups['train_count'].reindex(target).to_numpy(),
        'new_brand': [_known(b) and b not in hb for b, hb in zip(target_brand, history_brands)],
        'category_switch': category.reindex(target).to_numpy() != last_category,
        'gap_days': (sequences['test_ts'].to_numpy() - sequences['val_ts'].to_numpy()) / 86_400_000,
    })
    out['near_miss'] = (out.best_rank > K) & (out.best_rank <= NEAR_MISS_RANK)
    out['unpopular_target'] = out.target_group.isin(['mid', 'tail'])
    out['long_gap'] = out.gap_days > LONG_GAP_DAYS
    return out


def assign_category(features):
    """The first rule (in CATEGORIES order) that applies to each case; 'other' if none does."""
    labels = pd.Series('other', index=features.index, dtype=object)
    for name in reversed(CATEGORIES[:-1]):
        labels[features[name].to_numpy(dtype=bool)] = name
    return labels


def taxonomy(sample):
    """Template table: count, share and the first example case of each category, in rule order."""
    counts = sample['category'].value_counts()
    rows = []
    for name in CATEGORIES:
        cases = sample.index[sample['category'] == name]
        rows.append({'category': name, 'description': DESCRIPTIONS[name], 'count': int(counts.get(name, 0)),
                     'share': float(counts.get(name, 0) / len(sample)),
                     'example_case': int(sample.loc[cases[0], 'case']) if len(cases) else None})
    return rows


def prevalence(features):
    """Share of each condition among consistent failures and among consistent hits (not exclusive)."""
    failures, hits = features[features.fail_all], features[features.hit_all]
    conditions = {'near_miss': None, 'unpopular_target': 'unpopular_target', 'new_brand': 'new_brand',
                  'category_switch': 'category_switch', 'long_gap': 'long_gap'}
    rows = []
    for name, column in conditions.items():
        rows.append({'condition': name, 'description': DESCRIPTIONS[name],
                     'share_of_failures': float(failures[name].mean()),
                     'share_of_hits': None if column is None else float(hits[column].mean())})
    rows.append({'condition': 'short_history', 'description': '3 or 4 training interactions',
                 'share_of_failures': float((failures.n_train <= 4).mean()),
                 'share_of_hits': float((hits.n_train <= 4).mean())})
    return rows


def case_file(sample, sequences, items, topk):
    """Readable rows for manual review: recent history, the true item and the model's top 3 (first seed)."""
    title = items['title'].fillna('').str.slice(0, 80)
    by_user = sequences.set_index('user_idx')
    rows = []
    for _, case in sample.iterrows():
        user = by_user.loc[case.user_idx]
        history = (list(user.train_items) + [int(user.val_item)])[-5:]
        rows.append({'case': int(case.case), 'category': case.category, 'user_idx': int(case.user_idx),
                     'n_train': int(case.n_train), 'best_rank': case.best_rank, 'mean_rank': round(case.mean_rank, 1),
                     'gap_days': round(case.gap_days, 1), 'target_group': case.target_group,
                     'target_title': title.get(case.test_item, ''), 'target_brand': items['brand_key'].get(case.test_item),
                     'target_category': items['cat_l2'].get(case.test_item),
                     'recent_history': ' | '.join(title.get(i, '') for i in history),
                     'top3_predicted': ' | '.join(title.get(int(i), '') for i in topk[case.position][:3]),
                     'manual_category': '', 'notes': ''})
    return pd.DataFrame(rows)


def load_ranks(run_dirs):
    """Per-seed ranks and the first seed's top-K lists, after the integrity checks of significance.load_model."""
    users, _, source = load_model(run_dirs)
    ranks, topk = {}, None
    for run_dir in map(Path, run_dirs):
        record = json.loads((run_dir / 'test_evaluation.json').read_text())
        with np.load(run_dir / record['ranks_file']) as saved:
            order = np.argsort(saved['user_idx'], kind='stable')
            ranks[int(record['seed'])] = saved['ranks'][order]
            if int(record['seed']) == SEEDS[0]:
                topk = saved['topk'][order]
    return users, np.vstack([ranks[s] for s in sorted(ranks)]), topk, source


def run(args):
    if args.sample_size < MIN_SAMPLE:
        raise ValueError(f'the failure study needs at least {MIN_SAMPLE} sampled cases')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'{output} exists; results are never overwritten')
    processed = Path(args.processed)
    hashes = verify_files(processed, FILES)
    name, run_dirs = args.model
    users, ranks, topk, source = load_ranks(run_dirs)
    sequences = pd.read_parquet(processed / 'sequences.parquet').sort_values('user_idx').reset_index(drop=True)
    if not np.array_equal(sequences['user_idx'].to_numpy(), users):
        raise ValueError('ranked users differ from the frozen sequences')
    items = pd.read_parquet(processed / 'items.parquet').set_index('item_idx')
    item_groups = pd.read_parquet(processed / 'groups/item_groups.parquet').set_index('item_idx')
    features = case_features(sequences, items, item_groups, ranks)
    features['position'] = np.arange(len(features))
    failures = features[features.fail_all].copy()
    if len(failures) < args.sample_size:
        raise ValueError(f'only {len(failures)} failures, fewer than the sample size')
    failures['category'] = assign_category(failures)
    sample = failures.sample(args.sample_size, random_state=SEED).copy()
    sample['case'] = np.arange(1, len(sample) + 1)
    sample = sample.set_index('case', drop=False)
    table = taxonomy(sample)
    population = failures['category'].value_counts(normalize=True)
    payload = {'model': name, 'definition': f'true item outside the top {K} in every seed',
               'n_users': int(len(features)), 'n_failures': int(len(failures)),
               'n_hits_all_seeds': int(features.hit_all.sum()), 'sample_size': int(len(sample)),
               'sample_seed': SEED, 'rules_in_order': CATEGORIES, 'thresholds': {
                   'k': K, 'near_miss_rank': NEAR_MISS_RANK, 'long_gap_days': LONG_GAP_DAYS},
               'taxonomy_sample': table,
               'taxonomy_all_failures': {c: float(population.get(c, 0.0)) for c in CATEGORIES},
               'prevalence': prevalence(features), 'sources': {'ranks': source, 'verified_inputs': hashes},
               'code_sha256': digest(Path(__file__)),
               'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    output.mkdir(parents=True)
    write_new(output / 'failure_study.json', json.dumps(json_ready(payload), indent=2, allow_nan=False) + '\n')
    write_new(output / 'failure_study.md', report(payload))
    case_file(sample, sequences, items, topk).to_csv(output / 'cases_for_review.csv', index=False)
    print(report(payload))
    print(f'wrote {output}; cases_for_review.csv contains user histories and stays out of Git')


def report(payload):
    lines = [f"# Failure study: {payload['model']}", '',
             f"Failure: {payload['definition']}. {payload['n_failures']:,} of {payload['n_users']:,} test users fail; "
             f"{payload['n_hits_all_seeds']:,} are hits in every seed. Random sample of {payload['sample_size']} failures "
             f"(seed {payload['sample_seed']}), one category each by the first rule that applies, in this order.", '',
             '| Category | Count | % of failures (sample) | % of all failures | Example case | Definition |',
             '|---|---|---|---|---|---|']
    for row in payload['taxonomy_sample']:
        example = f"case {row['example_case']}" if row['example_case'] is not None else 'n/a'
        lines.append(f"| {row['category']} | {row['count']} | {row['share']:.0%} "
                     f"| {payload['taxonomy_all_failures'][row['category']]:.1%} | {example} | {row['description']} |")
    lines += [f"| total | {payload['sample_size']} | 100% | 100% | | |", '',
              'How common each condition is (not exclusive):', '',
              '| Condition | Among failures | Among hits in every seed |', '|---|---|---|']
    for row in payload['prevalence']:
        hits = f"{row['share_of_hits']:.1%}" if row['share_of_hits'] is not None else 'n/a'
        lines.append(f"| {row['condition']} | {row['share_of_failures']:.1%} | {hits} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--processed', default='data/processed')
    parser.add_argument('--model', required=True, type=parse_model, help='"Model 0=run_dir,run_dir,run_dir"')
    parser.add_argument('--sample-size', type=int, default=SAMPLE_SIZE)
    parser.add_argument('--output', required=True, help='new directory')
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        sys.exit(f'Not run: {error}')


if __name__ == '__main__':
    main()
