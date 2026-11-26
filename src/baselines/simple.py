"""Non-neural baselines for the Model 0 / Model 1 comparison table.

Three untrained baselines, all fitted on the TRAIN interactions only and scored through the
frozen evaluator (``rec_eval.evaluate``: full-catalog ranking, history masked, last-50-item input window):

* popularity     - item training counts (the project's reference baseline)
* markov         - next item = most frequent successor of the user's last item
* itemknn        - cosine item-item co-occurrence summed over the user's whole input window;
                   the similarity matrix is unpruned (no neighbour k)

The two co-occurrence baselines add a FIXED popularity term to every score: ``1e-6 * popularity``, with popularity
scaled so the most popular item is 1. For items with no co-occurrence evidence (score 0) this orders them by
popularity instead of leaving them tied. It is an additive term, not a pure tie-break: for the cosine scorer
two raw scores closer than the term can change order (0.5000000 versus 0.5000005 become 0.5000010 versus 0.50000051
when the popularity values are 1 and 0.01). For Markov the raw scores are integer counts, so the term can only
reorder equal counts. The weight was fixed a priori and no hyperparameter of these baselines is tuned.

Inputs are verified against the frozen manifest (src.evaluation.common.load_evaluator) before anything is scored, and the
result file records the verified input hashes, counts and code and environment provenance.

Usage (from the repository root):
    python -m src.baselines.simple --processed data/processed --output results/baselines
"""
import argparse
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.evaluation.common import digest, json_ready, load_evaluator

TIE_BREAK = 1e-6        # weight of the fixed additive popularity term (see the module docstring)
ROOT = Path(__file__).resolve().parents[2]


def build_stats(train_items, n_items):
    """Return (popularity counts, transition matrix T, normalised co-occurrence matrix S).

    ``train_items`` is an iterable of per-user training sequences with item ids in 1..n_items
    (0 is padding and never appears). Rows/columns of T and S are indexed by item id.
    """
    size = n_items + 1
    pop = np.zeros(size)
    t_rows, t_cols, u_rows, u_cols = [], [], [], []
    n_users = 0
    for u, items in enumerate(train_items):
        n_users += 1
        items = [int(i) for i in items]
        for i in items:
            pop[i] += 1
        t_rows += items[:-1]
        t_cols += items[1:]
        seen = sorted(set(items))
        u_rows += [u] * len(seen)
        u_cols += seen
    T = sp.csr_matrix((np.ones(len(t_rows)), (t_rows, t_cols)), shape=(size, size))
    UI = sp.csr_matrix((np.ones(len(u_rows)), (u_rows, u_cols)), shape=(n_users, size))
    C = (UI.T @ UI).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    users_per_item = np.asarray(UI.sum(axis=0)).ravel()
    inv = sp.diags(1.0 / np.sqrt(np.maximum(users_per_item, 1.0)))
    S = (inv @ C @ inv).tocsr()          # cosine similarity between items
    return pop, T.tocsr(), S


def popularity_scorer(pop):
    p = pop / max(pop.max(), 1.0)
    return lambda X, users: np.tile(p, (len(X), 1))


def markov_scorer(T, pop):
    p = pop / max(pop.max(), 1.0)

    def score(X, users):
        last = np.asarray(X)[:, -1]
        return T[last].toarray() + TIE_BREAK * p      # a padding-only user (last = 0) falls back to popularity
    return score


def itemknn_scorer(S, pop):
    p = pop / max(pop.max(), 1.0)

    def score(X, users):
        X = np.asarray(X)
        rows = np.repeat(np.arange(len(X)), X.shape[1])
        cols = X.ravel()
        keep = cols > 0
        H = sp.csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(len(X), S.shape[0]))
        H.data[:] = 1.0
        return (H @ S).toarray() + TIE_BREAK * p
    return score


def to_markdown(results, metrics):
    lines = ['| Baseline | Split | ' + ' | '.join(metrics) + ' |', '|---|---|' + '---|' * len(metrics)]
    for name, by_split in results.items():
        for split, m in by_split.items():
            lines.append(f'| {name} | {split} | ' + ' | '.join(f'{m[k]:.4f}' for k in metrics) + ' |')
    return '\n'.join(lines)


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(processed_dir, hashes, data, n_cases):
    """What was verified and what produced the numbers: input hashes, counts, code and environment."""
    return {
        'manifest_sha256': digest(Path(processed_dir) / 'manifest.json'),
        'verified_files': hashes,
        'dataset': {'n_users': int(data['n_users']), 'n_items': int(data['n_items']),
                    'max_seq_len': int(data['max_seq_len']), 'cases': n_cases},
        'code': {'git_commit': _git_commit(), 'simple_py_sha256': digest(Path(__file__).resolve())},
        'environment': {'python': platform.python_version(), 'numpy': _package_version('numpy'),
                        'pandas': _package_version('pandas'), 'scipy': _package_version('scipy')},
        'scorers': {'popularity_term': 'fixed additive: 1e-6 * popularity scaled to max 1, added to the co-occurrence scorers',
                    'itemknn': 'cosine similarity, unpruned (no neighbour k), summed over the evaluator input window',
                    'input_window': 'last max_seq_len items as supplied by the frozen evaluator'},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', default='data/processed')
    ap.add_argument('--output', default='results/baselines')
    args = ap.parse_args(argv)

    try:                                                   # nothing is scored or written unless the frozen inputs verify
        R, data, hashes = load_evaluator(args.processed)
    except (FileNotFoundError, KeyError, ValueError) as error:
        sys.exit(f'Baselines not run: frozen inputs could not be verified: {error!r}')
    pop, T, S = build_stats(data['sequences']['train_items'], data['n_items'])
    scorers = {'popularity': popularity_scorer(pop),
               'markov_last_item': markov_scorer(T, pop),
               'itemknn_cosine': itemknn_scorer(S, pop)}

    metrics = ['Recall@5', 'Recall@10', 'Recall@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR']
    results, groups, n_cases = {}, {}, {}
    for name, fn in scorers.items():
        results[name], groups[name] = {}, {}
        for split in ('val', 'test'):
            res = R.evaluate(fn, data, split=split, return_topk=False)
            n_cases[split] = int(res['n_cases'])
            results[name][split] = {k: float(res['metrics'][k]) for k in metrics}
            groups[name][split] = R.group_breakdown(res, data).to_dict(orient='records')
            print(name, split, {k: round(results[name][split][k], 4) for k in ('Recall@10', 'NDCG@10')}, flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    payload = {'metrics': results, 'groups': groups, 'tie_break_weight': TIE_BREAK,
               'provenance': provenance(args.processed, hashes, data, n_cases)}
    (out / 'baselines.json').write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False))
    (out / 'baselines.md').write_text(to_markdown(results, metrics) + '\n')
    print(f'wrote {out / "baselines.json"} and {out / "baselines.md"}')


if __name__ == '__main__':
    main()
