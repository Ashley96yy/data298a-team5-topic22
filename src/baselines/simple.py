"""Non-neural baselines for the Model 0 / Model 1 comparison table.

Three untrained baselines, all fitted on the TRAIN interactions only and scored through the
shared, frozen ``rec_eval.evaluate`` (full-catalog ranking, history masked):

* popularity     - item training counts (the project's reference baseline)
* markov         - next item = most frequent successor of the user's last item
* itemknn        - cosine item-item co-occurrence summed over the user's whole history

Popularity is used ONLY as a tiny tie-breaker for the two co-occurrence baselines (weight 1e-6),
so items with no co-occurrence evidence are ordered by popularity instead of tying.
No hyperparameter of these baselines is tuned; the tie-breaker weight was fixed a priori.

Usage (from the repository root):
    python -m src.baselines.simple --processed data/processed --output results/baselines
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

TIE_BREAK = 1e-6


def load_rec_eval(processed_dir):
    path = Path(processed_dir) / 'rec_eval.py'
    if not path.exists():
        raise FileNotFoundError(f'frozen rec_eval.py not found at {path}')
    spec = importlib.util.spec_from_file_location('shared_rec_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed', default='data/processed')
    ap.add_argument('--output', default='results/baselines')
    args = ap.parse_args(argv)

    R = load_rec_eval(args.processed)
    data = R.load_dataset(args.processed)
    pop, T, S = build_stats(data['sequences']['train_items'], data['n_items'])
    scorers = {'popularity': popularity_scorer(pop),
               'markov_last_item': markov_scorer(T, pop),
               'itemknn_cosine': itemknn_scorer(S, pop)}

    metrics = ['Recall@5', 'Recall@10', 'Recall@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR']
    results, groups = {}, {}
    for name, fn in scorers.items():
        results[name], groups[name] = {}, {}
        for split in ('val', 'test'):
            res = R.evaluate(fn, data, split=split, return_topk=False)
            results[name][split] = {k: float(res['metrics'][k]) for k in metrics}
            groups[name][split] = R.group_breakdown(res, data).to_dict(orient='records')
            print(name, split, {k: round(results[name][split][k], 4) for k in ('Recall@10', 'NDCG@10')}, flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'baselines.json').write_text(json.dumps({'metrics': results, 'groups': groups,
                                                    'tie_break_weight': TIE_BREAK}, indent=2))
    (out / 'baselines.md').write_text(to_markdown(results, metrics) + '\n')
    print(f'wrote {out / "baselines.json"} and {out / "baselines.md"}')


if __name__ == '__main__':
    main()
