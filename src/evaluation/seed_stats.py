"""Part E: aggregate per-seed metrics into mean and sample standard deviation.

Pure NumPy/pandas, no model code. Used for the main Model 0 seeds and the cold-start runs.
"""
import math

import numpy as np
import pandas as pd

GROUP_KEYS = ['grouping', 'group']
TABLE_METRICS = ['Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20', 'MRR', 'Coverage@10']


def summarize(values):
    """Mean and sample std (ddof=1). The std is None for a single run, never a fake 0."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Need at least one finite value per metric')
    return {'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else None,
            'min': float(values.min()), 'max': float(values.max()), 'n': int(len(values))}


def aggregate_metrics(per_seed):
    """per_seed: {seed: {metric: value}}. Every seed must report the same metrics."""
    if len(per_seed) < 1:
        raise ValueError('No runs to aggregate')
    names = [set(m) for m in per_seed.values()]
    if any(n != names[0] for n in names):
        raise ValueError('Seeds report different metric sets')
    seeds = sorted(per_seed)
    return {metric: {**summarize([per_seed[s][metric] for s in seeds]),
                     'values': {str(s): float(per_seed[s][metric]) for s in seeds}}
            for metric in sorted(names[0])}


def aggregate_groups(per_seed):
    """per_seed: {seed: DataFrame from rec_eval.group_breakdown}. Case counts must agree."""
    frames = []
    for seed, frame in sorted(per_seed.items()):
        frames.append(pd.DataFrame(frame).assign(seed=seed))
    rows = pd.concat(frames, ignore_index=True)
    counts = rows.groupby(GROUP_KEYS).n_cases.nunique()
    if (counts != 1).any():
        raise ValueError('Group case counts differ between seeds; runs used different users')
    metrics = [c for c in rows.columns if c not in (*GROUP_KEYS, 'n_cases', 'seed')]
    out = []
    for (grouping, group), d in rows.groupby(GROUP_KEYS, sort=False):
        row = {'grouping': grouping, 'group': group, 'n_cases': int(d.n_cases.iloc[0]),
               'n_seeds': len(d)}
        for metric in metrics:
            s = summarize(d[metric])
            row[f'{metric}_mean'], row[f'{metric}_std'] = s['mean'], s['std']
        out.append(row)
    return pd.DataFrame(out)


def mean_std(summary, digits=4):
    """'0.0412 ± 0.0011', or just the mean when only one seed exists or std is undefined."""
    if summary['std'] is None:
        return f"{summary['mean']:.{digits}f}"
    return f"{summary['mean']:.{digits}f} ± {summary['std']:.{digits}f}"


def markdown_table(rows, metrics=TABLE_METRICS, label='Model', digits=None):
    """rows: [(name, aggregate_metrics(...) result)]. Missing metrics print as a blank."""
    digits = digits or {}
    header = f"| {label} | " + ' | '.join(metrics) + ' |'
    lines = [header, '|' + '---|' * (len(metrics) + 1)]
    for name, aggregate in rows:
        cells = [mean_std(aggregate[m], digits.get(m, 4)) if m in aggregate else '' for m in metrics]
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def single_run(metrics):
    """Wrap deterministic baselines (popularity, chance) so they print in the same table."""
    return {k: {'mean': float(v), 'std': None, 'min': float(v), 'max': float(v), 'n': 1}
            for k, v in metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)}
