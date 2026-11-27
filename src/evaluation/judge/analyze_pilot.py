"""Describe cross-criterion correlations; this is not human calibration."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import correlation

from .run import CRITERIA


def ranks(values):
    """One-based ranks, with the average rank for ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + 1 + end) / 2
        start = end
    return result


def analyze(scores_path):
    scores_path = Path(scores_path)
    scores_bytes = scores_path.read_bytes()
    metadata_bytes = scores_path.with_name('run_metadata.json').read_bytes()
    metadata = json.loads(metadata_bytes)
    rows = [json.loads(line) for line in scores_bytes.decode('utf-8').splitlines() if line.strip()]
    if (not rows or metadata.get('status') != 'PASS'
            or metadata.get('cases_expected') != len(rows)
            or metadata.get('cases_completed') != len(rows)):
        raise ValueError('Require a complete PASS run with matching case counts')
    mode = metadata.get('mode')
    if mode not in ('smoke', 'evaluation') or metadata.get('synthetic_smoke') is not (mode == 'smoke'):
        raise ValueError('Run must identify its smoke/evaluation cohort')
    seen = set()
    for row in rows:
        cid = row.get('case_id')
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('Require unique case IDs')
        seen.add(cid)
        if row.get('task') not in ('explanation', 'profile'):
            raise ValueError('Unknown task')
        samples = row.get('samples')
        scores = row.get('scores')
        if not isinstance(samples, list) or len(samples) != 20:
            raise ValueError('Require 20 score vectors per case')
        for sample in samples:
            if (not isinstance(sample, dict) or set(sample) != set(CRITERIA)
                    or any(type(v) is not int or not 1 <= v <= 5 for v in sample.values())):
                raise ValueError('Invalid sample scores')
        if not isinstance(scores, dict) or set(scores) != set(CRITERIA):
            raise ValueError('Invalid criterion means')
        for key in CRITERIA:
            value = scores[key]
            expected = sum(s[key] for s in samples) / 20
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)):
                raise ValueError('Saved means differ from the 20 samples')
    groups = {}
    for task in ('explanation', 'profile'):
        cohort = [r for r in rows if r['task'] == task]
        pairs = []
        for first, second in itertools.combinations(CRITERIA, 2):
            x = [r['scores'][first] for r in cohort]
            y = [r['scores'][second] for r in cohort]
            reason = ('fewer_than_3_cases' if len(cohort) < 3 else
                      'constant_criterion' if len(set(x)) < 2 or len(set(y)) < 2 else None)
            pairs.append({'criteria': [first, second], 'n_cases': len(cohort),
                          'rho': None if reason else correlation(ranks(x), ranks(y)),
                          'undefined_reason': reason})
        groups[task] = {'n_cases': len(cohort), 'pairs': pairs}
    return {
        'analysis': 'pairwise_spearman_of_per_case_20_sample_means',
        'mode': mode, 'synthetic_smoke': mode == 'smoke',
        'scores_sha256': hashlib.sha256(scores_bytes).hexdigest(),
        'run_metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'groups': groups,
        'interpretation': ('Descriptive diagnostic only; no acceptance threshold or p-values. '
                           'Correlations do not prove prompt-induced coupling or independence. '
                           'Synthetic checks do not establish real-cohort behavior. '
                           'Independent human calibration remains required.'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.scores)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Saved descriptive pilot correlations:', args.output)


if __name__ == '__main__':
    main()
