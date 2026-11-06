"""Part F: the cold-start variant from notebook 05, prepared for training and evaluation.

Notebook 05 removed a fixed set of new products (cold items) from every training
sequence and blanked validation targets that were cold (val_item = -1). The main
split is unchanged, so cold items DO appear in main training data. A cold-start
result therefore needs a model trained on these cold sequences, not the main
Model 0 checkpoint.

Pure NumPy/pandas; no PyTorch.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ['user_idx', 'train_items', 'val_item', 'test_item']


def load_cold_variant(processed_dir, data):
    """Load the cold files and check they match the frozen main split row by row."""
    path = Path(processed_dir) / 'cold_start'
    cold = pd.read_parquet(path / 'cold_sequences.parquet').sort_values('user_idx').reset_index(drop=True)
    cold_items = np.load(path / 'cold_items.npy')
    info = json.loads((path / 'cold_start_info.json').read_text())
    seq = data['sequences']
    cold_set = set(cold_items.tolist())

    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(len(cold_set) == len(cold_items) == info['n_cold_items'], 'Cold item count mismatch')
    require(np.array_equal(cold.user_idx, seq.user_idx), 'Cold and main users differ')
    require(np.array_equal(cold.test_item, seq.test_item), 'Cold and main test items differ')
    val_cold = seq.val_item.isin(cold_set).to_numpy()
    require(np.array_equal(cold.val_item.eq(-1).to_numpy(), val_cold), 'Blanked validation cases mismatch')
    require(np.array_equal(cold.val_item[~val_cold], seq.val_item[~val_cold]), 'Validation items differ')
    require(np.array_equal(cold.test_is_cold, seq.test_item.isin(cold_set)), 'test_is_cold mismatch')
    require(int(cold.test_is_cold.sum()) == info['n_cold_test_cases'], 'Cold test case count mismatch')
    require(int(val_cold.sum()) == info['n_users_losing_val_case'], 'Lost validation count mismatch')
    require(cold.usable_for_training.all() and cold.train_items.str.len().ge(1).all(),
            'Users without cold-variant training items are not handled')
    removed = {}
    for user, main_items, kept, val in zip(seq.user_idx, seq.train_items, cold.train_items, seq.val_item):
        dropped = [i for i in main_items if i in cold_set]
        require(list(kept) == [i for i in main_items if i not in cold_set],
                'Cold training sequence is not the main sequence minus cold items')
        if val in cold_set:
            dropped.append(int(val))
        if dropped:
            removed[int(user)] = np.asarray(dropped, dtype=np.int64)
    return {'frame': cold[[*COLUMNS, 'test_is_cold']].copy(), 'cold_items': cold_items,
            'info': info, 'removed': removed}


def validation_users(frame):
    """Users who still have a validation case after cold validation items were blanked."""
    return frame.user_idx[frame.val_item.ne(-1)].to_numpy()


def test_view(frame):
    """Rows whose rec_eval test history (train_items + [val_item]) is the cold-variant history.

    rec_eval appends val_item to the test history. Where notebook 05 blanked a cold
    validation item (-1), the last kept training item moves into val_item instead,
    so the history is unchanged and -1 never reaches the model or the mask.
    Use this view with split='test' only; validation uses the frame itself.
    """
    view = frame[COLUMNS].copy()
    train, val = [], []
    for items, v in zip(view.train_items, view.val_item):
        items = list(items)
        if v == -1:
            train.append(items[:-1])
            val.append(items[-1])
        else:
            train.append(items)
            val.append(v)
    view['train_items'] = train
    view['val_item'] = np.asarray(val, dtype=np.int64)
    return view


def mask_items(score_fn, items_by_user):
    """Hide items a user already has but the model never saw (their removed cold items).

    rec_eval masks the history it is given; these items are not in the cold-variant
    history, so without this a model could be credited for re-recommending them.
    """
    def masked(inputs, users):
        scores = np.array(score_fn(inputs, users), dtype=np.float64)
        for row, user in enumerate(users):
            items = items_by_user.get(int(user))
            if items is not None:
                scores[row, items] = -np.inf
        return scores
    return masked


def candidate_counts(view, users, removed, n_items):
    """Items left to rank per test case: catalog minus test history minus removed items."""
    rows = view.set_index('user_idx').loc[users]
    history = rows.train_items.str.len().to_numpy() + 1
    dropped = np.array([len(removed.get(int(u), ())) for u in users])
    return n_items - history - dropped


def chance_metrics(candidates, ks=(5, 10, 20)):
    """Expected metrics if the target's rank were uniform over its candidates."""
    candidates = np.asarray(candidates, dtype=np.int64)
    top = int(candidates.max())
    discounts = np.cumsum(1.0 / np.log2(np.arange(2, top + 2)))
    harmonic = np.cumsum(1.0 / np.arange(1, top + 1))
    out = {}
    for k in ks:
        reach = np.minimum(k, candidates)
        out[f'Recall@{k}'] = float((reach / candidates).mean())
        out[f'HitRate@{k}'] = out[f'Recall@{k}']
        out[f'NDCG@{k}'] = float((discounts[reach - 1] / candidates).mean())
    out['MRR'] = float((harmonic[candidates - 1] / candidates).mean())
    expected_rank = (candidates + 1) / 2
    out['median_rank'] = float(np.median(expected_rank))
    out['mean_relative_rank'] = float((expected_rank / candidates).mean())
    return out


def popularity_scores(frame, n_items):
    """Training-interaction counts in the cold variant; every cold item scores 0."""
    counts = np.bincount(np.concatenate(frame.train_items.to_numpy()).astype(np.int64),
                         minlength=n_items + 1).astype(np.float64)
    counts[0] = 0.0

    def score_fn(inputs, users):
        return np.broadcast_to(counts, (len(users), n_items + 1))
    return score_fn


def evaluate_cases(evaluator, data, view, score_fn, users, removed, item_cat=None):
    """Full-catalog test evaluation for the given users with the shared rec_eval."""
    test_data = {**data, 'sequences': view}
    result = evaluator.evaluate(mask_items(score_fn, removed), test_data, split='test',
                                sampled=False, users=users)
    ranks = result['ranks']
    order = np.searchsorted(users, result['user_idx'])
    candidates = candidate_counts(view, users, removed, data['n_items'])[order]
    metrics = dict(result['metrics'])
    if item_cat is not None:
        metrics['ILD@10'] = evaluator.category_diversity(result['topk'][:, :10], item_cat)
    summary = {'n_cases': result['n_cases'], 'metrics': metrics,
               'median_rank': float(np.median(ranks)),
               # 0.5 = no better than chance, 1.0 = ranked last among its candidates.
               'mean_relative_rank': float(np.mean(ranks / candidates)),
               'chance_metrics': chance_metrics(candidates, tuple(data['protocol']['ks']))}
    return summary, result


def case_users(frame):
    """Sorted user_idx for cold test cases and for the warm control group."""
    cold = np.sort(frame.user_idx[frame.test_is_cold].to_numpy())
    warm = np.sort(frame.user_idx[~frame.test_is_cold].to_numpy())
    return cold, warm
