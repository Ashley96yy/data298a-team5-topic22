"""Verify frozen artifacts, then run a bounded real-data training/validation check."""
import argparse
import hashlib
import importlib.util
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.models.sasrec import SASRec
from src.training.config import load_config

REQUIRED = ['dataset_info.json', 'rec_eval.py', 'sequences.parquet',
            'eval/eval_protocol.json', 'eval/neg_val.npy', 'eval/neg_test.npy',
            'eval/eval_cases.parquet', 'groups/item_groups.parquet', 'groups/user_groups.parquet']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified_data(path, config):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    checked = {}
    for name in REQUIRED:
        actual = digest(path / name)
        if actual != manifest['processed_files'][name]['sha256']:
            raise ValueError(f'Manifest checksum mismatch: {name}')
        checked[name] = actual
    # Execute only after checking against the team's manifest and reviewing source.
    spec = importlib.util.spec_from_file_location('shared_rec_eval', path / 'rec_eval.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    data = evaluator.load_dataset(path)
    info = data['info']
    expected = config['data']
    if (info['dataset'] != expected['expected_dataset'] or
        info['subset']['value'] != expected['expected_subset'] or
        info['max_seq_len'] != expected['max_seq_len'] or info['padding_idx'] != 0):
        raise ValueError('Dataset identity, subset, sequence length or padding mismatch')
    protocol = data['protocol']
    if (protocol['main'] != 'full_catalog_ranking' or not protocol['mask_history'] or
        10 not in protocol['ks']):
        raise ValueError('Expected full-catalog, history-masked NDCG@10 protocol')
    seq = data['sequences']
    if len(seq) != data['n_users'] or seq.user_idx.nunique() != len(seq):
        raise ValueError('User count or uniqueness mismatch')
    interactions = 0
    for row in seq.itertuples():
        history = list(row.train_items)
        ids = history + [row.val_item, row.test_item]
        if len(history) < 3 or any(i < 1 or i > data['n_items'] for i in ids):
            raise ValueError('Invalid training length or item IDs')
        if len(set(ids)) != len(ids):
            raise ValueError('Repeated items or holdout overlap')
        interactions += len(ids)
    if interactions != info['n_interactions']:
        raise ValueError('Interaction count mismatch')
    for key in ('neg_val', 'neg_test'):
        if data[key].shape[0] != len(seq):
            raise ValueError('Negative sample row count mismatch')
    return evaluator, data, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--output-dir', default='results/issue1_smoke')
    parser.add_argument('--train-users', type=int, default=256)
    parser.add_argument('--val-users', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if min(args.train_users, args.val_users, args.epochs, args.batch_size) < 1:
        parser.error('Smoke-test counts must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'integration_report.json'
    if report_path.exists():
        parser.error('Output already contains a report; use a new output directory')
    config = load_config(args.config)
    evaluator, data, hashes = load_verified_data(args.processed_dir, config)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    # Explicit smoke settings, not silently promoted to full-training defaults.
    settings = dict(hidden_size=64, dropout=0.2, learning_rate=0.001,
                    num_heads=4, num_layers=2, weight_decay=1e-5, seed=42,
                    epochs=args.epochs, batch_size=args.batch_size, device='cpu')
    model = SASRec(data['n_items'], data['max_seq_len'], hidden_size=64,
                   num_heads=4, num_layers=2, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    seq = data['sequences'].iloc[:args.train_users]
    x, y = evaluator.train_arrays(seq, data['max_seq_len'])
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    losses = []
    before = model.item_embedding.weight.detach().clone()
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total, count = 0.0, 0
        for tokens, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = model.training_logits(tokens, targets)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(labels); count += len(labels)
        losses.append(total / count)
        print(f'epoch={epoch + 1} loss={losses[-1]:.6f}', flush=True)
    if torch.equal(before, model.item_embedding.weight):
        raise ValueError('Training did not update weights')

    @torch.no_grad()
    def score_fn(inputs, users):
        model.eval()
        scores = model.next_item_scores(torch.as_tensor(inputs)).numpy()
        if scores.shape != (len(inputs), data['n_items'] + 1) or not np.isfinite(scores).all():
            raise ValueError('Invalid score matrix')
        return scores

    users = data['sequences'].user_idx.iloc[:args.val_users].to_numpy()
    result = evaluator.evaluate(score_fn, data, split='val', sampled=False,
                                users=users, batch_size=args.batch_size)
    if not all(np.isfinite(v) for v in result['metrics'].values()):
        raise ValueError('Nonfinite validation metric')
    report = dict(status='PASS', purpose='integration_smoke_test_not_final_baseline',
                  settings=settings, train_users=len(seq), validation_users=result['n_cases'],
                  dataset_users=data['n_users'], catalog_items=data['n_items'],
                  split=result['split'], protocol=result['protocol'],
                  epoch_losses=losses, validation_metrics=result['metrics'],
                  test_evaluated=False, seconds=time.perf_counter()-start,
                  manifest_sha256=digest(Path(args.processed_dir)/'manifest.json'),
                  verified_files=hashes,
                  versions=dict(python=platform.python_version(), torch=torch.__version__,
                                numpy=np.__version__, pandas=pd.__version__))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
