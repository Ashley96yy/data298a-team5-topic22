"""Importable Part B adapter, derived from sequence_dataloader.ipynb sections 2-3.

The teammate notebook is unchanged. No Drive mount, file writes, model training,
or evaluator is executed on import. Keep sliding_window=False for the baseline.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def left_pad(seq, max_len):
    seq = np.asarray(seq, dtype=np.int64)[-max_len:]
    out = np.zeros(max_len, dtype=np.int64)
    if len(seq):
        out[max_len - len(seq):] = seq
    return out


def training_windows(items, max_len, sliding_window=False):
    items = list(items)
    if not sliding_window:
        return [items[-(max_len + 1):]]
    windows, end = [], len(items)
    while end >= 2:
        windows.append(items[max(0, end - max_len - 1):end])
        end -= max_len
    return windows


def build_train_arrays(sequences, max_len, sliding_window=False):
    users, inputs, targets = [], [], []
    for u, items in zip(sequences.user_idx.to_numpy(), sequences.train_items):
        for w in training_windows(items, max_len, sliding_window):
            users.append(u)
            inputs.append(left_pad(w[:-1], max_len))
            targets.append(left_pad(w[1:], max_len))
    return np.asarray(users, dtype=np.int64), np.stack(inputs), np.stack(targets)


def build_eval_arrays(sequences, split, max_len):
    if split == 'val':
        histories = [np.asarray(t, dtype=np.int64) for t in sequences.train_items]
        targets = sequences.val_item.to_numpy(dtype=np.int64)
    elif split == 'test':
        histories = [np.append(np.asarray(t, dtype=np.int64), v)
                     for t, v in zip(sequences.train_items, sequences.val_item)]
        targets = sequences.test_item.to_numpy(dtype=np.int64)
    else:
        raise ValueError("split must be 'val' or 'test'")
    inputs = np.stack([left_pad(h, max_len) for h in histories])
    return sequences.user_idx.to_numpy(dtype=np.int64), inputs, targets, histories


class SequenceDataset(Dataset):
    def __init__(self, sequences, mode='train', max_len=50, sliding_window=False):
        if mode not in ('train', 'val', 'test'):
            raise ValueError("mode must be 'train', 'val' or 'test'")
        self.mode, self.max_len = mode, max_len
        if mode == 'train':
            users, inputs, targets = build_train_arrays(sequences, max_len, sliding_window)
            self.histories = None
        else:
            users, inputs, targets, histories = build_eval_arrays(sequences, mode, max_len)
            # Parquet-backed NumPy views can be read-only; own these tensors safely.
            self.histories = [torch.from_numpy(h.copy()) for h in histories]
        self.user_idx = torch.from_numpy(users)
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        if self.mode == 'train':
            return {'user_idx': self.user_idx[i], 'input_seq': self.inputs[i],
                    'target_seq': self.targets[i]}
        return {'user_idx': self.user_idx[i], 'input_seq': self.inputs[i],
                'target': self.targets[i], 'history': self.histories[i]}


def eval_collate(batch):
    return {
        'user_idx': torch.stack([b['user_idx'] for b in batch]),
        'input_seq': torch.stack([b['input_seq'] for b in batch]),
        'target': torch.stack([b['target'] for b in batch]),
        'history': torch.nn.utils.rnn.pad_sequence(
            [b['history'] for b in batch], batch_first=True, padding_value=0),
    }


def make_dataloaders(sequences, max_len=50, batch_size=256, eval_batch_size=512,
                     num_workers=0, seed=42, sliding_window=False, pin_memory=None):
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    generator = torch.Generator().manual_seed(seed)
    return {
        'train': DataLoader(SequenceDataset(sequences, 'train', max_len, sliding_window),
                            batch_size=batch_size, shuffle=True, generator=generator,
                            drop_last=False, num_workers=num_workers, pin_memory=pin_memory),
        **{split: DataLoader(SequenceDataset(sequences, split, max_len),
                             batch_size=eval_batch_size, shuffle=False,
                             collate_fn=eval_collate, num_workers=num_workers,
                             pin_memory=pin_memory) for split in ('val', 'test')},
    }
