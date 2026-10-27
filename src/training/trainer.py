"""Part C: one configuration/seed, validation-only selection, resumable training.

Model contract: training_logits(input_seq, target_seq) -> logits, nonzero labels;
next_item_scores(input_seq) -> (batch, catalog + padding) scores.
This module does not choose a model, launch a grid, or evaluate the test split.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    patience: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    seed: int
    device: str = 'cpu'
    min_delta: float = 0.0
    grad_clip_norm: float = 1.0
    lr_step_size: int | None = None
    lr_gamma: float | None = None

    def __post_init__(self):
        for key in ('epochs', 'patience'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.optimizer not in ('Adam', 'AdamW'):
            raise ValueError('optimizer must explicitly be Adam or AdamW')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        for key in ('learning_rate', 'weight_decay', 'min_delta', 'grad_clip_norm'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.learning_rate == 0 or self.grad_clip_norm == 0:
            raise ValueError('learning_rate and grad_clip_norm must be positive')
        if (self.lr_step_size is None) != (self.lr_gamma is None):
            raise ValueError('Specify both lr_step_size and lr_gamma, or neither')
        if self.lr_step_size is not None:
            if type(self.lr_step_size) is not int or self.lr_step_size < 1:
                raise ValueError('lr_step_size must be positive')
            if type(self.lr_gamma) not in (float, int) or not 0 < self.lr_gamma <= 1:
                raise ValueError('lr_gamma must be in (0, 1]')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_write(path, write):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name,
                                     suffix='.tmp', delete=False) as temporary:
        name = Path(temporary.name)
    try:
        write(name)
        os.replace(name, path)
    finally:
        name.unlink(missing_ok=True)


def _json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    _atomic_write(path, lambda temporary: temporary.write_text(text))


def _rng_state(loader):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), *state[2:]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': loader.generator.get_state() if loader.generator is not None else None}


def _restore_rng(state, loader):
    random.setstate(state['python'])
    name, values, *rest = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), *rest))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    if state['loader'] is not None:
        if loader.generator is None:
            raise ValueError('Resume requires the saved DataLoader generator')
        loader.generator.set_state(state['loader'].cpu())


def train_epoch(model, loader, optimizer, device, grad_clip_norm):
    model.train()
    total_loss, n_targets = 0.0, 0
    for batch in loader:
        x = batch['input_seq'].to(device)
        y = batch['target_seq'].to(device)
        if x.dtype != torch.long or y.dtype != torch.long or x.shape != y.shape or x.ndim != 2:
            raise ValueError('Expected matching int64 (batch, length) input_seq/target_seq')
        if not torch.equal(x.eq(0), y.eq(0)) or (x < 0).any() or (y < 0).any():
            raise ValueError('Invalid IDs or misaligned padding')
        keep = y.ne(0)
        if not keep.any():
            continue
        optimizer.zero_grad(set_to_none=True)
        logits, labels = model.training_logits(x, y)
        if logits.ndim != 2 or len(logits) != int(keep.sum()) or not torch.equal(labels, y[keep]):
            raise ValueError('training_logits must cover exactly the nonpadding target positions')
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite training logits')
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        n_targets += len(labels)
    if not n_targets:
        raise ValueError('No nonpadding training targets')
    return total_loss / n_targets, n_targets


def checked_metrics(metrics):
    if 'NDCG@10' not in metrics:
        raise ValueError('Validation must return NDCG@10')
    clean = {key: float(value) for key, value in metrics.items()}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in clean.values()):
        raise ValueError('Ranking metrics must be finite and in [0, 1]')
    return clean


def make_shared_validator(evaluator, data, *, device='cpu', batch_size=128, users=None):
    """Always call the frozen rec_eval.py on full-catalog validation, never test."""
    @torch.no_grad()
    def validate(model):
        model.eval()

        def score_fn(inputs, user_ids):
            del user_ids
            scores = model.next_item_scores(torch.as_tensor(inputs, dtype=torch.long, device=device))
            if scores.shape != (len(inputs), data['n_items'] + 1) or not torch.isfinite(scores).all():
                raise ValueError('Invalid/nonfinite score matrix before history masking')
            return scores.detach().cpu().numpy()

        result = evaluator.evaluate(score_fn, data, split='val', sampled=False,
                                    users=users, batch_size=batch_size, return_topk=False)
        ranks = np.asarray(result['ranks'])
        if not len(ranks) or not np.isfinite(ranks).all() or (ranks < 1).any():
            raise ValueError('Invalid validation ranks')
        return checked_metrics(result['metrics'])
    return validate


def fit(model, train_loader, validate, settings, output_dir, *, metadata, resume_from=None):
    """Train one run. Resume is epoch-boundary, num_workers=0 only, into a NEW directory.

    Epoch budget may be extended on resume; other settings and metadata must match.
    Only load checkpoints produced by this project from a trusted location.
    """
    if train_loader.num_workers != 0:
        raise ValueError('Use num_workers=0 for reproducible epoch-boundary resume')
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Run directory already exists; choose a new output directory')
    config = asdict(settings)
    # Require serializable provenance before creating any run artifacts.
    json.dumps(metadata, allow_nan=False)
    model.to(settings.device)
    optimizer = getattr(torch.optim, settings.optimizer)(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, settings.lr_step_size, settings.lr_gamma)
                 if settings.lr_step_size is not None else None)
    history, start_epoch, bad_epochs = [], 0, 0
    best_score, best_epoch, best_state = -math.inf, 0, None
    stopping_score = -math.inf
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=settings.device, weights_only=True)
        if checkpoint['schema_version'] != 1 or checkpoint['metadata'] != metadata:
            raise ValueError('Checkpoint provenance/model/data mismatch')
        for key, value in config.items():
            if key not in ('epochs', 'device') and checkpoint['settings'][key] != value:
                raise ValueError(f'Checkpoint setting mismatch: {key}')
        start_epoch = checkpoint['epoch']
        if settings.epochs <= start_epoch or checkpoint['bad_epochs'] >= settings.patience:
            raise ValueError('Checkpoint is complete or already early-stopped')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        history, bad_epochs = checkpoint['history'], checkpoint['bad_epochs']
        best_score, best_epoch = checkpoint['best_score'], checkpoint['best_epoch']
        best_state = checkpoint['best_model']
        stopping_score = checkpoint['stopping_score']
        _restore_rng(checkpoint['rng'], train_loader)
    else:
        seed_everything(settings.seed)
    # Atomic ownership: two jobs must never both write the same run directory.
    output.mkdir(parents=True, exist_ok=False)
    _json(output / 'run_config.json', {'training': config, 'metadata': metadata,
                                     'resume_from': str(resume_from) if resume_from else None})
    try:
        for epoch in range(start_epoch + 1, settings.epochs + 1):
            begin = time.perf_counter()
            learning_rate = optimizer.param_groups[0]['lr']
            loss, count = train_epoch(model, train_loader, optimizer, settings.device,
                                       settings.grad_clip_norm)
            train_seconds = time.perf_counter() - begin
            model.eval()
            with torch.no_grad():
                metrics = checked_metrics(validate(model))
            improved = metrics['NDCG@10'] > best_score
            if improved:
                best_score, best_epoch = metrics['NDCG@10'], epoch
                best_state = deepcopy(model.state_dict())
            # min_delta controls stopping only, not which checkpoint is best.
            if metrics['NDCG@10'] > stopping_score + settings.min_delta:
                stopping_score, bad_epochs = metrics['NDCG@10'], 0
            else:
                bad_epochs += 1
            if scheduler is not None:
                scheduler.step()
            history.append({'epoch': epoch, 'train_loss': loss, 'train_targets': count,
                            'learning_rate': learning_rate, 'seconds': time.perf_counter() - begin,
                            'train_seconds': train_seconds,
                            'validation_seconds': time.perf_counter() - begin - train_seconds,
                            'improved': improved, **{'val_' + k: v for k, v in metrics.items()}})
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
            _atomic_write(output / 'history.csv', lambda p: p.write_text(stream.getvalue()))
            checkpoint = {'schema_version': 1, 'epoch': epoch, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict() if scheduler is not None else None,
                          'rng': _rng_state(train_loader), 'settings': config, 'metadata': metadata,
                          'history': history, 'bad_epochs': bad_epochs, 'best_score': best_score,
                          'stopping_score': stopping_score,
                          'best_epoch': best_epoch, 'best_model': best_state}
            _atomic_write(output / 'last.pt', lambda p: torch.save(checkpoint, p))
            if improved or not (output / 'best.pt').exists():
                _atomic_write(output / 'best.pt', lambda p: torch.save(
                    {'schema_version': 1, 'model': best_state, 'epoch': best_epoch,
                     'validation_NDCG@10': best_score, 'metadata': metadata}, p))
            print(f'epoch={epoch} loss={loss:.6f} val_NDCG@10={metrics["NDCG@10"]:.6f}', flush=True)
            if bad_epochs >= settings.patience:
                break
        summary = {'status': 'PASS', 'epochs_completed': history[-1]['epoch'],
                   'best_epoch': best_epoch, 'best_validation_NDCG@10': best_score,
                   'early_stopped': bad_epochs >= settings.patience, 'test_evaluated': False,
                   'checkpoint': 'best.pt', 'metadata': metadata}
        _json(output / 'summary.json', summary)
        return summary
    except Exception as error:
        _json(output / 'failure.json', {'status': 'FAILED', 'type': type(error).__name__,
                                       'message': str(error), 'test_evaluated': False})
        raise


def load_best(model, checkpoint_path, *, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.to(device).load_state_dict(checkpoint['model'])
    model.eval()
    return checkpoint
