"""Profile distillation for Model 1 (LLM-distilled SASRec student).

Method after Severin et al., "Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge
Distillation" (ECIR 2026, arXiv 2604.21536). An LLM writes a text profile of each training user; the profile is
embedded, reduced to the student's hidden size, and frozen as a per-user target. During stage 1 the student SASRec
is pulled toward that target with an auxiliary MSE loss on its pooled last-layer hidden states; stage 2 trains with
the recommendation loss only. At inference nothing changes: the student is plain SASRec (no LLM, no text encoder).

Choices that are ours, not the paper's, are marked [decision] (see docs/model1_spec.md):
  * PCA (not UMAP) reduces profile embeddings, fit on TRAINING users only, then standardised per dimension so the
    targets live on the same scale as LayerNorm-ed hidden states.
  * Stage 1 lasts a fixed number of epochs; checkpoint selection and early stopping apply to stage 2 only.
  * Users without a profile (subset ablation) simply contribute no distillation term.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ targets
class PCAReducer:
    """Deterministic PCA via SVD. Fit on training-user profile embeddings only (leakage rule)."""

    def __init__(self, n_components, standardize=True, eps=1e-8):
        self.n_components, self.standardize, self.eps = int(n_components), bool(standardize), float(eps)
        self.mean_ = self.components_ = self.scale_ = None

    @staticmethod
    def _checked(X, name):
        X = np.asarray(X, dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError(f'{name} contains non-finite values')
        return X

    def _project(self, X):
        # Some macOS numpy/BLAS builds emit spurious floating-point warnings for float64 matmul even when the
        # result is fine. We therefore check the input before and the output after, instead of relying on warnings.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            Z = (X - self.mean_) @ self.components_.T
        if not np.isfinite(Z).all():
            raise FloatingPointError('non-finite values after PCA projection')
        return Z

    def fit(self, X):
        X = self._checked(X, 'X')
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError('X must be (n_users, d) with at least two users')
        k = min(self.n_components, X.shape[0], X.shape[1])
        if k < self.n_components:
            raise ValueError(f'Cannot extract {self.n_components} components from X of shape {X.shape}')
        self.mean_ = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        comps = vt[:k]
        # Fix the sign of each component so repeated fits give identical output.
        signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
        self.components_ = comps * np.where(signs == 0, 1.0, signs)[:, None]
        Z = self._project(X)
        self.scale_ = Z.std(axis=0) + self.eps if self.standardize else np.ones(k)
        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError('fit the reducer first')
        Z = self._project(self._checked(X, 'X'))
        return (Z / self.scale_).astype(np.float32)


class ProfileTargets:
    """Frozen per-user distillation targets, indexed by user_idx (rows without a profile are marked invalid)."""

    def __init__(self, n_users, dim):
        self.values = torch.zeros(n_users, dim, dtype=torch.float32)
        self.valid = torch.zeros(n_users, dtype=torch.bool)

    @classmethod
    def from_arrays(cls, n_users, user_idx, vectors):
        user_idx = np.asarray(user_idx, dtype=np.int64)
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(user_idx) != len(vectors):
            raise ValueError('user_idx and vectors must have the same length')
        if len(np.unique(user_idx)) != len(user_idx):
            raise ValueError('duplicate user_idx')
        if len(user_idx) and (user_idx.min() < 0 or user_idx.max() >= n_users):
            raise ValueError('user_idx out of range')
        if not np.isfinite(vectors).all():
            raise ValueError('non-finite target vectors')
        obj = cls(n_users, vectors.shape[1])
        obj.values[torch.from_numpy(user_idx)] = torch.from_numpy(vectors)
        obj.valid[torch.from_numpy(user_idx)] = True
        return obj

    @property
    def dim(self):
        return self.values.shape[1]

    def lookup(self, user_idx, device=None):
        idx = user_idx.detach().cpu().long()
        out_v, out_m = self.values[idx], self.valid[idx]
        return (out_v.to(device), out_m.to(device)) if device is not None else (out_v, out_m)


# ------------------------------------------------------------------ losses
def pooled_representation(hidden, tokens):
    """Mean of the last-layer hidden states over NON-padding positions: (batch, length, d) -> (batch, d).

    A row with no real item returns zeros (never NaN).
    """
    keep = tokens.ne(0).unsqueeze(-1).to(hidden.dtype)
    return (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)


def distill_loss(pooled, target, valid):
    """MSE between pooled student representation and frozen target, over users that have a target."""
    if pooled.shape != target.shape:
        raise ValueError(f'shape mismatch: pooled {tuple(pooled.shape)} vs target {tuple(target.shape)}')
    if not bool(valid.any()):
        return pooled.sum() * 0.0                      # keeps the graph; contributes nothing
    return F.mse_loss(pooled[valid], target[valid])


def combine_losses(l_model, l_distill, alpha, use_beta=True, eps=1e-12):
    """L = alpha * beta * L_distill + (1 - alpha) * L_model, with beta = detach(L_model / L_distill).

    beta rescales the auxiliary term to the magnitude of the main loss; it carries no gradient.
    alpha = 0 returns L_model exactly.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be in [0, 1]')
    if alpha == 0.0:
        return l_model
    beta = (l_model.detach() / l_distill.detach().clamp_min(eps)) if use_beta else l_model.new_tensor(1.0)
    return alpha * beta * l_distill + (1.0 - alpha) * l_model


@dataclass(frozen=True)
class DistillSchedule:
    """Stage 1 (epochs 1..stage1_epochs): distillation on. Stage 2: recommendation loss only. [decision]"""
    alpha: float
    stage1_epochs: int
    use_beta: bool = True

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError('alpha must be in [0, 1]')
        if self.stage1_epochs < 0:
            raise ValueError('stage1_epochs must be >= 0')

    def alpha_for_epoch(self, epoch):
        """epoch is 1-based."""
        return self.alpha if epoch <= self.stage1_epochs else 0.0

    def selection_starts_at(self):
        """First 1-based epoch eligible for early stopping and best-checkpoint selection. [decision]"""
        return self.stage1_epochs + 1


# ------------------------------------------------------------------ one training epoch
def train_epoch_distill(model, loader, optimizer, device, grad_clip_norm, targets, alpha, use_beta=True):
    """Drop-in analogue of trainer.train_epoch with the auxiliary term. Returns (mean model loss, n_targets, mean distill loss).

    Each batch must carry 'user_idx', 'input_seq', 'target_seq' (SequenceDataset's training batches).
    With alpha == 0 the update is identical to the plain recommendation loss.
    """
    model.train()
    sum_model, sum_distill, n_targets, n_batches = 0.0, 0.0, 0, 0
    for batch in loader:
        x = batch['input_seq'].to(device)
        y = batch['target_seq'].to(device)
        if x.dtype != torch.long or y.dtype != torch.long or x.shape != y.shape or x.ndim != 2:
            raise ValueError('Expected matching int64 (batch, length) input_seq/target_seq')
        if not torch.equal(x.eq(0), y.eq(0)):
            raise ValueError('Misaligned padding')
        keep = y.ne(0)
        if not keep.any():
            continue
        optimizer.zero_grad(set_to_none=True)
        hidden = model.hidden(x)                                    # (batch, length, d), padding rows zero
        logits = hidden[keep] @ model.item_embedding.weight.T
        l_model = F.cross_entropy(logits, y[keep])
        if alpha > 0.0:
            target, valid = targets.lookup(batch['user_idx'], device)
            l_distill = distill_loss(pooled_representation(hidden, x), target, valid)
            loss = combine_losses(l_model, l_distill, alpha, use_beta)
            sum_distill += float(l_distill.detach())
        else:
            loss = l_model
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        count = int(keep.sum())
        sum_model += float(l_model.detach()) * count
        n_targets += count
        n_batches += 1
    if not n_targets:
        raise ValueError('No nonpadding training targets')
    return sum_model / n_targets, n_targets, (sum_distill / n_batches if alpha > 0.0 else 0.0)


def make_epoch_fn(targets, schedule):
    """Adapter for the trainer: (epoch, model, loader, optimizer, device, grad_clip_norm) -> (loss, n_targets).

    The returned loss is the recommendation loss only, so epoch histories stay comparable with Model 0.
    Requires the trainer's optional epoch hook (see docs/model1_spec.md).
    """
    def epoch_fn(epoch, model, loader, optimizer, device, grad_clip_norm):
        loss, n, _ = train_epoch_distill(model, loader, optimizer, device, grad_clip_norm, targets,
                                         schedule.alpha_for_epoch(epoch), schedule.use_beta)
        return loss, n
    return epoch_fn
