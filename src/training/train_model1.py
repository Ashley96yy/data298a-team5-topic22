"""Model 1 entry point: ONE configuration and ONE seed of the LLM-distilled SASRec student.

Same data, model, validator, trainer and 8-configuration grid as Model 0 (src.training.train). The additions are the
frozen per-user profile targets, the two-stage distillation schedule (src.models.distill) and the trainer hook from PR #11.

    python -m src.training.train_model1 --config configs/config.json --grid-index 7 --seed 42 --optimizer Adam \
        --device cuda --targets profiles.npz --alpha 0.6 --stage1-epochs 20 --output-dir results/model1/g7_s42_a0.6_e20

(alpha and stage-1 length above are placeholders until the team decides them.) Profile file format: src/data/profile_file.py.
Smoke test of the plumbing on synthetic data only (never a result):   --synthetic-smoke --output-dir /tmp/m1_smoke
"""
import argparse
import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.profile_file import read_profile_file, write_profile_file
from src.data.sequence_dataset import make_dataloaders
from src.models.distill import DistillSchedule, PCAReducer, ProfileTargets, make_epoch_fn
from src.models.sasrec import SASRec
from src.training.integration_check import digest, load_verified_data
from src.training.train import TIE_POLICY, code_provenance, resolve_config
from src.training.trainer import TrainingSettings, fit, make_shared_validator, seed_everything

ROOT = Path(__file__).resolve().parents[2]


def check_optimizer(training, optimizer):
    """Same guard as src.training.train: the CLI optimizer must equal the configured one (older configs may omit it)."""
    if training.get('optimizer', optimizer) != optimizer:
        raise ValueError('CLI optimizer differs from configured optimizer')


def make_schedule(alpha, stage1_epochs, epochs, use_beta=True):
    """Stage 1 must exist and must leave at least one epoch for stage 2 (otherwise nothing could be selected)."""
    if type(stage1_epochs) is not int or not 1 <= stage1_epochs < epochs:
        raise ValueError(f'--stage1-epochs must be an integer in [1, epochs - 1] (epochs = {epochs})')
    return DistillSchedule(alpha=alpha, stage1_epochs=stage1_epochs, use_beta=use_beta)


def build_targets(profile, n_users, n_components, users=None):
    """PCA fit on the (training-user) profile embeddings only, then one frozen target row per user with a profile."""
    user_idx, embeddings = profile['user_idx'], profile['embeddings']
    if user_idx.max() >= n_users:
        raise ValueError('profile file has a user_idx outside the dataset')
    if users is not None:
        keep = np.isin(user_idx, users)
        user_idx, embeddings = user_idx[keep], embeddings[keep]
    if len(user_idx) == 0:
        raise ValueError('no profile rows for the selected training users')
    reducer = PCAReducer(n_components).fit(embeddings)
    targets = ProfileTargets.from_arrays(n_users, user_idx, reducer.transform(embeddings))
    return targets, reducer


def targets_sha256(targets):
    h = hashlib.sha256()
    h.update(targets.values.numpy().tobytes())
    h.update(targets.valid.numpy().tobytes())
    return h.hexdigest()


def distill_metadata(profile, targets, schedule, n_components):
    return {
        'distill_schedule': asdict(schedule),
        'profile_file_sha256': profile['sha256'],
        'profile_source': profile['source'],
        'profile_rows': int(len(profile['user_idx'])),
        'users_with_target': int(targets.valid.sum()),
        'reduction': {'method': 'pca_standardised', 'n_components': n_components, 'fit_on': 'training users only'},
        'reduced_targets_sha256': targets_sha256(targets),
        'distill_source_sha256': digest(ROOT / 'src/models/distill.py'),
        'train_model1_sha256': digest(Path(__file__).resolve()),
    }


def run_distilled(model, loaders, validate, settings, output_dir, metadata, targets, schedule, resume_from=None):
    """Train with the distillation hook, then save the per-epoch stage log next to the run folder's files."""
    hook = make_epoch_fn(targets, schedule)
    summary = fit(model, loaders['train'], validate, settings, output_dir, metadata=metadata, resume_from=resume_from,
                  epoch_fn=hook, selection_start_epoch=schedule.selection_starts_at())
    name = 'distill_log_resumed.csv' if resume_from else 'distill_log.csv'      # after a resume the log covers only the resumed epochs
    pd.DataFrame(hook.log).to_csv(Path(output_dir) / name, index=False)
    return summary


def synthetic_model1_smoke(output_dir):
    """Plumbing check on a tiny synthetic fixture. Not Amazon data, not a research result."""
    torch.set_num_threads(2)
    seed_everything(42)
    n_users, n_items, hidden = 40, 30, 16
    sequences = pd.DataFrame({
        'user_idx': list(range(n_users)),
        'train_items': [[(3 * u + k) % n_items + 1 for k in range(3)] for u in range(n_users)],
        'val_item': [(u + 7) % n_items + 1 for u in range(n_users)],
        'test_item': [(u + 11) % n_items + 1 for u in range(n_users)]})
    loaders = make_dataloaders(sequences, max_len=50, batch_size=8, eval_batch_size=8, seed=42, pin_memory=False)
    model = SASRec(n_items, max_len=50, hidden_size=hidden, num_heads=2, num_layers=1, dropout=0.1)

    @torch.no_grad()
    def validate(current):                                   # synthetic-only ranking, as in src.training.train.synthetic_smoke
        ranks = []
        for batch in loaders['val']:
            scores = current.next_item_scores(batch['input_seq']).clone()
            scores[:, 0] = -torch.inf
            scores.scatter_(1, batch['history'], -torch.inf)
            target = batch['target']
            target_scores = scores.gather(1, target[:, None])
            ids = torch.arange(scores.shape[1])[None, :]
            ranks.extend((1 + (scores > target_scores).sum(1) + ((scores == target_scores) & (ids < target[:, None])).sum(1)).tolist())
        ranks = np.asarray(ranks, dtype=float)
        return {'NDCG@10': float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean())}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'synthetic_profiles.npz'
        write_profile_file(path, np.arange(n_users), np.random.default_rng(0).normal(size=(n_users, 24)),
                           'synthetic', True)
        profile = read_profile_file(path, allow_synthetic=True)
    targets, _ = build_targets(profile, n_users, hidden)
    schedule = make_schedule(0.6, 2, 4)
    settings = TrainingSettings(epochs=4, patience=2, optimizer='Adam', learning_rate=0.001, weight_decay=0.0, seed=42)
    metadata = {'purpose': 'synthetic_smoke_only_not_Amazon_results', 'selection_split': 'val',
                **distill_metadata(profile, targets, schedule, hidden)}
    return run_distilled(model, loaders, validate, settings, output_dir, metadata, targets, schedule)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--grid-index', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--optimizer', choices=['Adam', 'AdamW'])
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--output-dir', required=True, help='New run directory; never overwrite a run')
    parser.add_argument('--resume', help='Trusted last.pt, resuming into a new output directory')
    parser.add_argument('--targets', help='Profile file (.npz), see src/data/profile_file.py')
    parser.add_argument('--alpha', type=float, help='weight of the distillation term in stage 1, in (0, 1]')
    parser.add_argument('--stage1-epochs', type=int)
    parser.add_argument('--no-beta', action='store_true', help='turn off the dynamic rescaling of the distillation term')
    parser.add_argument('--eval-batch-size', type=int, default=128)
    parser.add_argument('--limit-train-users', type=int)
    parser.add_argument('--limit-val-users', type=int)
    parser.add_argument('--min-delta', type=float, default=0.0)
    parser.add_argument('--synthetic-smoke', action='store_true')
    args = parser.parse_args()
    if args.synthetic_smoke:
        summary = synthetic_model1_smoke(args.output_dir)
        print('Synthetic Model 1 smoke: PASS (plumbing only, not a result):', summary['best_epoch'])
        return
    try:
        if args.targets is None or args.alpha is None or args.stage1_epochs is None:
            raise ValueError('A real run needs --targets, --alpha and --stage1-epochs')
        if not 0.0 < args.alpha <= 1.0:
            raise ValueError('--alpha must be in (0, 1]')
        if args.optimizer is None:
            raise ValueError('Choose --optimizer explicitly; Adam and AdamW are not interchangeable')
        config = resolve_config(args.config, args.grid_index, args.seed)
        training, hyper = config['training'], config['model_hyperparameters']
        check_optimizer(training, args.optimizer)
        schedule = make_schedule(args.alpha, args.stage1_epochs, training['epochs'], not args.no_beta)
        settings = TrainingSettings(
            epochs=training['epochs'], patience=training['early_stopping_patience'], optimizer=args.optimizer,
            learning_rate=hyper['learning_rate'], weight_decay=training['weight_decay'], seed=args.seed,
            device=args.device, min_delta=args.min_delta)
        evaluator, data, hashes = load_verified_data(config['data']['processed_dir'], config)
        sequences = data['sequences'].iloc[:args.limit_train_users] if args.limit_train_users else data['sequences']
        users = data['sequences'].user_idx.iloc[:args.limit_val_users].to_numpy()
        profile = read_profile_file(args.targets)                       # synthetic profiles are refused here
        targets, _ = build_targets(profile, data['n_users'], hyper['hidden_size'], users=sequences.user_idx.to_numpy())
        seed_everything(args.seed)
        model = SASRec(data['n_items'], max_len=data['max_seq_len'], hidden_size=hyper['hidden_size'],
                       dropout=hyper['dropout'], num_heads=training['num_heads'], num_layers=training['num_layers'])
        loaders = make_dataloaders(sequences, max_len=data['max_seq_len'], batch_size=training['batch_size'],
                                   seed=args.seed, sliding_window=False, pin_memory=args.device == 'cuda')
        validate = make_shared_validator(evaluator, data, device=args.device, batch_size=args.eval_batch_size, users=users)
        metadata = {'purpose': 'model1_profile_distillation', 'selection_split': 'val', 'selection_metric': 'NDCG@10',
                    'sampled': False, 'tie_policy': TIE_POLICY.copy(), 'test_evaluated': False,
                    'model_hyperparameters': hyper, 'model_layers': training['num_layers'],
                    'model_heads': training['num_heads'], 'max_seq_len': data['max_seq_len'],
                    'batch_size': training['batch_size'], 'train_users': len(sequences), 'validation_users': len(users),
                    'eval_batch_size': args.eval_batch_size, 'sliding_window': False,
                    'manifest_sha256': digest(Path(config['data']['processed_dir']) / 'manifest.json'),
                    'verified_files': hashes, **distill_metadata(profile, targets, schedule, hyper['hidden_size']),
                    **code_provenance()}
        run_distilled(model, loaders, validate, settings, args.output_dir, metadata, targets, schedule, args.resume)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'Training not started/completed: {error}\n')


if __name__ == '__main__':
    main()
