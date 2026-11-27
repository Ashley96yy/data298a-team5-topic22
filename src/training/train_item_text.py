"""One item-text SASRec ablation run, selected on frozen validation only.

This is the item-only arm, not the full profile-distilled Model 1 experiment.
"""
import argparse
from pathlib import Path

from src.data.item_embeddings import load_embeddings
from src.data.sequence_dataset import make_dataloaders
from src.models.text_sasrec import TextSASRec, load_text_checkpoint
from src.training.integration_check import load_verified_data, digest
from src.training.train import resolve_config, code_provenance, TIE_POLICY
from src.training.trainer import TrainingSettings, fit, make_shared_validator, seed_everything


def run(config_path, grid_index, seed, embedding_dir, output_dir, device='cpu', smoke=False):
    if Path(output_dir).exists():
        raise FileExistsError('Run already exists; select a new output directory')
    config = resolve_config(config_path, grid_index, seed)
    training, hyper = config['training'], config['model_hyperparameters']
    if training['optimizer'] != 'Adam' or training['scheduler'] != 'none':
        raise ValueError('Item-text ablation keeps the agreed Adam/no-scheduler settings')
    evaluator, data, hashes = load_verified_data(config['data']['processed_dir'], config)
    manifest_hash = digest(Path(config['data']['processed_dir']) / 'manifest.json')
    vectors, embedding_metadata = load_embeddings(embedding_dir, manifest_sha256=manifest_hash,
                                                   n_items=data['n_items'])
    for name, expected_hash in embedding_metadata['verified_files'].items():
        if name not in ('dataset_info.json', 'item_text.parquet'):
            raise ValueError('Unexpected embedding source file')
        if digest(Path(config['data']['processed_dir']) / name) != expected_hash:
            raise ValueError('Embedding source differs from current frozen input: ' + name)
    if set(embedding_metadata['verified_files']) != {'dataset_info.json', 'item_text.parquet'}:
        raise ValueError('Embedding artifact must record both verified source files')
    seed_everything(seed)
    model = TextSASRec(data['n_items'], vectors, max_len=data['max_seq_len'],
                       hidden_size=hyper['hidden_size'], num_heads=training['num_heads'],
                       num_layers=training['num_layers'], dropout=hyper['dropout'])
    sequences = data['sequences'].iloc[:128] if smoke else data['sequences']
    users = sequences.user_idx.to_numpy()
    loaders = make_dataloaders(sequences, max_len=data['max_seq_len'], batch_size=training['batch_size'],
                               seed=seed, sliding_window=False, pin_memory=device == 'cuda')
    validate = make_shared_validator(evaluator, data, device=device, batch_size=128, users=users)
    settings = TrainingSettings(epochs=2 if smoke else training['epochs'],
        patience=training['early_stopping_patience'], optimizer='Adam', learning_rate=hyper['learning_rate'],
        weight_decay=training['weight_decay'], seed=seed, device=device)
    provenance = code_provenance()
    root = Path(__file__).resolve().parents[2]
    for name in ['src/models/text_sasrec.py', 'src/data/item_embeddings.py', 'src/training/train_item_text.py']:
        provenance['source_sha256'][name] = digest(root / name)
    metadata = {'purpose': 'item_text_smoke_only' if smoke else 'model1_item_text_ablation',
                'model_class': 'TextSASRec', 'text_architecture': model.architecture,
                'text_fusion': 'ID plus bias-free projected frozen text; tied input/output',
                'projection_initialization': 'zeros', 'embedding_artifact': embedding_metadata,
                'selection_split': 'val', 'selection_metric': 'NDCG@10', 'sampled': False,
                'test_evaluated': False, 'tie_policy': TIE_POLICY.copy(),
                'model_hyperparameters': hyper, 'model_layers': training['num_layers'],
                'model_heads': training['num_heads'], 'max_seq_len': data['max_seq_len'],
                'batch_size': training['batch_size'], 'train_users': len(sequences),
                'validation_users': len(users), 'sliding_window': False,
                'manifest_sha256': manifest_hash, 'verified_files': hashes, **provenance}
    summary = fit(model, loaders['train'], validate, settings, output_dir, metadata=metadata)
    restored, checkpoint = load_text_checkpoint(Path(output_dir) / 'best.pt', device)
    if checkpoint['epoch'] != summary['best_epoch']:
        raise ValueError('Reloaded checkpoint epoch differs from summary')
    print('Saved and loaded TextSASRec checkpoint. No test evaluation performed.')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/config.json')
    ap.add_argument('--grid-index', type=int)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    ap.add_argument('--smoke', action='store_true', help='128 users and 2 epochs; not a final result')
    args = ap.parse_args()
    run(args.config, args.grid_index, args.seed, args.embeddings, args.output_dir, args.device, args.smoke)


if __name__ == '__main__':
    main()
