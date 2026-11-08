import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.training.grid_search import run_grid
from src.training.train import TIE_POLICY, resolve_config

ROOT = Path(__file__).resolve().parents[1]


class GridSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = json.loads((ROOT / 'configs/config.json').read_text())
        config['training'].update(batch_size=4, epochs=2, num_layers=1, num_heads=2,
                                  weight_decay=0.0, early_stopping_patience=2, seeds=[42, 43])
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps(config))
        self.output = self.root / 'output'

    def run_search(self, **overrides):
        args = dict(seed=42, optimizer='Adam')
        args.update(overrides)
        return run_grid(self.config, self.output, **args)

    def fake_run(self, command, **kwargs):
        def arg(flag):
            return command[command.index(flag) + 1]
        self.assertNotIn('--resume', command)
        self.assertNotIn('--allow-unapproved-model', command)
        self.assertNotIn('--allow-legacy-tie-metrics', command)
        index = int(arg('--grid-index'))
        config = resolve_config(arg('--config'), index, int(arg('--seed')))
        directory = Path(arg('--output-dir'))
        directory.mkdir()
        # A deliberately nonmonotonic optimum; two equal winners test tie handling.
        score = [0.1, 0.2, 0.8, 0.3, 0.8, 0.4, 0.5, 0.6][index]
        (directory / 'best.pt').write_bytes(f'checkpoint {index}'.encode())
        (directory / 'last.pt').write_bytes(b'resume')
        (directory / 'summary.json').write_text(json.dumps({
            'status': 'PASS', 'test_evaluated': False, 'best_validation_NDCG@10': score,
            'best_epoch': 1, 'metadata': {'tie_policy': TIE_POLICY, 'selection_split': 'val', 'sampled': False,
            'model_hyperparameters': config['model_hyperparameters']}}))

    def test_eight_runs_select_validation_winner_and_roundtrip_config(self):
        with patch('src.training.grid_search.subprocess.run', side_effect=self.fake_run) as process:
            result = self.run_search()
        self.assertEqual(process.call_count, 8)
        self.assertEqual(result['best_grid_index'], 2)
        self.assertEqual((self.output / 'best_checkpoint.pt').read_bytes(), b'checkpoint 2')
        resolved = resolve_config(self.output / 'best_config.json', None, 42)
        self.assertEqual(resolved['training']['seeds'], [42])
        with (self.output / 'grid_results.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len({r['run_dir'] for r in rows}), 8)
        self.assertTrue(all(r['status'] == 'PASS' for r in rows))
        with self.assertRaises(FileExistsError):
            self.run_search()

    def test_failed_run_never_selects_partial_grid(self):
        def fail(command, **kwargs):
            if command[command.index('--grid-index') + 1] == '2':
                raise subprocess.CalledProcessError(1, command)
            self.fake_run(command, **kwargs)
        with patch('src.training.grid_search.subprocess.run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_search()
        self.assertFalse((self.output / 'best_config.json').exists())
        failure = json.loads((self.output / 'failure.json').read_text())
        self.assertEqual(failure['completed_runs'], 2)
        with (self.output / 'grid_results.csv').open() as stream:
            self.assertEqual(list(csv.DictReader(stream))[-1]['status'], 'FAILED')

    def test_guards_reject_before_creating_outputs(self):
        for kwargs in ({'seed': 44}, {'eval_batch_size': 0}, {'optimizer': 'AdamW'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_search(**kwargs)
            self.assertFalse(self.output.exists())
        config = json.loads(self.config.read_text())
        config['training']['epochs'] = None
        self.config.write_text(json.dumps(config))
        with self.assertRaises(ValueError):
            self.run_search()
        self.assertFalse(self.output.exists())

    def test_invalid_metrics_are_not_selected(self):
        def invalid(command, **kwargs):
            self.fake_run(command, **kwargs)
            directory = Path(command[command.index('--output-dir') + 1])
            path = directory / 'summary.json'
            summary = json.loads(path.read_text())
            summary['best_validation_NDCG@10'] = float('nan')
            path.write_text(json.dumps(summary))
        with patch('src.training.grid_search.subprocess.run', side_effect=invalid):
            with self.assertRaises(ValueError):
                self.run_search()
        self.assertFalse((self.output / 'selection.json').exists())

    def test_optional_resume_cleanup_keeps_best_checkpoints(self):
        with patch('src.training.grid_search.subprocess.run', side_effect=self.fake_run):
            self.run_search(discard_last_checkpoint=True)
        self.assertEqual(len(list(self.output.glob('run_*/best.pt'))), 8)
        self.assertEqual(len(list(self.output.glob('run_*/last.pt'))), 0)
