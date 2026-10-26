import copy
import unittest
from pathlib import Path
from src.training.config import load_config, validate_config, expand_grid

class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / 'configs/config.json')

    def test_grid_is_complete_and_independent(self):
        runs = expand_grid(self.config)
        self.assertEqual(len({tuple(r['model_hyperparameters'].values()) for r in runs}), 8)
        runs[0]['training']['epochs'] = 7
        self.assertIsNone(runs[1]['training']['epochs'])
        self.assertIsNone(self.config['training']['epochs'])

    def test_template_cannot_run(self):
        with self.assertRaisesRegex(ValueError, 'Unconfirmed'):
            validate_config(self.config, require_ready=True)

    def test_rejects_invalid_settings(self):
        for section, key, value in [('training', 'num_heads', 3), ('training', 'epochs', True),
                                    ('training', 'weight_decay', float('nan')),
                                    ('training', 'seeds', [42, 42]),
                                    ('search', 'dropout', [0.2, 0.2]),
                                    ('evaluation', 'selection_split', 'test'),
                                    ('evaluation', 'sampled', True)]:
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[section][key] = value
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_ready_configuration(self):
        self.config['training'].update(batch_size=32, epochs=2, num_layers=2,
            num_heads=4, early_stopping_patience=1, weight_decay=0.0, seeds=[42])
        self.config['data']['processed_dir'] = '/tmp/example-processed'
        self.assertEqual(validate_config(self.config, require_ready=True), [])

if __name__ == '__main__':
    unittest.main()
