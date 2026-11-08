"""The Colab environment gate rejects runtime drift before launching training."""
import copy
import unittest
from src.training.environment_check import environment_mismatches


class EnvironmentCheckTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'versions': {'python': '3.13.16', 'torch': '2.14.1+cu130',
                         'numpy': '2.2.6', 'pandas': '2.3.2', 'pyarrow': '21.0.0'},
                         'require_cuda': True}
        self.actual = {'versions': dict(self.expected['versions']), 'cuda_available': True}

    def test_declared_runtime_passes(self):
        self.assertEqual(environment_mismatches(self.actual, self.expected), [])

    def test_version_drift_and_missing_packages_fail(self):
        for name in self.expected['versions']:
            for missing in (False, True):
                with self.subTest(package=name, missing=missing):
                    actual = copy.deepcopy(self.actual)
                    if missing:
                        del actual['versions'][name]
                    else:
                        actual['versions'][name] = 'different'
                    failures = environment_mismatches(actual, self.expected)
                    self.assertEqual(len(failures), 1)
                    self.assertIn(name, failures[0])

    def test_missing_cuda_and_incomplete_target_fail(self):
        self.actual['cuda_available'] = False
        self.assertIn('CUDA', environment_mismatches(self.actual, self.expected)[0])
        del self.expected['versions']['torch']
        with self.assertRaises(ValueError):
            environment_mismatches(self.actual, self.expected)
