"""Significance tests on synthetic ranks: formulas, test behaviour, Holm, seed averaging and integrity checks."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation import significance as S


def rec_eval_ndcg(ranks, k=10):
    """rec_eval.metrics_from_ranks NDCG@k, written out independently."""
    ranks = np.asarray(ranks, dtype=float)
    return float(np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0).mean())


def write_run(directory, seed, users, ranks, recorded=None, tie='average_rank_then_metrics'):
    directory.mkdir(parents=True)
    np.savez(directory / 'test_ranks.npz', user_idx=np.asarray(users), ranks=np.asarray(ranks, dtype=float),
             topk=np.zeros((len(users), 20), dtype=np.int64))
    record = {'seed': seed, 'ranks_file': 'test_ranks.npz', 'n_test_cases': len(users),
              'tie_policy': {'name': tie},
              'test_full': {'NDCG@10': rec_eval_ndcg(ranks) if recorded is None else recorded}}
    (directory / 'test_evaluation.json').write_text(json.dumps(record))
    return directory


class StatisticsTests(unittest.TestCase):
    def test_per_user_metrics_match_rec_eval_including_tied_ranks(self):
        ranks = np.array([1, 2, 3.5, 10, 10.5, 11, 400])
        m = S.per_user_metrics(ranks)
        self.assertAlmostEqual(m['NDCG@10'].mean(), rec_eval_ndcg(ranks), places=15)
        self.assertEqual(m['Recall@10'].tolist(), [1, 1, 1, 1, 0, 0, 0])
        self.assertAlmostEqual(m['MRR'][2], 1 / 3.5)
        with self.assertRaises(ValueError):
            S.per_user_metrics([0.5, 2])

    def test_identical_models_give_p_one_and_zero_interval(self):
        p, observed = S.randomization_test(np.zeros(500), n_permutations=999)
        self.assertEqual((p, observed), (1.0, 0.0))
        self.assertEqual(S.bootstrap_interval(np.zeros(500), n_resamples=999), (0.0, 0.0))

    def test_clear_difference_gives_smallest_p_and_interval_excludes_zero(self):
        diff = np.random.default_rng(0).normal(0.05, 0.1, size=2000)
        p, _ = S.randomization_test(diff, n_permutations=999)
        self.assertEqual(p, 1 / 1000)
        low, high = S.bootstrap_interval(diff, n_resamples=999)
        self.assertGreater(low, 0)
        self.assertLess(low, diff.mean())
        self.assertGreater(high, diff.mean())

    def test_no_real_difference_is_not_significant(self):
        diff = np.random.default_rng(1).normal(0, 0.1, size=2000)
        p, _ = S.randomization_test(diff, n_permutations=999)
        self.assertGreater(p, 0.05)

    def test_test_is_two_sided_and_reproducible(self):
        diff = np.random.default_rng(2).normal(0.004, 0.1, size=1000)
        first = S.randomization_test(diff, 999, S._streams()[0])[0]
        self.assertEqual(first, S.randomization_test(diff, 999, S._streams()[0])[0])
        self.assertEqual(first, S.randomization_test(-diff, 999, S._streams()[0])[0])

    def test_holm_adjustment(self):
        np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        np.testing.assert_allclose(S.holm([0.5, 0.6]), [1.0, 1.0])

    def test_compare_reports_primary_and_secondary(self):
        rng = np.random.default_rng(3)
        a = S.per_user_metrics(rng.integers(1, 30, size=3000))
        b = S.per_user_metrics(rng.integers(1, 60, size=3000))
        result = S.compare(a, b, n_permutations=499, n_resamples=499)
        self.assertAlmostEqual(result['difference'], result['mean_a'] - result['mean_b'])
        self.assertLess(result['ci95'][0], result['ci95'][1])
        self.assertEqual(set(result['secondary']), {'Recall@10', 'MRR'})


class LoadingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.users = np.arange(50)
        rng = np.random.default_rng(4)
        self.ranks = {seed: rng.integers(1, 40, size=50).astype(float) for seed in S.SEEDS}

    def runs(self, prefix='m0', **override):
        dirs = []
        for seed in S.SEEDS:
            ranks = override.get(seed, self.ranks[seed])
            dirs.append(write_run(self.root / f'{prefix}_{seed}', seed, self.users, ranks))
        return dirs

    def test_seed_average_is_per_user_mean(self):
        users, metrics, source = S.load_model(self.runs())
        expected = np.mean([S.per_user_metrics(self.ranks[s])['NDCG@10'] for s in S.SEEDS], axis=0)
        np.testing.assert_allclose(metrics['NDCG@10'], expected)
        self.assertEqual(source['seeds'], [42, 43, 44])

    def test_users_are_sorted_before_averaging(self):
        dirs = self.runs()
        shuffled = np.random.default_rng(5).permutation(50)
        write_run(self.root / 'shuffled_43', 43, self.users[shuffled], self.ranks[43][shuffled])
        _, metrics, _ = S.load_model([dirs[0], self.root / 'shuffled_43', dirs[2]])
        _, reference, _ = S.load_model(dirs)
        np.testing.assert_allclose(metrics['NDCG@10'], reference['NDCG@10'])

    def test_ranks_that_do_not_reproduce_the_record_are_refused(self):
        dirs = self.runs()
        record = json.loads((dirs[1] / 'test_evaluation.json').read_text())
        record['test_full']['NDCG@10'] += 0.001
        (dirs[1] / 'test_evaluation.json').write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'do not reproduce'):
            S.load_model(dirs)

    def test_wrong_or_missing_seeds_are_refused(self):
        dirs = self.runs()
        with self.assertRaisesRegex(ValueError, 'committed seeds'):
            S.load_model(dirs[:2])
        with self.assertRaisesRegex(ValueError, 'twice'):
            S.load_model([dirs[0], dirs[0], dirs[1]])

    def test_different_users_are_refused(self):
        dirs = self.runs()
        write_run(self.root / 'other_44', 44, self.users + 1, self.ranks[44])
        with self.assertRaisesRegex(ValueError, 'users differ'):
            S.load_model([dirs[0], dirs[1], self.root / 'other_44'])

    def test_other_tie_policy_is_refused(self):
        dirs = self.runs()
        write_run(self.root / 'tie_44', 44, self.users, self.ranks[44], tie='expected_metrics')
        with self.assertRaisesRegex(ValueError, 'tie policy'):
            S.load_model([dirs[0], dirs[1], self.root / 'tie_44'])

    def test_align_reorders_to_reference_users(self):
        users = np.array([3, 1, 2])
        metrics = {'NDCG@10': np.array([0.3, 0.1, 0.2])}
        np.testing.assert_allclose(S.align(users, metrics, np.array([1, 2, 3]))['NDCG@10'], [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            S.align(users, metrics, np.array([1, 2, 4]))

    def write_baselines(self):
        directory = self.root / 'baselines'
        directory.mkdir()
        rng = np.random.default_rng(6)
        arrays = {name: rng.integers(1, 200, size=50).astype(float) for name in S.BASELINES}
        np.savez(directory / 'baseline_test_ranks.npz', user_idx=self.users, **arrays)
        record = {'ranks_sha256': S.digest(directory / 'baseline_test_ranks.npz'),
                  'provenance': {'verified_files': {}}}
        (directory / 'baseline_ranks.json').write_text(json.dumps(record))
        return directory

    def test_compare_command_writes_table_with_holm_over_present_comparisons(self):
        baselines = self.write_baselines()
        model = 'Model 0=' + ','.join(str(d) for d in self.runs())
        out = self.root / 'out'
        with contextlib.redirect_stdout(io.StringIO()):
            S.main(['compare', '--model', model, '--baseline-ranks', str(baselines), '--output', str(out)])
        payload = json.loads((out / 'significance.json').read_text())
        rows = payload['comparisons']
        self.assertEqual([(r['a'], r['b']) for r in rows], [('Model 0', b) for b in S.BASELINES])
        np.testing.assert_allclose([r['p_holm'] for r in rows], S.holm([r['p_value'] for r in rows]))
        self.assertIn('Model 1 vs Model 0', payload['not_yet_available'])
        self.assertIn('Model 0 vs popularity', (out / 'significance.md').read_text())
        with self.assertRaises(SystemExit):
            S.main(['compare', '--model', model, '--baseline-ranks', str(baselines), '--output', str(out)])

    def test_changed_baseline_file_is_refused(self):
        baselines = self.write_baselines()
        with np.load(baselines / 'baseline_test_ranks.npz') as saved:
            arrays = {k: saved[k] for k in saved.files}
        arrays['popularity'] = arrays['popularity'] + 1
        np.savez(baselines / 'baseline_test_ranks.npz', **arrays)
        with self.assertRaisesRegex(ValueError, 'changed'):
            S.load_baselines(baselines)


if __name__ == '__main__':
    unittest.main()
