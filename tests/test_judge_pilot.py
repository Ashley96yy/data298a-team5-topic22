import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.judge.analyze_pilot import analyze
from src.evaluation.judge.run import CRITERIA, ROOT, load_config, payload, validate_cases


class JudgePilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'scores.jsonl'
        self.rows = []
        for task in ('explanation', 'profile'):
            for i, value in enumerate([1, 2, 2, 4]):
                scores = dict(zip(CRITERIA, [value, 5-value, value, 3]))
                if task == 'profile':
                    scores['relevance'] = value
                self.rows.append({'case_id': f'{task}_{i}', 'task': task,
                                  'samples': [scores.copy() for _ in range(20)],
                                  'scores': scores.copy()})
        self.metadata = dict(status='PASS', cases_expected=8, cases_completed=8,
                             mode='smoke', synthetic_smoke=True)

    def run_analysis(self):
        self.path.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        self.path.with_name('run_metadata.json').write_text(json.dumps(self.metadata))
        return analyze(self.path)

    def test_ties_inverse_relationship_and_task_separation(self):
        result = self.run_analysis()
        explanation = result['groups']['explanation']
        self.assertEqual(explanation['n_cases'], 4)
        self.assertEqual(len(explanation['pairs']), 6)
        self.assertAlmostEqual(explanation['pairs'][0]['rho'], -1)
        self.assertAlmostEqual(result['groups']['profile']['pairs'][0]['rho'], 1)
        self.assertAlmostEqual(explanation['pairs'][1]['rho'], 1)
        self.assertIsNone(explanation['pairs'][2]['rho'])
        self.assertEqual(explanation['pairs'][2]['undefined_reason'], 'constant_criterion')
        json.dumps(result, allow_nan=False)

    def test_nontrivial_ties_match_hand_calculated_spearman(self):
        # ranks x=[1,2.5,2.5,4], y=[2.5,1,2.5,4]; covariance/variance=2.25/4.5.
        for row, y in zip(self.rows[:4], [2, 1, 2, 4]):
            row['scores']['relevance'] = y
            for sample in row['samples']:
                sample['relevance'] = y
        self.assertAlmostEqual(self.run_analysis()['groups']['explanation']['pairs'][0]['rho'], .5)

    def test_small_or_missing_cohort_is_undefined(self):
        self.rows = self.rows[:2]
        self.metadata.update(cases_expected=2, cases_completed=2)
        result = self.run_analysis()
        for group in result['groups'].values():
            for pair in group['pairs']:
                self.assertIsNone(pair['rho'])
                self.assertEqual(pair['undefined_reason'], 'fewer_than_3_cases')

    def test_invalid_incomplete_or_modified_scores_are_rejected(self):
        original_rows, original_meta = copy.deepcopy(self.rows), self.metadata.copy()
        for mutation in ('fail', 'count', 'duplicate', 'samples', 'mean', 'nan', 'bool', 'task', 'mode'):
            with self.subTest(mutation=mutation):
                self.rows, self.metadata = copy.deepcopy(original_rows), original_meta.copy()
                if mutation == 'fail': self.metadata['status'] = 'FAIL'
                if mutation == 'count': self.metadata['cases_completed'] = 7
                if mutation == 'duplicate': self.rows[1]['case_id'] = self.rows[0]['case_id']
                if mutation == 'samples': self.rows[0]['samples'].pop()
                if mutation == 'mean': self.rows[0]['scores']['faithfulness'] = 3
                if mutation == 'nan': self.rows[0]['scores']['faithfulness'] = float('nan')
                if mutation == 'bool': self.rows[0]['samples'][0]['faithfulness'] = True
                if mutation == 'task': self.rows[0]['task'] = 'unknown'
                if mutation == 'mode': self.metadata['synthetic_smoke'] = False
                with self.assertRaises(ValueError): self.run_analysis()

    def test_negative_controls_valid_only_for_smoke_and_no_expected_labels_sent(self):
        path = ROOT / 'tests/fixtures/judge/pilot_negative_cases.jsonl'
        cases = [json.loads(line) for line in path.read_text().splitlines()]
        config = load_config()
        validate_cases(cases, config, 'smoke')
        self.assertEqual(len(cases), 4)
        self.assertEqual({c['task'] for c in cases}, {'profile', 'explanation'})
        with self.assertRaises(ValueError): validate_cases(cases, config, 'evaluation')
        for case in cases:
            material = json.loads(payload(case, config, 'rubric', 2026)['messages'][1]['content'])
            self.assertNotIn('case_id', material)
            self.assertNotIn('synthetic', material)


if __name__ == '__main__':
    unittest.main()
