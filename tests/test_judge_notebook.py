"""Exercise the actual checkout cell against local Git history; no network/GPU."""
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/model_experiments/15_ministral_judge_colab.ipynb'


class JudgeNotebookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.remote = self.root / 'remote'
        self.remote.mkdir()
        self.git('init', '-b', 'main')
        self.git('config', 'user.name', 'Notebook Test')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'commit.gpgsign', 'false')
        (self.remote / 'README').write_text('main\n')
        self.git('add', '.')
        self.git('commit', '-m', 'Initial main')
        self.git('checkout', '-b', 'abhinita/ministral-judge')
        config = self.remote / 'configs/judge/ministral.json'
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({'model_id': 'fixture-model'}))
        self.git('add', '.')
        self.git('commit', '-m', 'Pinned judge code')
        self.pin = self.git('rev-parse', 'HEAD')
        self.git('checkout', 'main')
        self.git('merge', '--ff-only', 'abhinita/ministral-judge')
        self.git('branch', '-d', 'abhinita/ministral-judge')
        (self.remote / 'README').write_text('Later main\n')
        self.git('add', '.')
        self.git('commit', '-m', 'Advance main past the pin')
        notebook = json.loads(NOTEBOOK.read_text())
        self.cell = next(''.join(c['source']) for c in notebook['cells']
                         if c['cell_type'] == 'code' and '"clone"' in ''.join(c['source']))

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.remote), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()

    def checkout(self, project):
        scope = {'PROJECT': project, 'REPO_URL': str(self.remote),
                 'CODE_COMMIT': self.pin, 'subprocess': subprocess, 'json': json}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(self.cell, str(NOTEBOOK), 'exec'), scope)

    def test_checkout_survives_deleted_feature_branch_and_detaches_at_pin(self):
        self.assertNotIn('abhinita/ministral-judge', self.git('branch', '--list'))
        project = self.root / 'checkout'
        self.checkout(project)
        head = subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip()
        self.assertEqual(head, self.pin)
        self.assertNotEqual(head, self.git('rev-parse', 'HEAD'))
        symbolic = subprocess.run(['git', '-C', str(project), 'symbolic-ref', '-q', 'HEAD'], capture_output=True)
        self.assertEqual(symbolic.returncode, 1)
        self.checkout(project)  # An existing checkout at the exact pin is accepted.

    def test_existing_wrong_checkout_is_rejected(self):
        # The fixture's main has advanced past the pinned commit.
        with self.assertRaisesRegex(AssertionError, 'Existing checkout differs'):
            self.checkout(self.remote)

    def test_notebook_pin_contains_current_runner(self):
        notebook = json.loads(NOTEBOOK.read_text())
        settings = ast.parse(''.join(notebook['cells'][1]['source']))
        pin = next(ast.literal_eval(node.value) for node in settings.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'CODE_COMMIT' for t in node.targets))
        self.assertRegex(pin, r'^[a-f0-9]{40}$')
        for path in ('src/evaluation/judge/run.py', 'configs/judge/rubric.txt',
                     'src/evaluation/judge/analyze_pilot.py',
                     'tests/fixtures/judge/pilot_negative_cases.jsonl'):
            with self.subTest(path=path):
                pinned = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{pin}:{path}'])
                self.assertEqual(pinned, (ROOT / path).read_bytes())


if __name__ == '__main__':
    unittest.main()
