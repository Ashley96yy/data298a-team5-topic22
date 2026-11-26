"""Execute the notebook checkout cell against local Git history, without a GPU."""
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/model_experiments/13_item_text_embeddings_colab.ipynb'


class ItemTextNotebookTests(unittest.TestCase):
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
        (self.remote / 'README').write_text('Initial main\n')
        self.git('add', '.')
        self.git('commit', '-m', 'Initial main')
        self.git('checkout', '-b', 'feature')
        (self.remote / 'README').write_text('Pinned source\n')
        self.git('add', '.')
        self.git('commit', '-m', 'Item-text source')
        self.pin = self.git('rev-parse', 'HEAD')
        self.git('checkout', 'main')
        self.git('merge', '--ff-only', 'feature')
        self.git('branch', '-d', 'feature')
        (self.remote / 'README').write_text('Advanced main\n')
        self.git('add', '.')
        self.git('commit', '-m', 'Advance main')
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.cell = next(''.join(c['source']) for c in self.notebook['cells']
                         if c['cell_type'] == 'code' and '"clone"' in ''.join(c['source']))

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.remote), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()

    def checkout(self, project):
        scope = {'PROJECT': project, 'REPO_URL': str(self.remote), 'CODE_COMMIT': self.pin}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(self.cell, str(NOTEBOOK), 'exec'), scope)
        return scope['SOURCE_COMMIT']

    def test_deleted_branch_and_advanced_main_still_use_exact_pin(self):
        project = self.root / 'checkout'
        self.assertEqual(self.checkout(project), self.pin)
        self.assertNotEqual(self.pin, self.git('rev-parse', 'HEAD'))
        self.assertEqual(self.checkout(project), self.pin)
        self.assertEqual((project / 'README').read_text(), 'Pinned source\n')

    def test_wrong_existing_checkout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Existing checkout differs'):
            self.checkout(self.remote)

    def test_edited_existing_checkout_is_rejected(self):
        project = self.root / 'checkout'
        self.checkout(project)
        (project / 'README').write_text('Local edit\n')
        with self.assertRaisesRegex(ValueError, 'tracked changes'):
            self.checkout(project)

    def test_pin_contains_current_encoding_and_training_sources(self):
        settings = ast.parse(''.join(self.notebook['cells'][1]['source']))
        pin = next(ast.literal_eval(n.value) for n in settings.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                   and t.id == 'CODE_COMMIT' for t in n.targets))
        self.assertRegex(pin, r'^[a-f0-9]{40}$')
        for name in ['src/data/item_embeddings.py', 'src/models/text_sasrec.py',
                     'src/training/train_item_text.py', 'requirements-item-text.txt',
                     'src/models/distill.py', 'src/training/trainer.py']:
            with self.subTest(source=name):
                pinned = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{pin}:{name}'])
                self.assertEqual(pinned, (ROOT / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
