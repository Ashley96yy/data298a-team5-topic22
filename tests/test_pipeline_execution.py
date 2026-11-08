"""Execute real nbconvert processes on tiny stand-in notebooks, not the Amazon ETL."""
import json
from pathlib import Path
import tempfile
import unittest

try:
    import nbformat
except ImportError:
    nbformat = None

from pipeline.run_pipeline import NOTEBOOKS, RECORD_FILES, run_pipeline
from src.serving.monitor import find_run_dir


@unittest.skipUnless(nbformat is not None, 'needs pipeline/requirements.txt')
class NotebookExecutionTests(unittest.TestCase):
    def test_seven_notebooks_execute_and_the_monitor_finds_the_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebooks = root / 'notebooks'
            notebooks.mkdir()
            for index, name in enumerate(NOTEBOOKS):
                source = "import os,json\nfrom pathlib import Path\nbase=Path(os.environ['RECSYS_BASE'])\n"
                if index == 0:
                    source += "(base/'config.json').write_text(json.dumps({'include_review_text':True}))\n"
                if index == 1:
                    source += "assert json.loads((base/'config.json').read_text())['include_review_text'] is False\n"
                if index == len(NOTEBOOKS) - 1:
                    source += 'names=' + repr([n for n, _ in RECORD_FILES]) + '\n'
                    source += ("for name in names:\n p=base/name\n p.parent.mkdir(parents=True,exist_ok=True)\n"
                               " if not p.exists(): p.write_text('{}' if p.suffix=='.json' else 'record')\n"
                               "(base/'reports').mkdir(exist_ok=True)\n")
                nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]),
                               notebooks / (name + '.ipynb'))
            record = root / 'pipeline_runs' / '20261107T000000Z'
            status = run_pipeline(notebooks, root / 'data', record, skip_reviews=True, timeout=60)
            self.assertEqual(status['status'], 'PASS', status)
            self.assertEqual(find_run_dir(root), record)
            self.assertEqual(len(list((record / 'executed_notebooks').glob('*.ipynb'))), 7)


if __name__ == '__main__':
    unittest.main()
