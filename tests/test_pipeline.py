import json
import tempfile
import unittest
from pathlib import Path

from pipeline.run_pipeline import NOTEBOOKS, RECORD_FILES, collect_record, run_pipeline, status_markdown
from src.serving.monitor import find_run_dir, load_run, summary


def fake_data_folder(base):
    """What the seven notebooks would leave behind (tiny stand-ins)."""
    base = Path(base)
    for src, _ in RECORD_FILES:
        p = base / src
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():                                   # keep a config.json the notebooks already wrote
            p.write_text('{}' if src.endswith('.json') else 'text')
    (base / 'reports' / 'figures').mkdir(parents=True, exist_ok=True)
    (base / 'reports' / 'figures' / 'a.png').write_bytes(b'png')
    (base / 'reports' / 'dataset_statistics.csv').write_text('stage,users,items,interactions,avg_seq_len\nraw,1,1,1,1\n')
    (base / 'reports' / 'sasrec_smoke_test.json').write_text('{"NDCG@10": 1.66}')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root / 'data'
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def executor(self, fail_on=None, write_config=True):
        def run(notebook, executed, env, timeout):
            name = notebook.stem
            self.calls.append((name, env['RECSYS_BASE'], timeout))
            if name == fail_on:
                raise RuntimeError('boom')
            if name == NOTEBOOKS[0] and write_config:
                self.base.mkdir(parents=True, exist_ok=True)
                (self.base / 'config.json').write_text(json.dumps({'include_review_text': True, 'seed': 42}))
            if name == NOTEBOOKS[-1]:
                fake_data_folder(self.base)
        return run

    def test_success_runs_all_notebooks_in_order_and_writes_a_readable_record(self):
        out = self.root / 'pipeline_runs' / '20261107T000000Z'
        status = run_pipeline(self.root / 'nb', self.base, out, executor=self.executor())
        self.assertEqual(status['status'], 'PASS')
        self.assertEqual([c[0] for c in self.calls], NOTEBOOKS)
        self.assertEqual(self.calls[0][1], str(self.base.resolve()))
        for _, dst in RECORD_FILES:
            self.assertTrue((out / dst).exists(), dst)
        self.assertTrue((out / 'reports' / 'figures' / 'a.png').exists())
        self.assertFalse((out / 'reports' / 'sasrec_smoke_test.json').exists())     # invalid file is never kept
        self.assertEqual(find_run_dir(self.root), out)
        self.assertEqual(summary(load_run(out))['status'], 'PASS')

    def test_skip_reviews_patches_the_config_after_notebook_01(self):
        run_pipeline(self.root / 'nb', self.base, self.root / 'run', skip_reviews=True, executor=self.executor())
        config = json.loads((self.base / 'config.json').read_text())
        self.assertFalse(config['include_review_text'])
        self.assertEqual(config['seed'], 42)                                        # other settings untouched

    def test_a_failing_notebook_stops_the_run_and_is_recorded(self):
        out = self.root / 'run'
        status = run_pipeline(self.root / 'nb', self.base, out, executor=self.executor(fail_on='03_cleaning'))
        self.assertEqual(status['status'], 'FAILED')
        self.assertEqual(status['failed_step'], '03_cleaning')
        self.assertEqual([c[0] for c in self.calls], NOTEBOOKS[:3])                 # nothing runs after the failure
        saved = json.loads((out / 'run_status.json').read_text())
        self.assertEqual(saved['status'], 'FAILED')
        self.assertEqual(saved['notebooks'][-1]['status'], 'FAILED')
        self.assertIn('boom', saved['error'])

    def test_missing_config_after_notebook_01_fails_loudly_when_skipping_reviews(self):
        status = run_pipeline(self.root / 'nb', self.base, self.root / 'run', skip_reviews=True,
                              executor=self.executor(write_config=False))
        self.assertEqual((status['status'], status['failed_step']), ('FAILED', 'patch config.json'))

    def test_incomplete_record_is_a_failure_not_a_pass(self):
        def executor(notebook, executed, env, timeout):
            pass                                                                     # produces nothing
        status = run_pipeline(self.root / 'nb', self.base, self.root / 'run', executor=executor)
        self.assertEqual((status['status'], status['failed_step']), ('FAILED', 'collect run record'))

    def test_refuses_to_reuse_a_run_folder(self):
        (self.root / 'run').mkdir()
        with self.assertRaises(FileExistsError):
            run_pipeline(self.root / 'nb', self.base, self.root / 'run', executor=self.executor())

    def test_collect_record_reports_every_missing_file(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            (self.root / 'out').mkdir()
            collect_record(self.base, self.root / 'out')
        self.assertIn('manifest.json', str(ctx.exception))

    def test_markdown_summary_lists_every_notebook(self):
        status = run_pipeline(self.root / 'nb', self.base, self.root / 'run', executor=self.executor())
        text = status_markdown(status)
        for name in NOTEBOOKS:
            self.assertIn(name, text)


if __name__ == '__main__':
    unittest.main()
