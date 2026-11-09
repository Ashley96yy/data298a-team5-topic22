import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'app' / 'artifacts'

try:
    from streamlit.testing.v1 import AppTest
except ImportError:                                           # streamlit is an optional dependency
    AppTest = None


@unittest.skipUnless(AppTest is not None and (ARTIFACTS / 'items.parquet').exists(), 'needs streamlit and built app/artifacts')
class AppTests(unittest.TestCase):
    def run_app(self):
        at = AppTest.from_file(str(ROOT / 'app' / 'Recommendations.py'), default_timeout=60)
        at.run()
        return at

    def test_starts_without_exceptions_and_shows_cold_start_recommendations(self):
        at = self.run_app()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.dataframe), 1)
        self.assertEqual(len(at.dataframe[0].value), 10)

    def test_adding_a_product_changes_the_recommendations(self):
        at = self.run_app()
        before = at.dataframe[0].value['title'].tolist()
        at.text_input[0].set_value('serum').run()
        add_buttons = [b for b in at.button if b.label == 'Add']
        self.assertTrue(add_buttons)
        add_buttons[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state['history']), 1)
        after = at.dataframe[0].value['title'].tolist()
        self.assertNotEqual(before, after)

    def test_loading_a_real_example_user_fills_the_history(self):
        at = self.run_app()
        [b for b in at.button if b.label == 'Load a real example user'][0].click().run()
        self.assertFalse(at.exception)
        self.assertGreaterEqual(len(at.session_state['history']), 3)


RUN_RECORD = ROOT / 'docs' / 'data_run' / 'manifest.json'


@unittest.skipUnless(AppTest is not None and RUN_RECORD.exists(), 'needs streamlit and the committed run record')
class MonitorPageTests(unittest.TestCase):
    def test_monitor_page_renders_real_numbers(self):
        at = AppTest.from_file(str(ROOT / 'app' / 'pages' / '1_Pipeline_monitor.py'), default_timeout=60)
        at.run()
        self.assertFalse(at.exception)
        labels = [m.label for m in at.metric]
        self.assertIn('Leakage checks passed', labels)
        self.assertGreaterEqual(len(at.dataframe), 3)


if __name__ == '__main__':
    unittest.main()
