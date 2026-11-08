"""Optional real-checkpoint UI test; only inference, no benchmark metrics."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = os.environ.get('DEMO_TEST_SASREC_CHECKPOINT')
CHECKSUM = os.environ.get('DEMO_TEST_SASREC_SHA256')


@unittest.skipUnless(AppTest is not None and CHECKPOINT and CHECKSUM,
                     'needs Streamlit, built artifacts and a trusted demo checkpoint')
class SASRecPageTests(unittest.TestCase):
    def test_live_model_is_selectable_and_serves_a_real_example(self):
        with patch.dict(os.environ, {'SASREC_CHECKPOINT': CHECKPOINT,
                                     'SASREC_CHECKPOINT_SHA256': CHECKSUM}):
            st.cache_resource.clear()
            try:
                at = AppTest.from_file(str(ROOT / 'app/Recommendations.py'), default_timeout=60).run()
                self.assertFalse(at.exception)
                self.assertIn('sasrec', at.radio[0].options)
                at.radio[0].set_value('sasrec')
                [b for b in at.button if b.label == 'Load a real example user'][0].click().run()
                self.assertFalse(at.exception)
                self.assertGreater(len(at.session_state['history']), 0)
                self.assertEqual(len(at.dataframe[0].value), 10)
                self.assertTrue(any('Method: sasrec' in c.value for c in at.caption))
            finally:
                st.cache_resource.clear()


if __name__ == '__main__':
    unittest.main()
