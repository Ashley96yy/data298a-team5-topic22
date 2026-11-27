import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.guardrails import basic
from src.guardrails.basic import (
    MAX_HISTORY, MAX_K, RejectionLog, guard_output, guarded_recommend, popularity_fallback,
    remove_seen, sanitize_output, validate_request,
)

N = 5
POP = np.array([0, 10, 8, 8, 3, 1])        # index 0 is padding; item 1 most popular, 2 and 3 tied


class ValidateRequestTests(unittest.TestCase):
    def test_accepts_clean_request(self):
        c = validate_request([1, 2, 3], 5, N, request_id='r1')
        self.assertTrue(c.ok)
        self.assertEqual((c.history, c.k, c.request_id, c.flags), ((1, 2, 3), 5, 'r1', ()))

    def test_numpy_integers_are_accepted_and_converted(self):
        c = validate_request(np.array([1, 2]), np.int64(3), N)
        self.assertTrue(c.ok)
        self.assertTrue(all(type(i) is int for i in c.history))

    def test_rejections_carry_one_reason_code(self):
        cases = [
            (('abc', 3), basic.HISTORY_NOT_A_LIST),
            ((None, 3), basic.HISTORY_NOT_A_LIST),
            (({'a': 1}, 3), basic.HISTORY_NOT_A_LIST),
            (([1, 2.0], 3), basic.HISTORY_ITEM_NOT_INTEGER),
            (([1, True], 3), basic.HISTORY_ITEM_NOT_INTEGER),
            (([1, '2'], 3), basic.HISTORY_ITEM_NOT_INTEGER),
            (([0], 3), basic.HISTORY_ITEM_OUT_OF_CATALOG),            # padding is not an item
            (([N + 1], 3), basic.HISTORY_ITEM_OUT_OF_CATALOG),
            (([-1], 3), basic.HISTORY_ITEM_OUT_OF_CATALOG),
            (([1], 0), basic.INVALID_K),
            (([1], MAX_K + 1), basic.INVALID_K),
            (([1], 2.5), basic.INVALID_K),
            (([1], True), basic.INVALID_K),
        ]
        for (history, k), code in cases:
            c = validate_request(history, k, N)
            self.assertFalse(c.ok, (history, k))
            self.assertEqual(c.reason_code, code, (history, k))

    def test_unknown_method_only_checked_when_methods_given(self):
        self.assertTrue(validate_request([1], 3, N, method='whatever').ok)
        self.assertEqual(validate_request([1], 3, N, method='x', methods=['a']).reason_code, basic.UNKNOWN_METHOD)
        self.assertTrue(validate_request([1], 3, N, method='a', methods=['a']).ok)

    def test_empty_history_is_allowed_and_flagged(self):
        c = validate_request([], 3, N)
        self.assertTrue(c.ok)
        self.assertEqual(c.flags, (basic.EMPTY_HISTORY,))

    def test_long_history_keeps_most_recent_items(self):
        c = validate_request([1] * 10 + [2] * MAX_HISTORY, 3, N)
        self.assertEqual(c.history, (2,) * MAX_HISTORY)
        self.assertIn(basic.HISTORY_TRUNCATED, c.flags)

    def test_bad_request_id_is_replaced_not_echoed(self):
        c = validate_request([1], 3, N, request_id='bad id <script>')
        self.assertEqual(c.request_id, 'invalid-request-id')
        self.assertEqual(len(validate_request([1], 3, N).request_id), 32)    # generated when missing


class OutputTests(unittest.TestCase):
    def test_remove_seen_keeps_order(self):
        self.assertEqual(remove_seen([5, 1, 4, 2], [1, 2]), [5, 4])

    def test_sanitize_drops_bad_items_and_reports_codes(self):
        out, codes = sanitize_output([2, 2, 9, 1, 'x', 0, 3], [1], N, 10)
        self.assertEqual(out, [2, 3])
        self.assertEqual(set(codes), {basic.OUTPUT_DUPLICATE, basic.OUTPUT_OUT_OF_CATALOG,
                                      basic.OUTPUT_SEEN_ITEM, basic.OUTPUT_INVALID_ITEM})

    def test_sanitize_truncates_to_k(self):
        out, codes = sanitize_output([1, 2, 3, 4], [], N, 2)
        self.assertEqual(out, [1, 2])
        self.assertEqual(codes, [basic.OUTPUT_TRUNCATED])

    def test_sanitize_rejects_non_lists(self):
        for bad in (None, 'abc', 5):
            self.assertEqual(sanitize_output(bad, [], N, 3), ([], [basic.OUTPUT_NOT_A_LIST]))

    def test_popularity_fallback_order_ties_and_exclusions(self):
        self.assertEqual(popularity_fallback(POP, [], 3), [1, 2, 3])          # tie 2/3 -> lower id first
        self.assertEqual(popularity_fallback(POP, [1], 3), [2, 3, 4])         # seen item skipped
        self.assertEqual(popularity_fallback(POP, [1], 3, exclude=[2]), [3, 4, 5])
        self.assertNotIn(0, popularity_fallback(POP, [], 10))                 # padding never returned
        self.assertEqual(len(popularity_fallback(POP, [1, 2], 10)), 3)        # capped by what is left

    def test_guard_output_passes_a_good_list_through(self):
        r = guard_output([4, 5], [1], POP, 2)
        self.assertEqual((r.items, r.reason_codes, r.used_fallback), ([4, 5], [], False))

    def test_guard_output_fills_short_lists_from_popularity(self):
        r = guard_output([5, 1], [1], POP, 3)                 # 1 is seen, so only [5] is valid
        self.assertEqual(r.items, [5, 2, 3])
        self.assertTrue(r.used_fallback)
        self.assertIn(basic.FALLBACK_FILL, r.reason_codes)
        self.assertEqual(len(set(r.items)), 3)

    def test_guard_output_replaces_an_unusable_list(self):
        r = guard_output(None, [1], POP, 2)
        self.assertEqual(r.items, [2, 3])
        self.assertIn(basic.OUTPUT_NOT_A_LIST, r.reason_codes)


class RejectionLogTests(unittest.TestCase):
    def test_log_holds_only_id_code_and_time(self):
        log = RejectionLog()
        log.record('abc', basic.INVALID_K)
        self.assertEqual(set(log.records[0]), {'request_id', 'reason_code', 'time_utc'})
        self.assertEqual(log.counts(), {basic.INVALID_K: 1})

    def test_log_refuses_free_text(self):
        with self.assertRaises(ValueError):
            RejectionLog().record('abc', 'user 17 bought item 99')

    def test_log_never_contains_history_or_titles(self):
        log = RejectionLog()
        guarded_recommend(lambda h, m, k: [], POP, [1, 2, 999], 3, request_id='r9', log=log)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'rej.jsonl'
            log.write_jsonl(path)
            text = path.read_text()
        self.assertNotIn('999', text)
        self.assertEqual(json.loads(text)['reason_code'], basic.HISTORY_ITEM_OUT_OF_CATALOG)


class GuardedRecommendTests(unittest.TestCase):
    def test_happy_path(self):
        r = guarded_recommend(lambda h, m, k: [4, 5], POP, [1], 2, request_id='ok1')
        self.assertTrue(r['ok'])
        self.assertEqual(r['items'], [4, 5])
        self.assertFalse(r['used_fallback'])

    def test_rejected_request_returns_no_items_and_is_logged(self):
        log = RejectionLog()
        r = guarded_recommend(lambda h, m, k: [1], POP, [99], 2, request_id='bad1', log=log)
        self.assertFalse(r['ok'])
        self.assertEqual(r['items'], [])
        self.assertEqual(log.records[0]['request_id'], 'bad1')

    def test_model_returning_seen_items_is_repaired(self):
        r = guarded_recommend(lambda h, m, k: [1, 1, 4], POP, [1], 3)
        self.assertEqual(r['items'][0], 4)
        self.assertNotIn(1, r['items'])
        self.assertEqual(len(r['items']), 3)
        self.assertTrue(r['used_fallback'])

    def test_model_crash_falls_back_to_popularity(self):
        def boom(h, m, k):
            raise RuntimeError('checkpoint missing at /secret/path')
        log = RejectionLog()
        r = guarded_recommend(boom, POP, [1], 2, log=log)
        self.assertTrue(r['ok'])
        self.assertEqual(r['items'], [2, 3])
        self.assertIn(basic.RECOMMENDER_ERROR, r['reason_codes'])
        self.assertNotIn('secret', json.dumps(log.records))                 # exception text is never stored

    def test_cold_start_uses_popularity_without_calling_the_model(self):
        called = []
        r = guarded_recommend(lambda h, m, k: called.append(1) or [5], POP, [], 2)
        self.assertEqual(r['items'], [1, 2])
        self.assertEqual(called, [])
        self.assertIn(basic.EMPTY_HISTORY, r['reason_codes'])
        self.assertNotIn(basic.FALLBACK_FILL, r['reason_codes'])

    def test_flags_are_not_written_to_the_rejection_log(self):
        log = RejectionLog()
        guarded_recommend(lambda h, m, k: [], POP, [], 2, log=log)
        self.assertEqual(log.records, [])

    def test_works_with_the_real_recommender_output_shape(self):
        # Recommender.recommend returns a DataFrame; callers pass the item_idx column as a list.
        r = guarded_recommend(lambda h, m, k: list(np.array([3, 4, 5])), POP, [1, 2], 3)
        self.assertEqual(r['items'], [3, 4, 5])


class ReviewRegressionTests(unittest.TestCase):
    """Cases from the PR #26 review: truncation must not hide seen items, odd containers must not raise."""
    N_BIG = 100
    POP_BIG = np.concatenate([[0], np.arange(100, 0, -1)])         # item 1 is the most popular, then 2, 3, ...

    def test_full_history_is_kept_next_to_the_model_window(self):
        c = validate_request(list(range(1, 52)), 3, self.N_BIG)    # 51 items, item 1 is the oldest
        self.assertTrue(c.ok)
        self.assertEqual(len(c.history), MAX_HISTORY)
        self.assertEqual(c.history, tuple(range(2, 52)))
        self.assertEqual(c.full_history, tuple(range(1, 52)))
        self.assertIn(basic.HISTORY_TRUNCATED, c.flags)

    def test_old_seen_item_is_not_recommended_when_the_model_returns_it(self):
        r = guarded_recommend(lambda h, m, k: [1, 60], self.POP_BIG, list(range(1, 52)), 3)
        self.assertNotIn(1, r['items'])
        self.assertEqual(r['items'][0], 60)
        self.assertEqual(len(r['items']), 3)

    def test_old_seen_item_is_not_used_by_the_popularity_fallback(self):
        r = guarded_recommend(lambda h, m, k: [], self.POP_BIG, list(range(1, 52)), 3)
        self.assertEqual(r['items'], [52, 53, 54])                 # items 1..51 are all seen

    def test_model_only_sees_the_most_recent_window(self):
        seen_by_model = []
        guarded_recommend(lambda h, m, k: seen_by_model.append(list(h)) or [90], self.POP_BIG, list(range(1, 52)), 1)
        self.assertEqual(seen_by_model, [list(range(2, 52))])

    def test_zero_dimensional_array_as_history_is_rejected_not_raised(self):
        for bad in (np.array(5), np.array([[1, 2], [3, 4]])):
            c = validate_request(bad, 3, self.N_BIG)
            self.assertFalse(c.ok)
            self.assertEqual(c.reason_code, basic.HISTORY_NOT_A_LIST)

    def test_zero_dimensional_array_from_the_scorer_uses_the_popularity_fallback(self):
        r = guarded_recommend(lambda h, m, k: np.array(5), self.POP_BIG, [2, 3], 3)
        self.assertTrue(r['ok'])
        self.assertEqual(r['items'], [1, 4, 5])
        self.assertIn(basic.OUTPUT_NOT_A_LIST, r['reason_codes'])
        self.assertTrue(r['used_fallback'])

    def test_other_unsupported_shapes_fall_back_instead_of_raising(self):
        for bad in (np.array([[7, 8]]), iter([7, 8]), {7, 8}, 7, 'ab', b'ab', {7: 8}):
            r = guarded_recommend(lambda h, m, k, bad=bad: bad, self.POP_BIG, [2], 2)
            self.assertTrue(r['ok'], bad)
            self.assertEqual(r['items'], [1, 3], bad)
            self.assertIn(basic.OUTPUT_NOT_A_LIST, r['reason_codes'], bad)
        for bad in (iter([1, 2]), {1, 2}, {1: 2}, 'ab', 5.0, None):
            self.assertEqual(validate_request(bad, 3, self.N_BIG).reason_code, basic.HISTORY_NOT_A_LIST, bad)

    def test_supported_containers_are_accepted_for_input_and_output(self):
        for hist in ([2, 3], (2, 3), np.array([2, 3])):
            self.assertTrue(validate_request(hist, 3, self.N_BIG).ok)
        for out in ([4, 5], (4, 5), np.array([4, 5])):
            r = guarded_recommend(lambda h, m, k, out=out: out, self.POP_BIG, [2, 3], 2)
            self.assertEqual(r['items'], [4, 5])
            self.assertFalse(r['used_fallback'])


if __name__ == '__main__':
    unittest.main()
