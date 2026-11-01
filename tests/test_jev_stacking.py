import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.jev_model import JevAPIError  # noqa: E402
from scripts.run_jev_stacking_experiment import (  # noqa: E402
    logit, paired_auc_difference, pending_review_ids, read_cache, with_retries,
)


class StackingHelperTests(unittest.TestCase):
    def test_pending_ids_skip_cached_ones_and_keep_order(self):
        self.assertEqual(pending_review_ids([5, 3, 9, 1], {3, 1}), [5, 9])
        self.assertEqual(pending_review_ids([], {1}), [])

    def test_cache_reads_back_what_was_written_and_ignores_blank_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calls.jsonl"
            self.assertEqual(read_cache(path), {})
            path.write_text(json.dumps({"review_id": 7, "p_filtered": 0.3}) + "\n\n" + json.dumps({"review_id": 8, "p_filtered": None}) + "\n")
            cache = read_cache(path)
            self.assertEqual(set(cache), {7, 8})
            self.assertEqual(cache[7]["p_filtered"], 0.3)

    def test_logit_is_finite_at_the_edges_and_monotone(self):
        values = logit([0.0, 0.01, 0.5, 0.99, 1.0])
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((np.diff(values) > 0).all())

    def test_paired_difference_is_near_zero_for_identical_scores(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        s = rng.normal(size=300) + y
        low, high = paired_auc_difference(y, s, s, rng, rounds=200)
        self.assertEqual((low, high), (0.0, 0.0))


class RetryTests(unittest.TestCase):
    def test_a_dropped_connection_is_retried_then_succeeds(self):
        calls, waits = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise JevAPIError("Jev request failed (RemoteDisconnected).")
            return "answer"
        self.assertEqual(with_retries(flaky, sleep=waits.append), "answer")
        self.assertEqual((len(calls), waits), (3, [2.0, 4.0]))

    def test_it_gives_up_after_the_last_attempt(self):
        def always():
            raise JevAPIError("down")
        with self.assertRaises(JevAPIError):
            with_retries(always, attempts=3, sleep=lambda _: None)


if __name__ == "__main__":
    unittest.main()
