import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.decision_policy import filtered_score, order_queue


class FilteredScoreTests(unittest.TestCase):
    def test_choice_uses_the_filtered_probability_not_the_picked_labels(self):
        # Jev picked "recommended": the logged decision value would be 0.8, but the review is 0.2 filtered.
        self.assertEqual(filtered_score("choice", "{'filtered': 0.2, 'recommended': 0.8}", "0.8"), 0.2)

    def test_score_and_noul_use_the_logged_value(self):
        self.assertEqual(filtered_score("noul", None, "0.62"), 0.62)
        self.assertEqual(filtered_score("score", None, 0.4), 0.4)

    def test_missing_or_bad_values_give_no_score(self):
        for args in (("choice", None, None), ("choice", "not a dict", None), ("choice", "{'recommended': 0.8}", None),
                     ("choice", "{'filtered': 5}", None), ("noul", None, "nan"), ("noul", None, None),
                     (float("nan"), None, None), (None, None, None)):
            self.assertIsNone(filtered_score(*args), args)


class OrderQueueTests(unittest.TestCase):
    def test_appeals_stay_on_top_then_highest_score_first(self):
        scores = {1: 0.1, 2: 0.9, 3: 0.5, 4: 0.99}
        self.assertEqual(order_queue([4], [1, 2, 3, 4], scores), [4, 2, 3, 1])

    def test_unscored_cases_go_last_in_id_order(self):
        self.assertEqual(order_queue([], [5, 3, 9, 1], {9: 0.3, 1: None}), [9, 1, 3, 5])

    def test_without_any_scores_the_order_is_by_id_as_before(self):
        self.assertEqual(order_queue([], [3, 1, 2], {}), [1, 2, 3])

    def test_ties_break_by_id(self):
        self.assertEqual(order_queue([], [7, 2, 5], {7: 0.4, 2: 0.4, 5: 0.4}), [2, 5, 7])


if __name__ == "__main__":
    unittest.main()
