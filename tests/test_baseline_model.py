import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.baseline_model import (  # noqa: E402
    EXTRA_FEATURES, FEATURES, FEATURE_FAMILIES, accuracy_at_prevalence, build_pipeline,
    choose_cutoff_by_prevalence, cutoff_at_false_positive_rate, extract_extra_features, extract_features,
    pick_set_aside_threshold, share_needed_for_recall, time_split, top_share_captured,
)
from scripts.tools import ToolBox  # noqa: E402


def _toolbox(rows):
    df = pd.DataFrame({
        "user_id": [r[1] for r in rows], "prod_id": [r[2] for r in rows],
        "rating": [r[3] for r in rows], "date": pd.to_datetime([r[4] for r in rows]),
        "review_text": [r[5] for r in rows], "filtered": [False] * len(rows),
    }, index=pd.Index([r[0] for r in rows], name="review_id"))
    ordered = df.sort_values("date").index.tolist()
    return ToolBox({"df": df, "by_user": {1: ordered}, "by_prod": {10: ordered}})


CASE = {"review_id": 1, "user_id": "1", "prod_id": "10", "date": "2012-03-01",
        "review_text": "great pasta here", "rating": 5.0}
EARLIER = [(0, 1, 10, 4.0, "2012-01-01", "decent pasta"), (1, 1, 10, 5.0, "2012-03-01", "great pasta here")]


class FeatureTests(unittest.TestCase):
    def test_every_feature_is_present_and_numeric_or_nan(self):
        features = extract_features(_toolbox(EARLIER), CASE)
        self.assertEqual(list(features), list(FEATURES))
        self.assertTrue(all(isinstance(v, float) for v in features.values()))

    def test_later_reviews_cannot_change_a_reviews_features(self):
        later = EARLIER + [(2, 1, 10, 1.0, "2012-09-01", "great pasta here again"),
                           (3, 1, 10, 1.0, "2012-03-01", "great pasta")]  # a later one and a same-day one
        before, after = extract_features(_toolbox(EARLIER), CASE), extract_features(_toolbox(later), CASE)
        for name in FEATURES:
            a, b = before[name], after[name]
            self.assertTrue((math.isnan(a) and math.isnan(b)) or a == b, name)

    def test_no_earlier_reviews_gives_missing_reviewer_features_not_zeros(self):
        features = extract_features(_toolbox([(1, 1, 10, 5.0, "2012-03-01", "great pasta here")]), CASE)
        self.assertEqual(features["reviewer_prior_reviews_log"], 0.0)
        self.assertTrue(math.isnan(features["reviewer_avg_rating"]))

    def test_feature_families_only_name_real_features(self):
        for members in FEATURE_FAMILIES.values():
            self.assertTrue(set(members) <= set(FEATURES))


class ExtraFeatureTests(unittest.TestCase):
    def test_every_extra_feature_is_present(self):
        features = extract_extra_features(_toolbox(EARLIER), CASE)
        self.assertEqual(set(features), set(EXTRA_FEATURES))

    def test_later_and_same_day_reviews_cannot_change_extra_features(self):
        later = EARLIER + [(2, 1, 10, 1.0, "2012-09-01", "great pasta here again"),
                           (3, 1, 10, 1.0, "2012-03-01", "great pasta")]
        before, after = extract_extra_features(_toolbox(EARLIER), CASE), extract_extra_features(_toolbox(later), CASE)
        for name in EXTRA_FEATURES:
            a, b = before[name], after[name]
            self.assertTrue((math.isnan(a) and math.isnan(b)) or a == b, name)

    def test_prior_count_buckets_are_exclusive(self):
        none = extract_extra_features(_toolbox([(1, 1, 10, 5.0, "2012-03-01", "great pasta here")]), CASE)
        one = extract_extra_features(_toolbox(EARLIER), CASE)
        self.assertEqual((none["reviewer_prior_0"], none["reviewer_prior_1"]), (1.0, 0.0))
        self.assertEqual((one["reviewer_prior_0"], one["reviewer_prior_1"], one["reviewer_prior_2_3"]), (0.0, 1.0, 0.0))
        self.assertTrue(math.isnan(none["reviewer_days_since_first_log"]))


class CutoffTests(unittest.TestCase):
    def test_accuracy_at_prevalence(self):
        self.assertAlmostEqual(accuracy_at_prevalence(0.0, 0.0, 0.12), 0.88)   # flag nothing
        self.assertAlmostEqual(accuracy_at_prevalence(1.0, 1.0, 0.12), 0.12)   # flag everything
        self.assertAlmostEqual(accuracy_at_prevalence(0.5, 0.1, 0.2), 0.2 * 0.5 + 0.8 * 0.9)

    def test_chosen_cutoff_has_the_most_recall_that_still_meets_the_floor(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        labels = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        cutoff, tpr, fpr = choose_cutoff_by_prevalence(scores, labels, 0.3, 0.85)
        self.assertGreaterEqual(accuracy_at_prevalence(tpr, fpr, 0.3), 0.85)
        # lowering the cutoff any further must break the floor
        lower = scores[scores < cutoff]
        if len(lower):
            flagged = scores >= lower.max()
            tpr2 = (flagged & (labels == 1)).sum() / labels.sum()
            fpr2 = (flagged & (labels == 0)).sum() / (labels == 0).sum()
            self.assertLess(accuracy_at_prevalence(tpr2, fpr2, 0.3), 0.85)

    def test_no_cutoff_when_the_floor_is_unreachable(self):
        self.assertIsNone(choose_cutoff_by_prevalence([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0], 0.5, 0.99))

    def test_cutoff_at_false_positive_rate_flags_that_share_of_negatives(self):
        scores = np.linspace(0, 1, 101)
        labels = np.zeros(101, dtype=int)
        cutoff = cutoff_at_false_positive_rate(scores, labels, 0.10)
        self.assertAlmostEqual((scores >= cutoff).mean(), 0.10, delta=0.02)


class SplitAndMetricTests(unittest.TestCase):
    def test_time_split_partitions_by_date_with_no_overlap(self):
        dates = ["2012-12-31", "2013-01-01", "2013-12-31", "2014-01-01"]
        train, valid, test = time_split(dates, "2012-12-31", "2013-12-31")
        self.assertEqual((train.tolist(), valid.tolist(), test.tolist()),
                         ([True, False, False, False], [False, True, True, False], [False, False, False, True]))

    def test_top_share_and_share_needed(self):
        scores, labels = [0.9, 0.8, 0.7, 0.2, 0.1], [1, 0, 1, 0, 0]
        self.assertEqual(top_share_captured(scores, labels, (0.2, 0.6)), {0.2: 0.5, 0.6: 1.0})
        self.assertEqual(share_needed_for_recall(scores, labels, 1.0), 0.6)

    def test_set_aside_threshold_respects_the_filtered_rate_limit(self):
        scores = np.arange(100) / 100
        labels = np.zeros(100, dtype=int)
        labels[[60, 70, 80]] = 1  # filtered cases only appear at higher scores
        t = pick_set_aside_threshold(scores, labels, 0.02)
        tail = labels[scores <= t]
        self.assertLessEqual(tail.mean(), 0.02)
        self.assertGreater(len(tail), 50)

    def test_set_aside_threshold_is_none_when_nothing_qualifies(self):
        self.assertIsNone(pick_set_aside_threshold([0.1, 0.2], [1, 1], 0.0))


class PipelineTests(unittest.TestCase):
    def test_pipeline_handles_missing_values_and_returns_probabilities(self):
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
        y = (x["a"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
        x.loc[::7, "b"] = np.nan
        probs = build_pipeline(1.0).fit(x, y).predict_proba(x)[:, 1]
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())


if __name__ == "__main__":
    unittest.main()
