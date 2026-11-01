"""Candidate routing score: logistic regression on the evidence the app already builds.

Offline only for now — nothing here is wired into the live dashboard. Features come from
`build_time_aware_evidence`, so a review is described using only reviews dated before it.
The target is YelpZip's `filtered` label, which is Yelp's own filtering outcome, not verified fraud.
"""

import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dashboard.evidence import build_time_aware_evidence

FEATURES = (
    "own_rating", "own_text_len_log",
    "reviewer_prior_reviews_log", "reviewer_avg_rating", "reviewer_rating_std",
    "reviewer_min_gap_days_log", "reviewer_max_in_7_days",
    "reviewer_text_max_sim", "reviewer_text_avg_sim",
    "business_prior_reviews_log", "business_avg_rating", "business_reviews_in_window",
    "business_window_avg_rating", "business_burst_ratio", "own_rating_minus_business_avg",
    "business_text_max_sim", "business_text_avg_sim",
)

FEATURE_FAMILIES = {
    "reviewer activity": tuple(f for f in FEATURES if f.startswith("reviewer_")),
    "business trend": tuple(f for f in FEATURES if f.startswith("business_prior") or f in (
        "business_avg_rating", "business_reviews_in_window", "business_window_avg_rating",
        "business_burst_ratio", "own_rating_minus_business_avg")),
    "text reuse": ("reviewer_text_max_sim", "reviewer_text_avg_sim", "business_text_max_sim",
                   "business_text_avg_sim"),
}


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float("nan") if math.isnan(value) else float(value)


def _log1p(value):
    return float("nan") if math.isnan(value) else math.log1p(max(value, 0.0))


def extract_features(toolbox, review):
    """One review -> {feature: float}, NaN where the evidence is absent (e.g. no earlier reviews)."""
    evidence = build_time_aware_evidence(toolbox, review)
    reviewer = evidence["reviewer_history"]
    business = evidence["business_trend"]
    reviewer_text = evidence.get("reviewer_text_similarity", {})
    business_text = evidence.get("business_text_similarity", {})
    rating = _num(review["rating"])
    business_avg = _num(business.get("avg_rating_overall"))
    burst = _num(business.get("burst_ratio"))
    return {
        "own_rating": rating,
        "own_text_len_log": math.log1p(len(str(review["review_text"]))),
        "reviewer_prior_reviews_log": _log1p(_num(reviewer.get("other_review_count"))),
        "reviewer_avg_rating": _num(reviewer.get("avg_rating")),
        "reviewer_rating_std": _num(reviewer.get("rating_std")),
        "reviewer_min_gap_days_log": _log1p(_num(reviewer.get("min_gap_days_between_reviews"))),
        "reviewer_max_in_7_days": _num(reviewer.get("max_reviews_in_any_7_day_window")),
        "reviewer_text_max_sim": _num(reviewer_text.get("max_similarity")),
        "reviewer_text_avg_sim": _num(reviewer_text.get("avg_similarity")),
        "business_prior_reviews_log": _log1p(_num(business.get("other_review_count"))),
        "business_avg_rating": business_avg,
        "business_reviews_in_window": _num(business.get("reviews_in_window")),
        "business_window_avg_rating": _num(business.get("avg_rating_in_window")),
        "business_burst_ratio": min(burst, 10.0) if not math.isnan(burst) else burst,
        "own_rating_minus_business_avg": rating - business_avg,
        "business_text_max_sim": _num(business_text.get("max_similarity")),
        "business_text_avg_sim": _num(business_text.get("avg_similarity")),
    }


def build_pipeline(c=1.0):
    """Median-impute with missing-value flags, standardise, then regularised logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
        LogisticRegression(C=c, max_iter=2000),
    )


def time_split(dates, train_end, valid_end):
    """Boolean masks (train, validation, test) by review date: <= train_end, <= valid_end, later."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    train_end, valid_end = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    train = (dates <= train_end).to_numpy()
    valid = ((dates > train_end) & (dates <= valid_end)).to_numpy()
    return train, valid, ~(train | valid)


def coefficient_table(pipeline, columns):
    """Standardised coefficients with names, largest magnitude first (includes missing-value flags)."""
    imputer, model = pipeline[0], pipeline[-1]
    names = list(columns)
    flags = [f"{columns[i]} (missing)" for i in imputer.indicator_.features_]
    # SimpleImputer drops all-NaN columns; training data here never has one, so the order is stable.
    table = pd.DataFrame({"feature": names + flags, "coefficient": model.coef_[0]})
    return table.reindex(table["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def top_share_captured(scores, labels, shares):
    """{share: fraction of positives found if the top `share` of reviews by score are reviewed}."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    labels = np.asarray(labels)[order]
    total = labels.sum()
    return {s: float(labels[:int(round(len(labels) * s))].sum() / total) for s in shares}


def share_needed_for_recall(scores, labels, recall):
    """Smallest share of reviews (best-scored first) that contains `recall` of the positives."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    caught = np.cumsum(np.asarray(labels)[order]) / np.asarray(labels).sum()
    return float((np.searchsorted(caught, recall) + 1) / len(caught))


def pick_set_aside_threshold(scores, labels, max_filtered_rate):
    """Largest score t such that reviews scoring <= t contain at most `max_filtered_rate` filtered ones.

    Chosen on validation data only. Returns None if even the single lowest score breaks the limit.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    ok = np.where(cumulative <= max_filtered_rate)[0]
    return None if len(ok) == 0 else float(scores[order][ok[-1]])
