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

from dashboard.evidence import build_time_aware_evidence, prior_rows

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


# --------------------------------------------------------------- extra features (v2)
# Everything below uses only reviews dated before the one being described, like the evidence above.

EXTRA_FAMILIES = {
    "reviewer extras": (
        "reviewer_prior_0", "reviewer_prior_1", "reviewer_prior_2_3", "reviewer_prior_4_9",
        "reviewer_days_since_first_log", "reviewer_days_since_last_log", "reviewer_extreme_share",
        "reviewer_same_rating_share", "reviewer_own_rating_deviation",
    ),
    "business extras": (
        "business_days_since_first_log", "business_rating_std", "business_extreme_share",
        "business_reviews_last_30d_log", "business_own_rating_deviation",
    ),
    "text style": (
        "own_word_count_log", "own_exclamations", "own_caps_share", "own_digit_share",
        "own_avg_word_len", "own_sentences_log", "own_weekend",
    ),
}
EXTRA_FEATURES = tuple(f for members in EXTRA_FAMILIES.values() for f in members)


def extract_extra_features(toolbox, review):
    date = pd.Timestamp(review["date"])
    rating = float(review["rating"])
    reviewer = prior_rows(toolbox, "user_id", review["user_id"], date)
    business = prior_rows(toolbox, "prod_id", review["prod_id"], date)
    text = str(review["review_text"])
    words = text.split()
    nan = float("nan")
    n = len(reviewer)
    ratings, b_ratings = reviewer["rating"], business["rating"]
    last_30 = (date - business["date"]).dt.days <= 30 if len(business) else pd.Series([], dtype=bool)
    return {
        "reviewer_prior_0": float(n == 0), "reviewer_prior_1": float(n == 1),
        "reviewer_prior_2_3": float(2 <= n <= 3), "reviewer_prior_4_9": float(4 <= n <= 9),
        "reviewer_days_since_first_log": math.log1p((date - reviewer["date"].min()).days) if n else nan,
        "reviewer_days_since_last_log": math.log1p((date - reviewer["date"].max()).days) if n else nan,
        "reviewer_extreme_share": float(ratings.isin([1.0, 5.0]).mean()) if n else nan,
        "reviewer_same_rating_share": float((ratings == rating).mean()) if n else nan,
        "reviewer_own_rating_deviation": abs(rating - float(ratings.mean())) if n else nan,
        "business_days_since_first_log": math.log1p((date - business["date"].min()).days) if len(business) else nan,
        "business_rating_std": float(b_ratings.std()) if len(business) > 1 else nan,
        "business_extreme_share": float(b_ratings.isin([1.0, 5.0]).mean()) if len(business) else nan,
        "business_reviews_last_30d_log": math.log1p(int(last_30.sum())) if len(business) else nan,
        "business_own_rating_deviation": abs(rating - float(b_ratings.mean())) if len(business) else nan,
        "own_word_count_log": math.log1p(len(words)),
        "own_exclamations": float(text.count("!")),
        "own_caps_share": sum(ch.isupper() for ch in text) / max(len(text), 1),
        "own_digit_share": sum(ch.isdigit() for ch in text) / max(len(text), 1),
        "own_avg_word_len": float(np.mean([len(w) for w in words])) if words else nan,
        "own_sentences_log": math.log1p(sum(text.count(c) for c in ".!?")),
        "own_weekend": float(date.weekday() >= 5),
    }


def accuracy_at_prevalence(tpr, fpr, prevalence):
    """Accuracy a rule with these rates would have if `prevalence` of reviews were filtered."""
    return prevalence * tpr + (1 - prevalence) * (1 - fpr)


def choose_cutoff_by_prevalence(scores, labels, prevalence, accuracy_floor):
    """Score cutoff (flag when score >= cutoff) with the highest recall whose accuracy, recomputed
    at a fixed `prevalence`, stays >= `accuracy_floor`. Uses rates, so it does not move when a year's
    filtered share does. Returns (cutoff, recall, false_positive_rate) or None if none qualifies."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    best = None
    for cutoff in np.unique(scores):
        flagged = scores >= cutoff
        tpr = float((flagged & (labels == 1)).sum() / positives)
        fpr = float((flagged & (labels == 0)).sum() / negatives)
        if accuracy_at_prevalence(tpr, fpr, prevalence) >= accuracy_floor and (best is None or tpr >= best[1]):
            best = (float(cutoff), tpr, fpr)
    return best


def cutoff_at_false_positive_rate(scores, labels, fpr_target):
    """Score cutoff that flags about `fpr_target` of the not-filtered reviews (chosen on validation data)."""
    negatives = np.sort(np.asarray(scores)[np.asarray(labels) == 0])[::-1]
    index = max(int(round(len(negatives) * fpr_target)) - 1, 0)
    return float(negatives[index])
