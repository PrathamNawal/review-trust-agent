"""
The agent's three evidence-gathering tools, plus a text-similarity helper.
Each tool takes an `exclude_review_id` so a review is never compared against
itself, and each degrades gracefully (explicit "no data" signal, not a crash)
when a reviewer/business has no other history — that's itself evidence.
"""

import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_COMPARISON_TEXTS = 300  # cap for the similarity tool's candidate pool


class ToolBox:
    def __init__(self, idx):
        self.df = idx["df"]
        self.by_user = idx["by_user"]
        self.by_prod = idx["by_prod"]

    def get_review(self, review_id):
        row = self.df.loc[review_id]
        return {
            "review_id": int(review_id),
            "user_id": str(int(row["user_id"])),
            "prod_id": str(int(row["prod_id"])),
            "rating": float(row["rating"]),
            "date": row["date"].strftime("%Y-%m-%d"),
            "review_text": row["review_text"],
        }

    def _other_review_ids(self, id_list, exclude_review_id):
        if exclude_review_id is None:
            return list(id_list)
        return [rid for rid in id_list if rid != exclude_review_id]

    def reviewer_history_lookup(self, user_id, exclude_review_id=None):
        user_id = int(user_id)
        ids = self._other_review_ids(self.by_user.get(user_id, []), exclude_review_id)
        if not ids:
            return {
                "user_id": str(user_id),
                "other_review_count": 0,
                "note": "Singleton reviewer — no other reviews found for this user. "
                        "Singleton accounts are themselves a known spam signal in this domain.",
            }

        sub = self.df.loc[ids].sort_values("date")
        dates = sub["date"]
        gaps_days = dates.diff().dt.days.dropna().tolist()

        # Burst metric: most reviews this reviewer posted within any 7-day window.
        max_in_7_days = 1
        dates_list = dates.tolist()
        for i, d in enumerate(dates_list):
            count = sum(1 for d2 in dates_list if 0 <= (d2 - d).days <= 7)
            max_in_7_days = max(max_in_7_days, count)

        rating_std = sub["rating"].std()
        return {
            "user_id": str(user_id),
            "other_review_count": len(ids),
            "avg_rating": round(float(sub["rating"].mean()), 2),
            "rating_std": round(float(rating_std), 2) if not math.isnan(rating_std) else 0.0,
            "min_gap_days_between_reviews": int(min(gaps_days)) if gaps_days else None,
            "max_reviews_in_any_7_day_window": max_in_7_days,
            "first_review_date": dates.min().strftime("%Y-%m-%d"),
            "last_review_date": dates.max().strftime("%Y-%m-%d"),
        }

    def business_trend_lookup(self, prod_id, around_date, window_days=14, exclude_review_id=None):
        import pandas as pd

        prod_id = int(prod_id)
        ids = self._other_review_ids(self.by_prod.get(prod_id, []), exclude_review_id)
        if not ids:
            return {
                "prod_id": str(prod_id),
                "other_review_count": 0,
                "note": "No other reviews found for this business.",
            }

        sub = self.df.loc[ids]
        center = pd.to_datetime(around_date)
        window_mask = (sub["date"] - center).dt.days.abs() <= window_days
        windowed = sub[window_mask]

        overall_span_days = max((sub["date"].max() - sub["date"].min()).days, 1)
        baseline_rate = len(sub) / overall_span_days  # reviews per day, all-time
        window_rate = len(windowed) / (2 * window_days)  # reviews per day, in window
        burst_ratio = round(window_rate / baseline_rate, 2) if baseline_rate > 0 else None

        return {
            "prod_id": str(prod_id),
            "other_review_count": len(ids),
            "avg_rating_overall": round(float(sub["rating"].mean()), 2),
            "reviews_in_window": len(windowed),
            "avg_rating_in_window": round(float(windowed["rating"].mean()), 2) if len(windowed) else None,
            "window_days": window_days,
            "burst_ratio": burst_ratio,  # >1 means review velocity spiked around this date
        }

    def text_similarity_check(self, review_text, compare_against, target_id, exclude_review_id=None, top_k=3):
        target_id = int(target_id)
        if compare_against == "reviewer":
            ids = self._other_review_ids(self.by_user.get(target_id, []), exclude_review_id)
        elif compare_against == "business":
            ids = self._other_review_ids(self.by_prod.get(target_id, []), exclude_review_id)
        else:
            raise ValueError("compare_against must be 'reviewer' or 'business'")

        if not ids:
            return {"max_similarity": 0.0, "note": "No comparison texts available.", "similar_snippets": []}

        ids = ids[-MAX_COMPARISON_TEXTS:]  # most recent N, for performance
        candidate_texts = self.df.loc[ids, "review_text"].tolist()

        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform([review_text] + candidate_texts)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

        top_indices = sims.argsort()[::-1][:top_k]
        similar_snippets = [
            {"similarity": round(float(sims[i]), 3), "snippet": candidate_texts[i][:150]}
            for i in top_indices
        ]

        return {
            "max_similarity": round(float(sims.max()), 3),
            "avg_similarity": round(float(sims.mean()), 3),
            "compared_against_n_reviews": len(ids),
            "similar_snippets": similar_snippets,
        }
