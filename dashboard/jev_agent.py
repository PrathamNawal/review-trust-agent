"""Jev architecture adapter for Streamlit's shared investigation contract."""

import math
import os
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dashboard.agent_architectures import register_architecture
from dashboard.config_store import DEFAULT_JEV_EVIDENCE_SOURCES, DEFAULT_JEV_PROMPTS
from dashboard.jev_model import DECISION_METHODS, JevAPIError, classify_review
from dashboard.live_agent import get_toolbox


MODEL_ID = "jev-1.13.0"
MAX_SIMILARITY_TEXTS = 300


def get_typesafe_api_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("TYPESAFE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("TYPESAFE_API_KEY="):
                return stripped.split("=", 1)[1].strip().strip("\"'") or None
    return None


def _prior_rows(toolbox, column, entity_id, target_date):
    ids = toolbox.by_user.get(int(entity_id), []) if column == "user_id" else toolbox.by_prod.get(int(entity_id), [])
    if not ids:
        return toolbox.df.iloc[0:0]
    rows = toolbox.df.loc[ids]
    return rows[rows["date"] < pd.Timestamp(target_date)].sort_values("date")


def _time_aware_evidence(toolbox, review, selected_sources=None):
    """Build selected evidence fields while excluding future reviews."""
    selected_sources = set(selected_sources or (
        "reviewer_history", "business_trend", "reviewer_text_similarity",
        "business_text_similarity",
    ))
    target_date = pd.Timestamp(review["date"])
    reviewer_rows = None
    business_rows = None
    if selected_sources.intersection({"reviewer_history", "reviewer_text_similarity"}):
        reviewer_rows = _prior_rows(toolbox, "user_id", review["user_id"], target_date)
    if selected_sources.intersection({"business_trend", "business_text_similarity"}):
        business_rows = _prior_rows(toolbox, "prod_id", review["prod_id"], target_date)

    evidence = {}
    if "reviewer_history" in selected_sources:
        if reviewer_rows.empty:
            evidence["reviewer_history"] = {
                "user_id": str(review["user_id"]), "other_review_count": 0,
                "note": "No earlier reviews found for this reviewer.",
            }
        else:
            dates = reviewer_rows["date"].sort_values().tolist()
            gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
            max_in_week = max(
                sum(0 <= (other - date).days <= 7 for other in dates[i:])
                for i, date in enumerate(dates)
            )
            rating_std = reviewer_rows["rating"].std()
            evidence["reviewer_history"] = {
                "user_id": str(review["user_id"]),
                "other_review_count": len(reviewer_rows),
                "avg_rating": round(float(reviewer_rows["rating"].mean()), 2),
                "rating_std": round(float(rating_std), 2) if not math.isnan(rating_std) else 0.0,
                "min_gap_days_between_reviews": min(gaps) if gaps else None,
                "max_reviews_in_any_7_day_window": max_in_week,
                "first_review_date": dates[0].strftime("%Y-%m-%d"),
                "last_review_date": dates[-1].strftime("%Y-%m-%d"),
            }

    if "business_trend" in selected_sources:
        if business_rows.empty:
            evidence["business_trend"] = {
                "prod_id": str(review["prod_id"]), "other_review_count": 0,
                "note": "No earlier reviews found for this business.",
            }
        else:
            days_back = (target_date - business_rows["date"]).dt.days
            window_rows = business_rows[(days_back >= 0) & (days_back <= 14)]
            span = max((business_rows["date"].max() - business_rows["date"].min()).days, 1)
            baseline_rate = len(business_rows) / span
            window_rate = len(window_rows) / 14
            evidence["business_trend"] = {
                "prod_id": str(review["prod_id"]),
                "other_review_count": len(business_rows),
                "avg_rating_overall": round(float(business_rows["rating"].mean()), 2),
                "reviews_in_window": len(window_rows),
                "avg_rating_in_window": round(float(window_rows["rating"].mean()), 2) if len(window_rows) else None,
                "window_days": 14,
                "burst_ratio": round(window_rate / baseline_rate, 2) if baseline_rate > 0 else None,
            }

    def similarity(rows):
        if rows.empty:
            return {"max_similarity": 0.0, "note": "No earlier comparison texts available.",
                    "similar_snippets": []}
        texts = rows.sort_values("date", ascending=False)["review_text"].head(MAX_SIMILARITY_TEXTS).tolist()
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(
            [review["review_text"]] + texts
        )
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        top = scores.argsort()[::-1][:3]
        return {
            "max_similarity": round(float(scores.max()), 3),
            "avg_similarity": round(float(scores.mean()), 3),
            "compared_against_n_reviews": len(texts),
            "similar_snippets": [
                {"similarity": round(float(scores[i]), 3), "snippet": texts[i][:150]}
                for i in top
            ],
        }

    if "reviewer_text_similarity" in selected_sources:
        evidence["reviewer_text_similarity"] = similarity(reviewer_rows)
    if "business_text_similarity" in selected_sources:
        evidence["business_text_similarity"] = similarity(business_rows)
    return evidence


def run_jev_investigation_stream(review, config):
    """Use Jev Choice as an advisory label; every result routes to a human."""
    api_key = get_typesafe_api_key()
    if not api_key:
        yield {"type": "error", "message": "TYPESAFE_API_KEY is not set. Add it in the sidebar or Streamlit secrets."}
        return

    try:
        yield {"type": "turn", "turn": 0}
        toolbox = get_toolbox()
        selected_sources = config.get("jev_evidence_sources", DEFAULT_JEV_EVIDENCE_SOURCES)
        if not selected_sources:
            raise ValueError("Select at least one Jev evidence source in System Config.")
        all_evidence = _time_aware_evidence(toolbox, review, selected_sources)
        unknown_sources = set(selected_sources).difference(all_evidence)
        if unknown_sources:
            raise ValueError("The deployed Jev config contains an unknown evidence source.")
        evidence = {key: all_evidence[key] for key in selected_sources}
        event_meta = {
            "reviewer_history": ("reviewer_history_lookup", {"user_id": review["user_id"]}),
            "business_trend": ("business_trend_lookup", {
                "prod_id": review["prod_id"], "around_date": review["date"]}),
            "reviewer_text_similarity": ("text_similarity_check", {
                "compare_against": "reviewer", "target_id": review["user_id"]}),
            "business_text_similarity": ("text_similarity_check", {
                "compare_against": "business", "target_id": review["prod_id"]}),
        }
        for key, (tool_name, tool_input) in event_meta.items():
            if key not in evidence:
                continue
            yield {"type": "tool_call", "turn": 0, "tool": tool_name,
                   "input": tool_input, "result": evidence[key]}

        method = config.get("jev_method", "choice")
        if method not in DECISION_METHODS:
            raise ValueError("The deployed Jev config has an unsupported decision method.")
        prompts = {**DEFAULT_JEV_PROMPTS, **config.get("jev_prompts", {})}
        threshold = float(config.get("jev_threshold", 0.5))
        result = classify_review(
            review, evidence, api_key=api_key, model=MODEL_ID,
            instructions=prompts[method], method=method, threshold=threshold,
        )
        predicted = result["predicted_filtered"]
        choice_probabilities = result.get("choice_probabilities", {})
        method_label = method.title()
        if method == "choice":
            decision_summary = f"Jev Choice selected {'filtered' if predicted else 'recommended'}"
        else:
            decision_summary = (
                f"Jev {method_label} returned {result['decision_value']:.3f}; the configured "
                f"threshold is {threshold:.2f}"
            )
        reasoning = (
            f"{decision_summary} after reviewing {len(evidence)} selected evidence source(s). "
            "This experimental output is not a calibrated fraud probability; a human must review it."
        )
        yield {
            "type": "final",
            "predicted_filtered": predicted,
            "confidence": None,
            "confidence_calibrated": False,
            "primary_signal": f"Jev {method_label}",
            "reasoning": reasoning,
            "turns_used": 1,
            "jev_choice": ("filtered" if predicted else "recommended") if method == "choice" else None,
            "jev_choice_probabilities": choice_probabilities,
            "jev_method": method,
            "jev_decision_value": result.get("decision_value"),
            "jev_threshold": result.get("decision_threshold"),
            "model": result.get("model", MODEL_ID),
        }
    except JevAPIError as exc:
        yield {"type": "error", "message": str(exc)}
    except Exception as exc:
        yield {"type": "error", "message": f"Jev investigation failed ({type(exc).__name__})."}


register_architecture("jev", run_jev_investigation_stream)
