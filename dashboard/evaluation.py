"""Repeatable, side-effect-free evaluation of an agent configuration."""

import pandas as pd

from dashboard.agent_architectures import run_investigation_stream
from dashboard.metrics import binary_classification_metrics


def evaluate_reviews(reviews, config, runner=run_investigation_stream):
    """Run a fixed review set and return row-level results plus summary metrics.

    `runner` is injectable so architecture variants can be compared under the
    same contract. This function does not write to MLflow or modify config.
    """
    rows = []
    for review in reviews:
        verdict = None
        error = None
        tool_calls = 0
        for event in runner(review, config):
            if event.get("type") == "tool_call":
                tool_calls += 1
            elif event.get("type") == "error":
                error = event.get("message")
            elif event.get("type") == "final":
                verdict = event.get("predicted_filtered")

        rows.append({
            "review_id": review["review_id"],
            "true_filtered": bool(review["filtered"]),
            "predicted_filtered": verdict,
            "correct": verdict is not None and bool(verdict) == bool(review["filtered"]),
            "tool_calls": tool_calls,
            "error": error,
        })

    results = pd.DataFrame(rows, columns=[
        "review_id", "true_filtered", "predicted_filtered", "correct", "tool_calls", "error"
    ])
    completed = results[results["predicted_filtered"].notna()] if not results.empty else results
    metrics = binary_classification_metrics(
        completed["true_filtered"].tolist(), completed["predicted_filtered"].tolist()
    )
    metrics["attempted"] = len(results)
    metrics["completed"] = len(completed)
    metrics["completion_rate"] = len(completed) / len(results) if len(results) else None
    metrics["mean_tool_calls"] = float(results["tool_calls"].mean()) if len(results) else None
    return results, metrics


def load_evaluation_reviews(path, limit=None):
    """Load labeled reviews from a CSV containing the app review fields."""
    frame = pd.read_csv(path)
    required = {"review_id", "user_id", "prod_id", "rating", "date", "review_text", "filtered"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {', '.join(sorted(missing))}")
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_dict(orient="records")


def compare_configurations(baseline, candidate, reviews, runner=run_investigation_stream):
    """Evaluate baseline and candidate on identical rows; report metric deltas."""
    baseline_rows, baseline_metrics = evaluate_reviews(reviews, baseline, runner)
    candidate_rows, candidate_metrics = evaluate_reviews(reviews, candidate, runner)
    deltas = {}
    for key in ("recall", "precision", "f1", "balanced_accuracy", "accuracy", "completion_rate"):
        before, after = baseline_metrics[key], candidate_metrics[key]
        deltas[key] = after - before if before is not None and after is not None else None
    return {
        "review_ids": [review["review_id"] for review in reviews],
        "baseline": {"rows": baseline_rows, "metrics": baseline_metrics},
        "candidate": {"rows": candidate_rows, "metrics": candidate_metrics},
        "delta": deltas,
    }
