"""
MLflow-backed experiment tracking for this dashboard. One MLflow "run" per
judged review, tagged with the config version and judge type (human/agent),
so real before/after comparison across config deploys is a group-by away —
not a hand-rolled CSV.

Local file store (./mlruns) — free, no server needed, per the zero-cost
toolkit's guidance to avoid tool sprawl.
"""

import os

import mlflow
import pandas as pd
from dashboard.metrics import binary_classification_metrics, wilson_interval

TRACKING_DB = os.path.join(os.path.dirname(__file__), "..", "data", "mlflow.db")
EXPERIMENT_NAME = "review-trust-agent"

mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(TRACKING_DB)}")
mlflow.set_experiment(EXPERIMENT_NAME)


def log_judgment(*, judge_type, config_version, review_id, true_filtered, predicted_filtered,
                  confidence, primary_signal, reasoning=None, turns_used=None, n_tool_calls=None,
                  provider=None, model=None, temperature=None, extra_tags=None):
    """judge_type: 'human' or 'agent'. Logs one MLflow run per judged review.

    reasoning is stored as a tag (truncated) so the Analyst Queue and Check My
    Review pages can retrieve a past agent judgment's explanation without
    needing to re-run the agent."""
    correct = (predicted_filtered == true_filtered) if predicted_filtered is not None else False

    with mlflow.start_run(run_name=f"{judge_type}_{config_version}_{review_id}"):
        mlflow.set_tag("judge_type", judge_type)
        mlflow.set_tag("config_version", config_version)
        mlflow.set_tag("review_id", str(review_id))
        if provider:
            mlflow.set_tag("provider", provider)
        if model:
            mlflow.set_tag("model", model)
        if reasoning:
            mlflow.set_tag("reasoning", str(reasoning)[:800])
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

        if temperature is not None:
            mlflow.log_param("temperature", temperature)

        mlflow.log_metric("correct", int(correct))
        mlflow.log_metric("true_filtered", int(true_filtered))
        if predicted_filtered is not None:
            mlflow.log_metric("predicted_filtered", int(predicted_filtered))
        mlflow.log_metric("confidence", confidence or 0.0)
        if turns_used is not None:
            mlflow.log_metric("turns_used", turns_used)
        if n_tool_calls is not None:
            mlflow.log_metric("n_tool_calls", n_tool_calls)
        mlflow.set_tag("primary_signal", primary_signal or "")


def get_agent_judgment(review_id, config_version):
    """Returns the most recent real agent judgment for this review under this
    config version, or None if the agent hasn't investigated it yet. Lets the
    Analyst Queue and Check My Review pages show a past verdict without
    re-running the LLM."""
    runs = get_all_runs()
    if runs.empty or "tags.judge_type" not in runs.columns:
        return None
    mask = ((runs["tags.judge_type"] == "agent") & (runs.get("tags.config_version") == config_version)
             & (runs.get("tags.review_id") == str(review_id)))
    sub = runs.loc[mask]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    predicted = row.get("metrics.predicted_filtered")
    return {
        "predicted_filtered": bool(predicted) if pd.notna(predicted) else None,
        "confidence": row.get("metrics.confidence"),
        "primary_signal": row.get("tags.primary_signal", ""),
        "reasoning": row.get("tags.reasoning", ""),
    }


def log_analyst_action(*, config_version, review_id, agent_confidence, action, override_to=None,
                        reason="", resolves_appeal=False):
    """Logs a real, inspectable analyst decision on an agent-investigated case —
    separate from the agent's own judgment run — so agreement/override rate is
    a real, queryable metric rather than invisible. action: 'accept' | 'override'
    | 'escalate'."""
    with mlflow.start_run(run_name=f"analyst_{config_version}_{review_id}_{action}"):
        mlflow.set_tag("record_type", "analyst_action")
        mlflow.set_tag("config_version", config_version)
        mlflow.set_tag("review_id", str(review_id))
        mlflow.set_tag("action", action)
        if override_to is not None:
            mlflow.set_tag("override_to", str(override_to))
        if reason:
            mlflow.set_tag("reason", str(reason)[:800])
        mlflow.log_metric("agent_confidence", agent_confidence or 0.0)
        mlflow.log_metric("agreed_with_agent", int(action == "accept"))
    if resolves_appeal:
        resolve_appeal(review_id, config_version)


def log_appeal(*, review_id, config_version, note=""):
    """A reviewer/business contesting an auto-removal. Status starts 'pending'
    and shows up in the Analyst Queue until an analyst acts on it."""
    with mlflow.start_run(run_name=f"appeal_{config_version}_{review_id}"):
        mlflow.set_tag("record_type", "appeal")
        mlflow.set_tag("config_version", config_version)
        mlflow.set_tag("review_id", str(review_id))
        mlflow.set_tag("status", "pending")
        if note:
            mlflow.set_tag("note", str(note)[:800])


def get_pending_appeals(config_version):
    """review_ids with a pending appeal under this config version, most recent first."""
    runs = get_all_runs()
    if runs.empty or "tags.record_type" not in runs.columns:
        return []
    mask = ((runs["tags.record_type"] == "appeal") & (runs.get("tags.config_version") == config_version)
             & (runs.get("tags.status") == "pending"))
    ids = runs.loc[mask, "tags.review_id"].astype(int).tolist()
    return list(dict.fromkeys(reversed(ids)))  # de-dup, most recent first


def resolve_appeal(review_id, config_version):
    """Marks any pending appeal for this review/config as resolved, once an
    analyst has acted on it."""
    from mlflow.tracking import MlflowClient
    runs = get_all_runs()
    if runs.empty or "tags.record_type" not in runs.columns:
        return
    mask = ((runs["tags.record_type"] == "appeal") & (runs.get("tags.config_version") == config_version)
             & (runs.get("tags.review_id") == str(review_id)) & (runs.get("tags.status") == "pending"))
    client = MlflowClient()
    for run_id in runs.loc[mask, "run_id"]:
        client.set_tag(run_id, "status", "resolved")


def _wilson_ci(successes, n, z=1.96):
    """Compatibility wrapper for the shared metrics module."""
    return wilson_interval(successes, n, z)


def get_all_runs():
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_ids=[exp.experiment_id])


def summarize_by_group(group_cols=("tags.judge_type", "tags.config_version")):
    """Returns a dataframe: one row per (judge_type, config_version) group, with
    n, accuracy, precision, recall, confidence interval, avg turns/tool calls."""
    runs = get_all_runs()
    if runs.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in runs.groupby(list(group_cols)):
        n = len(group)
        correct = group["metrics.correct"].sum()
        acc_lo, acc_hi = _wilson_ci(correct, n)
        classification = binary_classification_metrics(
            group["metrics.true_filtered"].tolist(),
            group["metrics.predicted_filtered"].tolist(),
        )

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n": n,
            "accuracy": round(classification["accuracy"], 3),
            "accuracy_ci": f"{acc_lo:.0%}-{acc_hi:.0%}",
            "precision": round(classification["precision"], 3) if classification["precision"] is not None else None,
            "recall": round(classification["recall"], 3) if classification["recall"] is not None else None,
            "specificity": round(classification["specificity"], 3) if classification["specificity"] is not None else None,
            "false_positive_rate": round(classification["false_positive_rate"], 3)
            if classification["false_positive_rate"] is not None else None,
            "f1": round(classification["f1"], 3) if classification["f1"] is not None else None,
            "balanced_accuracy": round(classification["balanced_accuracy"], 3)
            if classification["balanced_accuracy"] is not None else None,
            "true_positive": classification["true_positive"],
            "false_positive": classification["false_positive"],
            "false_negative": classification["false_negative"],
            "true_negative": classification["true_negative"],
            "avg_confidence": round(group["metrics.confidence"].mean(), 3),
        })
        if "metrics.turns_used" in group.columns:
            row["avg_turns"] = round(group["metrics.turns_used"].mean(), 2)
        rows.append(row)

    return pd.DataFrame(rows)


def seed_historical_baselines():
    """Loads this project's REAL pre-existing results (golden set human labels,
    the 50-review Claude-manual batch) into MLflow as historical baseline runs —
    so comparison views have real data from first launch, not an empty state."""
    runs = get_all_runs()
    if not runs.empty and (runs["tags.config_version"] == "baseline").any():
        return  # already seeded

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    golden_path = os.path.join(data_dir, "golden_set.csv")
    golden_labels_path = os.path.join(data_dir, "golden_set_human_labels.csv")
    if os.path.exists(golden_path) and os.path.exists(golden_labels_path):
        golden = pd.read_csv(golden_path)[["review_id", "filtered"]]
        labels = pd.read_csv(golden_labels_path)
        merged = labels.merge(golden, on="review_id", how="inner")
        for _, row in merged.iterrows():
            log_judgment(
                judge_type="human", config_version="baseline", review_id=row["review_id"],
                true_filtered=bool(row["filtered"]), predicted_filtered=bool(row["predicted_filtered"]),
                confidence=row["confidence"], primary_signal=row["primary_signal"],
            )

    batch_path = os.path.join(data_dir, "manual_batch.csv")
    batch_judgments_path = os.path.join(data_dir, "manual_batch_judgments.csv")
    if os.path.exists(batch_path) and os.path.exists(batch_judgments_path):
        batch = pd.read_csv(batch_path)[["review_id", "filtered"]]
        judgments = pd.read_csv(batch_judgments_path)
        merged = judgments.merge(batch, on="review_id", how="inner")
        for _, row in merged.iterrows():
            log_judgment(
                judge_type="agent", config_version="baseline", review_id=row["review_id"],
                true_filtered=bool(row["filtered"]), predicted_filtered=bool(row["predicted_filtered"]),
                confidence=row["confidence"], primary_signal=row["primary_signal"],
                extra_tags={"note": "Claude-manual reasoning, precomputed, not live API"},
            )
