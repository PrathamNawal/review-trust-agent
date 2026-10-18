"""Run Jev on the fixed manual batch and compare with its recorded baseline.

Requires TYPESAFE_API_KEY in the process environment. Outputs are written only to
the requested local report directory, which is ignored by Git.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dashboard.metrics import binary_classification_metrics, wilson_interval
from dashboard.jev_model import classify_review
from scripts.data_index import load_index
from scripts.tool_schemas import dispatch_tool_call
from scripts.tools import ToolBox


ROOT = Path(__file__).resolve().parent.parent


def collect_evidence(toolbox, review):
    return {
        "reviewer_history": dispatch_tool_call(
            toolbox, review, "reviewer_history_lookup", {"user_id": review["user_id"]}),
        "business_trend": dispatch_tool_call(
            toolbox, review, "business_trend_lookup",
            {"prod_id": review["prod_id"], "around_date": review["date"]}),
        "reviewer_text_similarity": dispatch_tool_call(
            toolbox, review, "text_similarity_check", {
                "review_text": review["review_text"], "compare_against": "reviewer",
                "target_id": review["user_id"]}),
        "business_text_similarity": dispatch_tool_call(
            toolbox, review, "text_similarity_check", {
                "review_text": review["review_text"], "compare_against": "business",
                "target_id": review["prod_id"]}),
    }


def summarize(rows, pred_col):
    done = [r for r in rows if r.get(pred_col) is not None]
    metrics = binary_classification_metrics(
        [r["true_filtered"] for r in done], [r[pred_col] for r in done])
    metrics["attempted"] = len(rows)
    metrics["completed"] = len(done)
    positive_support = metrics["true_positive"] + metrics["false_negative"]
    metrics["recall_95ci"] = (
        wilson_interval(metrics["true_positive"], positive_support) if positive_support else None)
    return metrics


def load_baseline():
    labels = pd.read_csv(ROOT / "data" / "manual_batch.csv")[["review_id", "filtered"]]
    judgments = pd.read_csv(ROOT / "data" / "manual_batch_judgments.csv")[["review_id", "predicted_filtered"]]
    merged = labels.merge(judgments, on="review_id", how="left", validate="one_to_one")
    return {
        int(row.review_id): {
            "true_filtered": bool(row.filtered),
            "baseline_predicted": bool(row.predicted_filtered) if pd.notna(row.predicted_filtered) else None,
        }
        for row in merged.itertuples(index=False)
    }


def run(limit=None, output_dir=None, model="jev-latest", delay=0.0):
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise RuntimeError("Set TYPESAFE_API_KEY before running; no request was sent.")
    batch = pd.read_csv(ROOT / "data" / "manual_batch.csv")
    judgments = pd.read_csv(ROOT / "data" / "manual_batch_judgments.csv")
    batch = batch.merge(judgments[["review_id", "predicted_filtered"]], on="review_id", how="left", validate="one_to_one")
    if limit:
        batch = batch.head(limit)
    index = load_index()
    toolbox = ToolBox(index)
    rows = []
    for row in batch.to_dict(orient="records"):
        review = {
            "review_id": int(row["review_id"]), "user_id": str(int(row["user_id"])),
            "prod_id": str(int(row["prod_id"])), "rating": float(row["rating"]),
            "date": str(row["date"]), "review_text": str(row["review_text"]),
        }
        evidence = collect_evidence(toolbox, review)
        try:
            result = classify_review(review, evidence, model=model)
            err = None
        except Exception as exc:
            result, err = {}, f"{type(exc).__name__}: {exc}"
        rows.append({
            "review_id": review["review_id"], "true_filtered": bool(row["filtered"]),
            "baseline_predicted": bool(row["predicted_filtered"]) if pd.notna(row["predicted_filtered"]) else None,
            "jev_predicted": result.get("predicted_filtered"), "jev_confidence": result.get("confidence"),
            "jev_probability_filtered": result.get("probability_filtered"),
            "jev_model": result.get("model"), "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"), "cost_usd": result.get("cost_usd"),
            "elapsed_ms": result.get("elapsed_ms"), "error": err,
        })
        if delay:
            import time
            time.sleep(delay)
    baseline_metrics = summarize(rows, "baseline_predicted")
    jev_metrics = summarize(rows, "jev_predicted")
    output = Path(output_dir or ROOT / "data" / "jev_experiment_runs")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output / f"jev_manual_batch_{stamp}.csv"
    json_path = output / f"jev_manual_batch_{stamp}.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report = {
        "experiment": "jev_recall_experiment", "run_at_utc": stamp,
        "dataset": "data/manual_batch.csv", "n": len(rows),
        "dataset_note": "Exploratory only: this batch is part of data/sample.csv and contains only six filtered labels.",
        "baseline": {"name": "recorded manual Claude judgments", "metrics": baseline_metrics},
        "candidate": {"name": "Jev Model over shared local evidence", "model": model, "metrics": jev_metrics},
        "recall_delta": (jev_metrics["recall"] - baseline_metrics["recall"]
                         if jev_metrics.get("recall") is not None and baseline_metrics.get("recall") is not None else None),
        "cost_usd_reported": (sum(r["cost_usd"] for r in rows if r.get("cost_usd") is not None)
                              if any(r.get("cost_usd") is not None for r in rows) else None),
        "successful_calls": sum(r.get("jev_predicted") is not None for r in rows),
        "failed_calls": sum(r.get("error") is not None for r in rows),
        "cost_reporting_note": "Null means Jev did not return per-call cost; check account usage before estimating total cost.",
        "outputs": {"per_review_csv": str(csv_path), "summary_json": str(json_path)},
        "rows": rows,
    }
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, default=str))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Small smoke run; omit for all 50 fixed reviews")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional seconds between calls")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    run(limit=args.limit, output_dir=args.output_dir, model=args.model, delay=args.delay)
