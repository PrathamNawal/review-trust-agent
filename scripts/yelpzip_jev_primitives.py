"""Paired Choice-versus-Score Jev comparison on a development subset only."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dashboard.jev_model import compare_choice_and_score
from dashboard.metrics import binary_classification_metrics, wilson_interval
from scripts.yelpzip_jev_eval import (
    PROJECT_ESTIMATED_SPEND_CEILING_USD,
    PUBLIC_INPUT_PRICE_PER_MILLION,
    YelpZipEvidence,
    _load_local_typesafe_key,
    _project_estimated_spend,
    collect_evidence,
)
from scripts.yelpzip_eval_data import DB_PATH, OUTPUT_DIR


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "dev-choice-score-300-v1"
SCORE_THRESHOLD = 0.5
DEFAULT_LIMIT = 300
CHECKPOINT_EVERY = 100


def _metrics(rows, prediction_key, score_key=None):
    latest = {row["review_id"]: row for row in rows}
    rows = list(latest.values())
    completed = [row for row in rows if row.get("error") is None]
    labels = [bool(row["true_filtered"]) for row in completed]
    if score_key:
        predictions = [float(row[score_key]) >= SCORE_THRESHOLD for row in completed]
    else:
        predictions = [bool(row[prediction_key]) for row in completed]
    result = binary_classification_metrics(labels, predictions)
    positive_count = result["true_positive"] + result["false_negative"]
    correct = result["true_positive"] + result["true_negative"]
    result["recall_95ci"] = wilson_interval(result["true_positive"], positive_count) if positive_count else None
    result["accuracy_95ci"] = wilson_interval(correct, result["n"]) if result["n"] else None
    result["queue_rate"] = sum(predictions) / len(predictions) if predictions else None
    result["completed"] = len(completed)
    result["errors"] = len(rows) - len(completed)
    return result


def _summary(rows, spend, planned):
    completed = [row for row in rows if row.get("error") is None]
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "stage": "development_only",
        "dataset": "first 300 cases from frozen 500-case development partition",
        "planned_reviews": planned,
        "paired_reviews_completed": len({row["review_id"] for row in completed}),
        "model": "jev-1.13.0",
        "decision_rule": f"Choice uses Jev's filtered/recommended label; Score flags at score >= {SCORE_THRESHOLD:.2f}.",
        "score_cutoff_frozen_before_run": SCORE_THRESHOLD,
        "results": {
            "choice": _metrics(rows, "choice_filtered"),
            "score": _metrics(rows, "score_filtered", "score"),
        },
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in completed),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in completed),
        "estimated_project_spend_usd": round(spend, 4),
        "internal_spend_ceiling_usd": PROJECT_ESTIMATED_SPEND_CEILING_USD,
        "artifacts": {
            "records": f"data/yelpzip_eval/runs/{RUN_ID}.jsonl",
            "summary": f"data/yelpzip_eval/runs/{RUN_ID}.summary.json",
        },
        "limitations": [
            "Development data only; not final or generalization evidence.",
            "The first 300 development cases contain 40 YelpZip filtered labels.",
            "Choice confidence and Score values are uncalibrated model outputs.",
            "Choice and Score were asked in the same Jev request on identical evidence.",
        ],
    }


def run(limit=DEFAULT_LIMIT, delay=0.25, model="jev-1.13.0"):
    if not 1 <= limit <= DEFAULT_LIMIT:
        raise ValueError(f"Limit must be between 1 and {DEFAULT_LIMIT}; this runner is dev-only.")
    if not _load_local_typesafe_key():
        raise RuntimeError("Set TYPESAFE_API_KEY in the ignored local .env or process environment; no Jev requests were sent.")

    ids_path = OUTPUT_DIR / "development_ids.csv"
    labels_path = OUTPUT_DIR / "development_labels.csv"
    if not ids_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Frozen YelpZip development manifests are missing.")
    ids = [int(value) for value in pd.read_csv(ids_path)["review_id"].tolist()[:limit]]
    labels = pd.read_csv(labels_path).set_index("review_id")["true_filtered"].astype(bool).to_dict()
    if len(ids) != limit or any(review_id not in labels for review_id in ids):
        raise ValueError("Development IDs and labels do not match the requested subset.")

    run_dir = OUTPUT_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{RUN_ID}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()] if output_path.exists() else []
    subset_hash = hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()
    if any(row.get("subset_sha256") != subset_hash for row in rows):
        raise ValueError("Development subset changed under this run ID; create a new run ID.")
    done = {row["review_id"] for row in rows if row.get("error") is None}
    spend = _project_estimated_spend()
    store = YelpZipEvidence(DB_PATH)
    calls_since_checkpoint = 0
    try:
        with output_path.open("a", encoding="utf-8") as stream:
            for review_id in ids:
                if review_id in done:
                    continue
                if spend >= PROJECT_ESTIMATED_SPEND_CEILING_USD:
                    raise RuntimeError("Estimated Jev spend reached the $4.50 internal stop under the $5 budget.")
                review = store.get_review(review_id)
                evidence = collect_evidence(store, review)
                started = time.monotonic()
                try:
                    result = compare_choice_and_score(review, evidence, model=model)
                    error = None
                except Exception as exc:
                    result, error = {}, f"{type(exc).__name__}: {exc}"
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": RUN_ID,
                    "review_id": review_id,
                    "true_filtered": labels[review_id],
                    "choice_filtered": result.get("choice_filtered"),
                    "choice_confidence": result.get("choice_confidence"),
                    "choice_probabilities": result.get("choice_probabilities"),
                    "score": result.get("score"),
                    "score_filtered": (
                        result.get("score") >= SCORE_THRESHOLD if result.get("score") is not None else None
                    ),
                    "score_confidence": result.get("score_confidence"),
                    "score_probabilities": result.get("score_probabilities"),
                    "model": result.get("model", model),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "elapsed_ms": result.get("elapsed_ms", round((time.monotonic() - started) * 1000)),
                    "subset_sha256": subset_hash,
                    "error": error,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                if error is None:
                    done.add(review_id)
                    spend += (result.get("input_tokens") or 0) * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
                calls_since_checkpoint += 1
                if delay:
                    time.sleep(delay)
                if calls_since_checkpoint >= CHECKPOINT_EVERY:
                    summary_path.write_text(json.dumps(_summary(rows, spend, limit), indent=2))
                    calls_since_checkpoint = 0
                    print(f"Checkpoint: {len(done)}/{limit} reviews complete; estimated project spend ${spend:.4f}.", flush=True)
        summary_path.write_text(json.dumps(_summary(rows, spend, limit), indent=2))
    finally:
        store.close()
    print(summary_path.read_text())
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--model", default="jev-1.13.0")
    args = parser.parse_args()
    run(args.limit, args.delay, args.model)


if __name__ == "__main__":
    main()
