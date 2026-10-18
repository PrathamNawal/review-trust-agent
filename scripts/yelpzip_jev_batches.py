"""Run resumable 50-review paired Jev batches on the development partition.

The baseline uses Jev's single binary Choice. The candidate asks five focused
Noul questions in one request and stores the overall and per-signal probabilities.
Thresholds are tuned only on development batches; the final holdout is rejected.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dashboard.jev_model import (
    DEFAULT_DECISION_INSTRUCTIONS,
    DECOMPOSED_QUESTIONS,
    classify_review,
    classify_review_dimensions,
)
from dashboard.metrics import binary_classification_metrics, wilson_interval
from scripts.yelpzip_jev_eval import (
    PROJECT_ESTIMATED_SPEND_CEILING_USD,
    PUBLIC_INPUT_PRICE_PER_MILLION,
    YelpZipEvidence,
    _load_local_typesafe_key,
    collect_evidence,
)
from scripts.yelpzip_eval_data import DB_PATH, OUTPUT_DIR


ROOT = Path(__file__).resolve().parents[1]
BASELINE_VERSION = "jev-v1"
CANDIDATE_VERSION = "jev-decomposed-v1"
CONFIG_PATH = ROOT / "experiments" / "prompts" / "jev-decomposed-v1.json"
POLICY_PATH = ROOT / "experiments" / "prompts" / "jev-business-nonburst-policy-v1.json"


def _prior_spend(run_path):
    tokens = 54031
    directory = OUTPUT_DIR / "runs"
    if directory.exists():
        for path in directory.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if line.strip():
                    tokens += int(json.loads(line).get("input_tokens") or 0)
    return tokens * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000


def _latest(rows, arm, batch_index=None):
    latest = {}
    for row in rows:
        if row["arm"] != arm or (batch_index is not None and row["batch_index"] != batch_index):
            continue
        latest[row["review_id"]] = row
    return list(latest.values())


def _score(records, score_key=None, threshold=0.5):
    done = [r for r in records if r.get("predicted_filtered") is not None]
    if score_key is not None:
        done = [r for r in done if r.get("risk_scores", {}).get(score_key) is not None]
        predicted = [r["risk_scores"][score_key] >= threshold for r in done]
    else:
        predicted = [bool(r["predicted_filtered"]) for r in done]
    labels = [bool(r["true_filtered"]) for r in done]
    metrics = binary_classification_metrics(labels, predicted)
    positives = metrics["true_positive"] + metrics["false_negative"]
    correct = metrics["true_positive"] + metrics["true_negative"]
    metrics["recall_95ci"] = wilson_interval(metrics["true_positive"], positives) if positives else None
    metrics["accuracy_95ci"] = wilson_interval(correct, metrics["n"]) if metrics["n"] else None
    metrics["queue_rate"] = sum(predicted) / len(predicted) if predicted else None
    metrics["completed"] = len(done)
    metrics["errors"] = len(records) - len(done)
    return metrics


def _best_threshold(records, score_key, accuracy_floor=0.80):
    done = [r for r in records if r.get("predicted_filtered") is not None
            and r.get("risk_scores", {}).get(score_key) is not None]
    values = sorted({0.0, 1.01, *(float(r["risk_scores"][score_key]) for r in done)})
    options = []
    for threshold in values:
        metrics = _score(done, score_key, threshold)
        if metrics["accuracy"] is not None and metrics["accuracy"] >= accuracy_floor:
            options.append((metrics["recall"] or 0.0, metrics["accuracy"],
                            -(metrics["queue_rate"] or 0.0), threshold, metrics))
    if not options:
        return {"threshold": None, "note": f"No threshold met accuracy >= {accuracy_floor:.0%}."}
    recall, accuracy, _, threshold, metrics = max(options)
    return {"threshold": threshold, "recall": recall, "accuracy": accuracy, "metrics": metrics}


def _summary(rows, batch_index, run_id, spend, accuracy_floor):
    rows = [dict(r) for r in rows]
    for row in rows:
        probabilities = row.get("signal_probabilities") or {}
        if row.get("arm") == CANDIDATE_VERSION and "business_burst" in probabilities:
            row.setdefault("risk_scores", {})["business_no_burst"] = 1.0 - float(probabilities["business_burst"])
    summaries = {}
    for scope, selected_batch in (("this_batch", batch_index), ("cumulative_development", None)):
        arms = {
            BASELINE_VERSION: _latest(rows, BASELINE_VERSION, selected_batch),
            CANDIDATE_VERSION: _latest(rows, CANDIDATE_VERSION, selected_batch),
        }
        batch_summary = {BASELINE_VERSION: _score(arms[BASELINE_VERSION])}
        score_keys = ("overall_risk", "max_factor", "mean_factor", "business_no_burst")
        for key in score_keys:
            batch_summary[key] = _score(arms[CANDIDATE_VERSION], key, 0.5)
        batch_summary["best_development_thresholds"] = {
            key: _best_threshold(arms[CANDIDATE_VERSION], key, accuracy_floor)
            for key in score_keys
        }
        if scope == "this_batch":
            prior_candidates = [r for r in rows
                                if r["arm"] == CANDIDATE_VERSION
                                and r["batch_index"] < batch_index]
            batch_summary["prior_batch_thresholds_applied_here"] = {}
            for key in score_keys:
                prior = _best_threshold(prior_candidates, key, accuracy_floor)
                batch_summary["prior_batch_thresholds_applied_here"][key] = {
                    "selected_on_prior_batches": prior,
                    "current_batch_metrics": (
                        _score(arms[CANDIDATE_VERSION], key, prior["threshold"])
                        if prior.get("threshold") is not None else None
                    ),
                }
        summaries[scope] = batch_summary
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "stage": "development_tuning",
        "batch_index": batch_index,
        "baseline_version": BASELINE_VERSION,
        "candidate_version": CANDIDATE_VERSION,
        "candidate_config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "decision_policy_version": "jev-business-nonburst-policy-v1",
        "decision_policy_sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
        "model_requested": "jev-1.13.0",
        "target_recall": 0.50,
        "accuracy_floor": accuracy_floor,
        "spend_estimate_usd": round(spend, 4),
        "spend_ceiling_usd": PROJECT_ESTIMATED_SPEND_CEILING_USD,
        "results": summaries,
        "note": "Development-only tuning. Do not report as final performance; final holdout remains untouched.",
    }


def run_batch(db_path, ids_path, labels_path, batch_index, batch_size, run_id,
              model="jev-1.13.0", delay=0.25, accuracy_floor=0.80):
    if not _load_local_typesafe_key():
        raise RuntimeError("Set TYPESAFE_API_KEY in the ignored local .env or process environment; no Jev requests were sent.")
    if Path(ids_path).name != "development_ids.csv":
        raise ValueError("This runner only accepts development_ids.csv; final holdout is not available here.")
    ids = [int(v) for v in pd.read_csv(ids_path)["review_id"].tolist()]
    labels = pd.read_csv(labels_path).set_index("review_id")["true_filtered"].astype(bool).to_dict()
    if set(ids) != set(labels):
        raise ValueError("Development IDs and labels must match exactly.")
    start = batch_index * batch_size
    batch_ids = ids[start:start + batch_size]
    if not batch_ids:
        raise ValueError(f"Batch {batch_index} is past the end of the development manifest.")

    run_dir = OUTPUT_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{run_id}.jsonl"
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()] \
        if output_path.exists() else []
    config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    if any(r.get("config_sha256") != config_hash for r in rows):
        raise ValueError("Candidate config changed under an existing run ID; use a new run ID.")
    prompt_hash = hashlib.sha256(DEFAULT_DECISION_INSTRUCTIONS.encode()).hexdigest()
    done = {(r["review_id"], r["arm"]) for r in rows if r.get("error") is None}
    spend = _prior_spend(output_path)
    calls_since_checkpoint = 0
    store = YelpZipEvidence(db_path)
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for review_id in batch_ids:
                review = store.get_review(review_id)
                evidence = collect_evidence(store, review)
                for arm in (BASELINE_VERSION, CANDIDATE_VERSION):
                    if (review_id, arm) in done:
                        continue
                    if spend >= PROJECT_ESTIMATED_SPEND_CEILING_USD:
                        raise RuntimeError("Estimated Jev spend reached the $4.50 internal stop under the $5 budget.")
                    started = time.monotonic()
                    try:
                        if arm == BASELINE_VERSION:
                            result = classify_review(review, evidence, model=model)
                            scores = {
                                "overall_risk": result.get("probability_filtered"),
                                "max_factor": result.get("probability_filtered"),
                                "mean_factor": result.get("probability_filtered"),
                            }
                            predicted = result["predicted_filtered"]
                        else:
                            result = classify_review_dimensions(review, evidence, model=model)
                            scores = {
                                "overall_risk": result["probabilities"]["overall_risk"],
                                "max_factor": result["max_signal_probability"],
                                "mean_factor": result["mean_signal_probability"],
                            }
                            predicted = scores["overall_risk"] >= 0.5
                        error = None
                    except Exception as exc:
                        result, scores, predicted = {}, {}, None
                        error = f"{type(exc).__name__}: {exc}"
                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_id, "batch_index": batch_index,
                        "review_id": review_id, "arm": arm,
                        "true_filtered": labels[review_id],
                        "predicted_filtered": predicted,
                        "risk_scores": scores,
                        "signal_probabilities": result.get("probabilities"),
                        "model": result.get("model", model),
                        "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "elapsed_ms": result.get("elapsed_ms", round((time.monotonic() - started) * 1000)),
                        "config_sha256": config_hash,
                        "baseline_prompt_sha256": prompt_hash,
                        "error": error,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    rows.append(record)
                    if error is None:
                        done.add((review_id, arm))
                        spend += (result.get("input_tokens") or 0) * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
                    calls_since_checkpoint += 1
                    if delay:
                        time.sleep(delay)
                    if calls_since_checkpoint >= 100:
                        summary = _summary(rows, batch_index, run_id, spend, accuracy_floor)
                        output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
                        print(json.dumps(summary, indent=2), flush=True)
                        calls_since_checkpoint = 0
        summary = _summary(rows, batch_index, run_id, spend, accuracy_floor)
        output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--ids", default=str(OUTPUT_DIR / "development_ids.csv"))
    parser.add_argument("--labels", default=str(OUTPUT_DIR / "development_labels.csv"))
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--run-id", default="dev-v1-decomposed")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--accuracy-floor", type=float, default=0.80)
    args = parser.parse_args()
    run_batch(args.db, args.ids, args.labels, args.batch_index, args.batch_size,
              args.run_id, args.model, args.delay, args.accuracy_floor)


if __name__ == "__main__":
    main()
