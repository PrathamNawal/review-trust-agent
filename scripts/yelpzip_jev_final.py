"""Run a locked paired Jev comparison on the 3,000-review final holdout.

The decomposed candidate uses the pre-registered business non-burst score
1 - P(business_burst) with a fixed threshold. No threshold search is run here.
Labels are used only for the final report and are never included in API input.
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
IDS_PATH = OUTPUT_DIR / "final_holdout_ids.csv"
LABELS_PATH = OUTPUT_DIR / "final_holdout_labels.csv"
DECOMPOSED_CONFIG = ROOT / "experiments" / "prompts" / "jev-decomposed-v1.json"
POLICY_CONFIG = ROOT / "experiments" / "prompts" / "jev-business-nonburst-policy-v1.json"
BASELINE = "jev-v1"
CANDIDATE = "jev-decomposed-business-no-burst"


def _project_spend_from_logs():
    tokens = 54031
    runs_dir = OUTPUT_DIR / "runs"
    if runs_dir.exists():
        for path in runs_dir.glob("*.jsonl"):
            for line in path.read_text().splitlines():
                if line.strip():
                    tokens += int(json.loads(line).get("input_tokens") or 0)
    return tokens * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000


def _latest_rows(rows):
    latest = {}
    for row in rows:
        latest[(row["review_id"], row["arm"])] = row
    return latest


def _metrics(records, labels, predictions):
    valid = [r for r in records if r["review_id"] in predictions]
    truth = [bool(labels[r["review_id"]]) for r in valid]
    predicted = [bool(predictions[r["review_id"]]) for r in valid]
    result = binary_classification_metrics(truth, predicted)
    positives = result["true_positive"] + result["false_negative"]
    correct = result["true_positive"] + result["true_negative"]
    result["recall_95ci"] = wilson_interval(result["true_positive"], positives) if positives else None
    result["accuracy_95ci"] = wilson_interval(correct, result["n"]) if result["n"] else None
    result["queue_rate"] = sum(predicted) / len(predicted) if predicted else None
    result["completed"] = len(valid)
    result["errors"] = len(records) - len(valid)
    result["mean_latency_ms"] = (
        sum(r["elapsed_ms"] for r in valid if r.get("elapsed_ms") is not None)
        / max(1, sum(r.get("elapsed_ms") is not None for r in valid))
    )
    result["input_tokens"] = sum(r.get("input_tokens") or 0 for r in valid)
    result["output_tokens"] = sum(r.get("output_tokens") or 0 for r in valid)
    result["estimated_cost_usd"] = result["input_tokens"] * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
    return result


def _summarize(rows, ids, labels, threshold, spend_estimate, model):
    latest = _latest_rows(rows)
    base = [latest[(rid, BASELINE)] for rid in ids if (rid, BASELINE) in latest]
    cand = [latest[(rid, CANDIDATE)] for rid in ids if (rid, CANDIDATE) in latest]
    base_pred = {r["review_id"]: r["predicted_filtered"] for r in base
                 if r.get("error") is None and r.get("predicted_filtered") is not None}
    cand_pred = {r["review_id"]: r["business_no_burst_score"] >= threshold for r in cand
                 if r.get("error") is None and r.get("business_no_burst_score") is not None}
    paired_ids = set(base_pred) & set(cand_pred)
    paired_base = [r for r in base if r["review_id"] in paired_ids]
    paired_cand = [r for r in cand if r["review_id"] in paired_ids]
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": "final-v1-decomposed-threshold-098",
        "stage": "final_locked",
        "model_requested": model,
        "holdout_manifest_sha256": hashlib.sha256(IDS_PATH.read_bytes()).hexdigest(),
        "baseline_config_sha256": hashlib.sha256(DEFAULT_DECISION_INSTRUCTIONS.encode()).hexdigest(),
        "candidate_config_sha256": hashlib.sha256(DECOMPOSED_CONFIG.read_bytes()).hexdigest(),
        "decision_policy": "1 - business_burst_probability >= frozen threshold",
        "decision_policy_config_sha256": hashlib.sha256(POLICY_CONFIG.read_bytes()).hexdigest(),
        "threshold": threshold,
        "requested_reviews": len(ids),
        "completed_pairs": len(paired_ids),
        "failed_or_incomplete_pairs": len(ids) - len(paired_ids),
        "results": {
            BASELINE: _metrics(base, labels, base_pred),
            CANDIDATE: _metrics(cand, labels, cand_pred),
            "paired_baseline": _metrics(paired_base, labels, base_pred),
            "paired_candidate": _metrics(paired_cand, labels, cand_pred),
        },
        "spend_estimate_project_usd": round(spend_estimate, 4),
        "spend_ceiling_usd": PROJECT_ESTIMATED_SPEND_CEILING_USD,
        "note": "Final holdout evaluated once at frozen threshold 0.98; no tuning was performed.",
    }


def run(run_id="final-v1-decomposed-threshold-098", threshold=0.98,
        model="jev-1.13.0", delay=0.25):
    if threshold != 0.98:
        raise ValueError("The locked final run uses the preselected threshold 0.98.")
    if not _load_local_typesafe_key():
        raise RuntimeError("Set TYPESAFE_API_KEY in the ignored local .env or process environment; no Jev requests were sent.")
    if not IDS_PATH.is_file() or not LABELS_PATH.is_file():
        raise FileNotFoundError("Frozen final-holdout IDs and labels are not available locally.")
    ids = [int(v) for v in pd.read_csv(IDS_PATH)["review_id"].tolist()]
    labels = pd.read_csv(LABELS_PATH).set_index("review_id")["true_filtered"].astype(bool).to_dict()
    if len(ids) != 3000 or len(set(ids)) != 3000 or set(ids) != set(labels):
        raise ValueError("Expected exactly 3,000 unique final-holdout IDs with matching scorer labels.")

    run_dir = OUTPUT_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{run_id}.jsonl"
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()] \
        if output_path.exists() else []
    config_hash = hashlib.sha256(DECOMPOSED_CONFIG.read_bytes()).hexdigest()
    if any(r.get("candidate_config_sha256") != config_hash for r in rows):
        raise ValueError("Candidate config changed under this final run ID; create a new run ID.")
    latest = _latest_rows(rows)
    done = {key for key, record in latest.items() if record.get("error") is None}
    spend_estimate = _project_spend_from_logs()
    store = YelpZipEvidence(DB_PATH)
    calls_since_status = 0
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for index, review_id in enumerate(ids, start=1):
                if all((review_id, arm) in done for arm in (BASELINE, CANDIDATE)):
                    continue
                review = store.get_review(review_id)
                evidence = collect_evidence(store, review)
                for arm in (BASELINE, CANDIDATE):
                    key = (review_id, arm)
                    if key in done:
                        continue
                    if spend_estimate >= PROJECT_ESTIMATED_SPEND_CEILING_USD:
                        raise RuntimeError("Estimated Jev spend reached the $4.50 internal stop under the $5 budget.")
                    started = time.monotonic()
                    try:
                        if arm == BASELINE:
                            result = classify_review(review, evidence, model=model)
                            policy_score = None
                            prediction = bool(result["predicted_filtered"])
                        else:
                            result = classify_review_dimensions(review, evidence, model=model)
                            policy_score = 1.0 - result["probabilities"]["business_burst"]
                            prediction = None
                        error = None
                    except Exception as exc:
                        result, policy_score, prediction = {}, None, None
                        error = f"{type(exc).__name__}: {exc}"
                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_id,
                        "review_id": review_id,
                        "arm": arm,
                        "true_filtered": labels[review_id],
                        "predicted_filtered": prediction,
                        "business_no_burst_score": policy_score,
                        "signal_probabilities": result.get("probabilities"),
                        "model": result.get("model", model),
                        "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "elapsed_ms": result.get("elapsed_ms", round((time.monotonic() - started) * 1000)),
                        "candidate_config_sha256": config_hash,
                        "error": error,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    rows.append(record)
                    if error is None:
                        done.add(key)
                        spend_estimate += (result.get("input_tokens") or 0) * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
                    calls_since_status += 1
                    if calls_since_status >= 100:
                        print(f"Completed {sum(1 for rid in ids if all((rid, a) in done for a in (BASELINE, CANDIDATE)))} / {len(ids)} paired reviews; performance remains sealed until the full run completes.", flush=True)
                        calls_since_status = 0
                    if delay:
                        time.sleep(delay)
        summary = _summarize(rows, ids, labels, threshold, spend_estimate, model)
        output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="final-v1-decomposed-threshold-098")
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()
    run(args.run_id, args.threshold, args.model, args.delay)


if __name__ == "__main__":
    main()
