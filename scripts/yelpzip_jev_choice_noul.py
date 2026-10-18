"""Paired Choice-versus-Noul comparison on the frozen YelpZip dev subset."""

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from dashboard.jev_model import compare_choice_and_noul
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


RUN_ID = "dev-choice-noul-300-v1"
LIMIT = 300
CHOICE_THRESHOLD = None
NOUL_OVERALL_THRESHOLD = 0.5
BUSINESS_NO_BURST_THRESHOLD = 0.98
CHECKPOINT_EVERY = 100


def _metrics(rows, prediction):
    latest = {row["review_id"]: row for row in rows}
    rows = list(latest.values())
    completed = [row for row in rows if row.get("error") is None]
    labels = [bool(row["true_filtered"]) for row in completed]
    predicted = [bool(prediction(row)) for row in completed]
    result = binary_classification_metrics(labels, predicted)
    positives = result["true_positive"] + result["false_negative"]
    correct = result["true_positive"] + result["true_negative"]
    result["recall_95ci"] = wilson_interval(result["true_positive"], positives) if positives else None
    result["accuracy_95ci"] = wilson_interval(correct, result["n"]) if result["n"] else None
    result["queue_rate"] = sum(predicted) / len(predicted) if predicted else None
    result["completed"] = len(completed)
    result["errors"] = len(rows) - len(completed)
    return result


def _summary(rows, planned, spend):
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "stage": "development_only",
        "dataset": "same first 300 reviews from frozen development partition used in Choice-vs-Score run",
        "planned_reviews": planned,
        "paired_reviews_completed": len({row["review_id"] for row in rows if row.get("error") is None}),
        "model": "jev-1.13.0",
        "policies": {
            "choice": "Use Jev's filtered/recommended Choice label.",
            "noul_overall": f"Flag when overall_risk Noul >= {NOUL_OVERALL_THRESHOLD:.2f}.",
            "noul_business_policy": (
                f"Flag when 1 - business_burst Noul >= {BUSINESS_NO_BURST_THRESHOLD:.2f}."
            ),
        },
        "results": {
            "choice": _metrics(rows, lambda row: row["choice_filtered"]),
            "noul_overall_risk_at_0_50": _metrics(
                rows, lambda row: row["noul_overall_risk"] >= NOUL_OVERALL_THRESHOLD
            ),
            "noul_business_nonburst_at_0_98": _metrics(
                rows, lambda row: 1.0 - row["noul_business_burst"] >= BUSINESS_NO_BURST_THRESHOLD
            ),
        },
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows if row.get("error") is None),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows if row.get("error") is None),
        "estimated_project_spend_usd": round(spend, 4),
        "internal_spend_ceiling_usd": PROJECT_ESTIMATED_SPEND_CEILING_USD,
        "artifacts": {
            "records": f"data/yelpzip_eval/runs/{RUN_ID}.jsonl",
            "summary": f"data/yelpzip_eval/runs/{RUN_ID}.summary.json",
        },
        "limitations": [
            "Development data only; not final or generalization evidence.",
            "The 300-case subset has 40 YelpZip filtered labels.",
            "Noul values are uncalibrated model outputs; the chosen cutoffs are policies, not confidence claims.",
            "The direct Noul and 0.98 business non-burst policy answer different decision questions.",
        ],
    }


def run(limit=LIMIT, delay=0.25, model="jev-1.13.0"):
    if limit != LIMIT:
        raise ValueError(f"This paired comparison is frozen to the same {LIMIT}-review development subset.")
    if not _load_local_typesafe_key():
        raise RuntimeError("Set TYPESAFE_API_KEY in the ignored local .env or process environment; no Jev requests were sent.")
    ids_path = OUTPUT_DIR / "development_ids.csv"
    labels_path = OUTPUT_DIR / "development_labels.csv"
    if not ids_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError("Frozen YelpZip development manifests are missing.")
    ids = [int(value) for value in pd.read_csv(ids_path)["review_id"].tolist()[:LIMIT]]
    labels = pd.read_csv(labels_path).set_index("review_id")["true_filtered"].astype(bool).to_dict()
    if len(ids) != LIMIT or any(review_id not in labels for review_id in ids):
        raise ValueError("Development IDs and labels do not match the requested subset.")

    run_dir = OUTPUT_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{RUN_ID}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()] if output_path.exists() else []
    done = {row["review_id"] for row in rows if row.get("error") is None}
    spend = _project_estimated_spend()
    store = YelpZipEvidence(DB_PATH)
    since_checkpoint = 0
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
                    result = compare_choice_and_noul(review, evidence, model=model)
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
                    "noul_overall_risk": result.get("noul_overall_risk"),
                    "noul_overall_filtered": (
                        result.get("noul_overall_risk", -1) >= NOUL_OVERALL_THRESHOLD
                        if result.get("noul_overall_risk") is not None else None
                    ),
                    "noul_business_burst": result.get("noul_business_burst"),
                    "noul_business_nonburst_filtered": (
                        1.0 - result["noul_business_burst"] >= BUSINESS_NO_BURST_THRESHOLD
                        if result.get("noul_business_burst") is not None else None
                    ),
                    "model": result.get("model", model),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "elapsed_ms": result.get("elapsed_ms", round((time.monotonic() - started) * 1000)),
                    "error": error,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                rows.append(row)
                if error is None:
                    done.add(review_id)
                    spend += (result.get("input_tokens") or 0) * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
                since_checkpoint += 1
                if delay:
                    time.sleep(delay)
                if since_checkpoint >= CHECKPOINT_EVERY:
                    summary_path.write_text(json.dumps(_summary(rows, LIMIT, spend), indent=2))
                    print(f"Checkpoint: {len(done)}/{LIMIT} paired reviews complete; estimated spend ${spend:.4f}.", flush=True)
                    since_checkpoint = 0
        summary_path.write_text(json.dumps(_summary(rows, LIMIT, spend), indent=2))
    finally:
        store.close()
    print(summary_path.read_text())
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--model", default="jev-1.13.0")
    args = parser.parse_args()
    run(args.limit, args.delay, args.model)


if __name__ == "__main__":
    main()
