"""Freeze and run a fresh stratified 3,000-review Choice-vs-Noul holdout."""

import argparse
import csv
import hashlib
import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

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
from scripts.yelpzip_eval_data import DB_PATH, OUTPUT_DIR, _previously_used_ids


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
RUN_ID = "final-choice-noul-fresh-3000-v1"
MANIFEST_PREFIX = "fresh_choice_noul_3000"
MANIFEST_IDS = OUTPUT_DIR / f"{MANIFEST_PREFIX}_ids.csv"
MANIFEST_LABELS = OUTPUT_DIR / f"{MANIFEST_PREFIX}_labels.csv"
MANIFEST_META = OUTPUT_DIR / f"{MANIFEST_PREFIX}_manifest.json"
RUN_LIMIT = 3000
SEED = 2026101803
NOUL_OVERALL_THRESHOLD = 0.5
BUSINESS_NO_BURST_THRESHOLD = 0.98
CHECKPOINT_EVERY = 100


def _ids_from_json(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "review_id":
                try:
                    found.add(int(item))
                except (TypeError, ValueError):
                    pass
            else:
                found.update(_ids_from_json(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_ids_from_json(item))
    return found


def _all_previously_used_ids():
    """Exclude known examples plus IDs in all existing manifests and run artifacts."""
    used = set(_previously_used_ids())
    for path in DATA_ROOT.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if "review_id" in (reader.fieldnames or []):
                    for row in reader:
                        try:
                            used.add(int(row["review_id"]))
                        except (TypeError, ValueError):
                            continue
        except (OSError, UnicodeError, csv.Error):
            continue
    for pattern in ("*.jsonl", "*.json"):
        for path in DATA_ROOT.rglob(pattern):
            try:
                if path.suffix == ".jsonl":
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            used.update(_ids_from_json(json.loads(line)))
                else:
                    used.update(_ids_from_json(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
    return used


def freeze_manifest(size=RUN_LIMIT):
    """Create a stratified holdout once; refuse to overwrite any frozen artifact."""
    existing = [path.exists() for path in (MANIFEST_IDS, MANIFEST_LABELS, MANIFEST_META)]
    if any(existing):
        if not all(existing):
            raise RuntimeError("Fresh holdout files are incomplete; refusing to resample or overwrite them.")
        meta = json.loads(MANIFEST_META.read_text(encoding="utf-8"))
        if meta.get("size") != size or meta.get("run_id") != RUN_ID:
            raise RuntimeError("Existing fresh holdout manifest does not match this frozen run configuration.")
        return meta
    if size != RUN_LIMIT:
        raise ValueError(f"This final evaluation is frozen to {RUN_LIMIT} reviews.")

    used = _all_previously_used_ids()
    con = sqlite3.connect(DB_PATH)
    try:
        available = {0: [], 1: []}
        for review_id, label in con.execute("SELECT review_id,label FROM reviews ORDER BY review_id"):
            if int(review_id) not in used:
                available[int(label)].append(int(review_id))
    finally:
        con.close()
    population = len(available[0]) + len(available[1])
    if population < size:
        raise RuntimeError("Not enough unused YelpZip reviews remain for a fresh holdout.")
    positive_n = round(size * len(available[1]) / population)
    rng = random.Random(SEED)
    selected = rng.sample(available[1], positive_n) + rng.sample(available[0], size - positive_n)
    rng.shuffle(selected)
    selected_positive = set()
    labels = {}
    # Recover the selected positive IDs deterministically from the ordered selection groups.
    # The stratified counts are stored as labels separately from all inference inputs.
    positive_pool = set(available[1])
    for review_id in selected:
        labels[review_id] = int(review_id in positive_pool)
        if labels[review_id]:
            selected_positive.add(review_id)
    ids_hash = hashlib.sha256("\n".join(map(str, selected)).encode()).hexdigest()
    label_rows = [(review_id, labels[review_id]) for review_id in selected]
    MANIFEST_IDS.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_IDS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["review_id"])
        writer.writerows((review_id,) for review_id in selected)
    with MANIFEST_LABELS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["review_id", "true_filtered"])
        writer.writerows(label_rows)
    source_summary_path = DATA_ROOT / "yelpzip_eval" / "source_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.is_file() else {}
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "size": size,
        "seed": SEED,
        "stratified_positive_count": len(selected_positive),
        "stratified_negative_count": size - len(selected_positive),
        "unused_population_positive": len(available[1]),
        "unused_population_negative": len(available[0]),
        "excluded_prior_unique_review_ids": len(used),
        "source_review_count": source_summary.get("rows"),
        "source_file_sha256": source_summary.get("source_file_sha256"),
        "ids_sha256": ids_hash,
        "selection": "Stratified random sample from YelpZip IDs not present in previous CSV, JSON, JSONL, manifests, run records, or documented examples.",
        "inference_label_separation": "Labels are stored separately and are not included in review/evidence API payloads.",
        "model": "jev-1.13.0",
        "policies_frozen_before_calls": {
            "choice": "Use Jev filtered/recommended choice.",
            "noul_overall": f"Flag at >= {NOUL_OVERALL_THRESHOLD:.2f}.",
            "noul_business_nonburst": f"Flag at 1 - P(business_burst) >= {BUSINESS_NO_BURST_THRESHOLD:.2f}.",
        },
    }
    MANIFEST_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


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


def _summary(rows, manifest, spend):
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "stage": "fresh_locked_holdout",
        "manifest": str(MANIFEST_META.relative_to(ROOT)),
        "manifest_ids_sha256": manifest["ids_sha256"],
        "planned_reviews": RUN_LIMIT,
        "paired_reviews_completed": len({row["review_id"] for row in rows if row.get("error") is None}),
        "filtered_label_count": manifest["stratified_positive_count"],
        "model": "jev-1.13.0",
        "policies": manifest["policies_frozen_before_calls"],
        "results": {
            "choice": _metrics(rows, lambda row: row["choice_filtered"]),
            "noul_overall_risk_at_0_50": _metrics(rows, lambda row: row["noul_overall_filtered"]),
            "noul_business_nonburst_at_0_98": _metrics(rows, lambda row: row["noul_business_nonburst_filtered"]),
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
            "YelpZip filtered labels are benchmark labels, not verified fraud ground truth.",
            "Noul values and the resulting decisions are specific to the stated frozen cutoffs.",
            "Choice and Noul were queried together to make the comparison paired; repeated runs may vary.",
        ],
    }


def run(delay=0.25):
    if not _load_local_typesafe_key():
        raise RuntimeError("Set TYPESAFE_API_KEY in the ignored local .env or process environment; no Jev requests were sent.")
    manifest = freeze_manifest()
    ids = [int(value) for value in pd.read_csv(MANIFEST_IDS)["review_id"].tolist()]
    label_frame = pd.read_csv(MANIFEST_LABELS).set_index("review_id")
    labels = label_frame["true_filtered"].astype(bool).to_dict()
    if len(ids) != RUN_LIMIT or len(set(ids)) != RUN_LIMIT or set(ids) != set(labels):
        raise ValueError("Fresh holdout IDs and labels do not align with the frozen manifest.")
    actual_hash = hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()
    if actual_hash != manifest["ids_sha256"]:
        raise ValueError("Fresh holdout ID manifest hash does not match its frozen metadata.")

    run_dir = OUTPUT_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{RUN_ID}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")
    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()] if output_path.exists() else []
    if any(row.get("manifest_ids_sha256") != actual_hash for row in rows):
        raise ValueError("Run records do not match the frozen holdout manifest.")
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
                    result = compare_choice_and_noul(review, evidence, model="jev-1.13.0")
                    error = None
                except Exception as exc:
                    result, error = {}, f"{type(exc).__name__}: {exc}"
                row = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "run_id": RUN_ID,
                    "manifest_ids_sha256": actual_hash,
                    "review_id": review_id,
                    "true_filtered": labels[review_id],
                    "choice_filtered": result.get("choice_filtered"),
                    "choice_confidence": result.get("choice_confidence"),
                    "choice_probabilities": result.get("choice_probabilities"),
                    "noul_overall_risk": result.get("noul_overall_risk"),
                    "noul_overall_filtered": (
                        result["noul_overall_risk"] >= NOUL_OVERALL_THRESHOLD
                        if result.get("noul_overall_risk") is not None else None
                    ),
                    "noul_business_burst": result.get("noul_business_burst"),
                    "noul_business_nonburst_filtered": (
                        1.0 - result["noul_business_burst"] >= BUSINESS_NO_BURST_THRESHOLD
                        if result.get("noul_business_burst") is not None else None
                    ),
                    "model": result.get("model", "jev-1.13.0"),
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
                    summary_path.write_text(json.dumps(_summary(rows, manifest, spend), indent=2))
                    print(f"Checkpoint: {len(done)}/{RUN_LIMIT} paired cases complete; estimated spend ${spend:.4f}.", flush=True)
                    since_checkpoint = 0
        summary_path.write_text(json.dumps(_summary(rows, manifest, spend), indent=2))
    finally:
        store.close()
    print(summary_path.read_text())
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--prepare-only", action="store_true", help="Freeze the fresh manifest without API calls.")
    args = parser.parse_args()
    if args.prepare_only:
        print(json.dumps(freeze_manifest(), indent=2))
    else:
        run(args.delay)


if __name__ == "__main__":
    main()
