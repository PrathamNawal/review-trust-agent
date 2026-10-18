"""Run resumable paired Jev prompt evaluations on frozen local YelpZip IDs.

Each prompt comparison makes 100 logical Jev calls: previous and candidate
instructions on the same 50 tuning reviews. Results and labels stay in the
Git-ignored data/yelpzip_eval/runs/ directory.
"""

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dashboard.jev_model import DEFAULT_DECISION_INSTRUCTIONS, classify_review
from dashboard.metrics import binary_classification_metrics, wilson_interval
from scripts.yelpzip_eval_data import DB_PATH, OUTPUT_DIR


CHECKPOINT_EVERY_CALLS = 100
MAX_SIMILARITY_TEXTS = 300
PUBLIC_INPUT_PRICE_PER_MILLION = 0.042
PROJECT_ESTIMATED_SPEND_CEILING_USD = 4.50  # keep $0.50 headroom under the user's $5 cap
PRIOR_RUN_INPUT_TOKENS = 54031  # measured in the earlier 50-review Jev run
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_typesafe_key():
    """Load only TYPESAFE_API_KEY from the ignored local .env without printing it."""
    import os

    if os.environ.get("TYPESAFE_API_KEY"):
        return True
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return False
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("TYPESAFE_API_KEY="):
            value = stripped.split("=", 1)[1].strip().strip("\"'")
            if value:
                os.environ["TYPESAFE_API_KEY"] = value
                return True
    return False


class YelpZipEvidence:
    """SQLite-backed, label-blind and time-aware feature lookups."""

    def __init__(self, db_path):
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row
        self.current = None

    def close(self):
        self.con.close()

    def set_current(self, review):
        self.current = review

    def get_review(self, review_id):
        row = self.con.execute(
            "SELECT review_id,user_id,prod_id,rating,date,review_text FROM reviews WHERE review_id=?",
            (int(review_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown YelpZip review ID: {review_id}")
        return {"review_id": row["review_id"], "user_id": str(row["user_id"]),
                "prod_id": str(row["prod_id"]), "rating": row["rating"],
                "date": row["date"], "review_text": row["review_text"]}

    def get_label(self, review_id):
        row = self.con.execute("SELECT label FROM reviews WHERE review_id=?", (int(review_id),)).fetchone()
        if row is None:
            raise KeyError(f"Unknown YelpZip review ID: {review_id}")
        return bool(row["label"])

    def _past_rows(self, entity_column, entity_id, exclude_review_id=None):
        if entity_column not in ("user_id", "prod_id"):
            raise ValueError("Unsupported entity column")
        current = self.current
        if current is None:
            raise RuntimeError("Set the target review before querying evidence")
        excluded_id = int(current["review_id"])
        if exclude_review_id is not None:
            excluded_id = int(exclude_review_id)
        query = (f"SELECT review_id,rating,date FROM reviews "
                 f"WHERE {entity_column}=? AND date<? AND review_id<>? ORDER BY date,review_id")
        return self.con.execute(query, (int(entity_id), current["date"], excluded_id)).fetchall()

    def reviewer_history_lookup(self, user_id, exclude_review_id=None):
        rows = self._past_rows("user_id", user_id, exclude_review_id)
        if not rows:
            return {"user_id": str(user_id), "other_review_count": 0,
                    "note": "No earlier reviews found for this reviewer."}
        dates = [pd.Timestamp(r["date"]) for r in rows]
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        max_in_week = 1
        for i, date in enumerate(dates):
            max_in_week = max(max_in_week, sum(0 <= (other - date).days <= 7 for other in dates[i:]))
        ratings = [float(r["rating"]) for r in rows]
        rating_std = pd.Series(ratings).std()
        return {
            "user_id": str(user_id), "other_review_count": len(rows),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "rating_std": round(float(rating_std), 2) if pd.notna(rating_std) else 0.0,
            "min_gap_days_between_reviews": min(gaps) if gaps else None,
            "max_reviews_in_any_7_day_window": max_in_week,
            "first_review_date": dates[0].strftime("%Y-%m-%d"),
            "last_review_date": dates[-1].strftime("%Y-%m-%d"),
        }

    def business_trend_lookup(self, prod_id, around_date, window_days=14, exclude_review_id=None):
        rows = self._past_rows("prod_id", prod_id, exclude_review_id)
        if not rows:
            return {"prod_id": str(prod_id), "other_review_count": 0,
                    "note": "No earlier reviews found for this business."}
        center = pd.Timestamp(around_date)
        past = [r for r in rows if pd.Timestamp(r["date"]) < center]
        windowed = [r for r in past if 0 <= (center - pd.Timestamp(r["date"])).days <= window_days]
        ratings = [float(r["rating"]) for r in past]
        window_ratings = [float(r["rating"]) for r in windowed]
        span = max((pd.Timestamp(past[-1]["date"]) - pd.Timestamp(past[0]["date"])).days, 1)
        baseline_rate = len(past) / span
        window_rate = len(windowed) / max(window_days, 1)
        return {
            "prod_id": str(prod_id), "other_review_count": len(past),
            "avg_rating_overall": round(sum(ratings) / len(ratings), 2),
            "reviews_in_window": len(windowed),
            "avg_rating_in_window": round(sum(window_ratings) / len(window_ratings), 2) if windowed else None,
            "window_days": window_days,
            "burst_ratio": round(window_rate / baseline_rate, 2) if baseline_rate > 0 else None,
        }

    def text_similarity_check(self, review_text, compare_against, target_id,
                              exclude_review_id=None, top_k=3):
        column = {"reviewer": "user_id", "business": "prod_id"}.get(compare_against)
        if column is None:
            raise ValueError("compare_against must be 'reviewer' or 'business'")
        if self.current is None:
            raise RuntimeError("Set the target review before querying evidence")
        rows = self.con.execute(
            f"SELECT review_text FROM reviews WHERE {column}=? AND date<? AND review_id<>? "
            "ORDER BY date DESC,review_id DESC LIMIT ?",
            (int(target_id), self.current["date"], int(self.current["review_id"]), MAX_SIMILARITY_TEXTS),
        ).fetchall()
        if not rows:
            return {"max_similarity": 0.0, "note": "No earlier comparison texts available.",
                    "similar_snippets": []}
        texts = [r["review_text"] for r in rows]
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(
            [review_text] + texts
        )
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        top = scores.argsort()[::-1][:top_k]
        return {
            "max_similarity": round(float(scores.max()), 3),
            "avg_similarity": round(float(scores.mean()), 3),
            "compared_against_n_reviews": len(rows),
            "similar_snippets": [{"similarity": round(float(scores[i]), 3),
                                  "snippet": texts[i][:150]} for i in top],
        }


def collect_evidence(store, review):
    store.set_current(review)
    return {
        "reviewer_history": store.reviewer_history_lookup(review["user_id"], review["review_id"]),
        "business_trend": store.business_trend_lookup(
            review["prod_id"], review["date"], exclude_review_id=review["review_id"]),
        "reviewer_text_similarity": store.text_similarity_check(
            review["review_text"], "reviewer", review["user_id"], review["review_id"]),
        "business_text_similarity": store.text_similarity_check(
            review["review_text"], "business", review["prod_id"], review["review_id"]),
    }


def _metrics(rows, version):
    # A resumed run may contain failed attempts followed by a successful retry.
    # Score only the latest record for each review and prompt version while
    # retaining the raw attempt log for auditability.
    latest = {}
    attempts = [r for r in rows if r["prompt_version"] == version]
    for row in attempts:
        latest[row["review_id"]] = row
    latest_rows = list(latest.values())
    completed = [r for r in latest_rows if r.get("predicted_filtered") is not None]
    metrics = binary_classification_metrics(
        [r["true_filtered"] for r in completed], [r["predicted_filtered"] for r in completed]
    )
    support = metrics["true_positive"] + metrics["false_negative"]
    metrics["recall_95ci"] = wilson_interval(metrics["true_positive"], support) if support else None
    metrics["attempted"] = len(latest_rows)
    metrics["request_attempts"] = len(attempts)
    metrics["completed"] = len(completed)
    metrics["completion_rate"] = len(completed) / metrics["attempted"] if metrics["attempted"] else None
    metrics["error_count"] = sum(r.get("predicted_filtered") is None for r in latest_rows)
    metrics["queue_rate"] = (
        sum(bool(r["predicted_filtered"]) for r in completed) / len(completed) if completed else None
    )
    latencies = [r["elapsed_ms"] for r in completed if r.get("elapsed_ms") is not None]
    metrics["mean_latency_ms"] = sum(latencies) / len(latencies) if latencies else None
    metrics["input_tokens"] = sum(r.get("input_tokens") or 0 for r in completed)
    metrics["output_tokens"] = sum(r.get("output_tokens") or 0 for r in completed)
    metrics["estimated_cost_usd"] = metrics["input_tokens"] * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
    return metrics


def _project_estimated_spend():
    input_tokens = PRIOR_RUN_INPUT_TOKENS
    run_dir = OUTPUT_DIR / "runs"
    if run_dir.exists():
        for path in run_dir.glob("*.jsonl"):
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        input_tokens += int(json.loads(line).get("input_tokens") or 0)
    return input_tokens * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000


def _enforce_spend_ceiling(estimate):
    if estimate >= PROJECT_ESTIMATED_SPEND_CEILING_USD:
        raise RuntimeError(
            f"Estimated project Jev spend is ${estimate:.2f}; stopping before the $4.50 "
            "internal ceiling to preserve headroom under the user's $5 budget."
        )


def run_pair(db_path, ids_path, labels_path, old_prompt_path, new_prompt_path,
             old_version, new_version, run_id, change_reason,
             stage="tuning", model="jev-latest", delay=0.25):
    if not _load_local_typesafe_key():
        raise RuntimeError("Set TYPESAFE_API_KEY in the local ignored .env or process environment; no Jev requests were sent.")
    ids = [int(v) for v in pd.read_csv(ids_path)["review_id"].tolist()]
    expected_manifest = {"tuning": "tuning_ids.csv", "development": "development_ids.csv",
                         "final": "final_holdout_ids.csv"}
    if stage not in expected_manifest or Path(ids_path).name != expected_manifest[stage]:
        raise ValueError(f"Stage {stage!r} requires {expected_manifest.get(stage)!r}.")
    labels = pd.read_csv(labels_path).set_index("review_id")["true_filtered"].astype(bool).to_dict()
    if set(ids) != set(labels):
        raise ValueError("Review IDs and scorer labels must match exactly.")
    old_text = Path(old_prompt_path).read_text() if old_prompt_path else DEFAULT_DECISION_INSTRUCTIONS
    new_text = Path(new_prompt_path).read_text()
    prompts = [(old_version, old_text), (new_version, new_text)]
    prompt_hashes = {version: hashlib.sha256(text.encode()).hexdigest() for version, text in prompts}
    if old_version == new_version or prompt_hashes[old_version] == prompt_hashes[new_version]:
        raise ValueError("Paired prompt versions must have distinct names and instruction text.")

    run_dir = OUTPUT_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{run_id}.jsonl"
    previous = []
    if output_path.exists():
        previous = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    for record in previous:
        expected_hash = prompt_hashes.get(record.get("prompt_version"))
        if expected_hash and record.get("prompt_sha256") != expected_hash:
            raise ValueError("A prompt changed under an existing version/run ID; create a new prompt version and run ID.")
    done = {(r["review_id"], r["prompt_version"], r["prompt_sha256"])
            for r in previous if r.get("predicted_filtered") is not None}
    store = YelpZipEvidence(db_path)
    calls_since_checkpoint = 0
    rows = list(previous)
    spend_estimate = _project_estimated_spend()
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for review_id in ids:
                review = store.get_review(review_id)
                evidence = collect_evidence(store, review)
                for version, instructions in prompts:
                    signature = (review_id, version, prompt_hashes[version])
                    if signature in done:
                        continue
                    _enforce_spend_ceiling(spend_estimate)
                    try:
                        result = classify_review(review, evidence, model=model, instructions=instructions)
                        error = None
                    except Exception as exc:
                        result, error = {}, f"{type(exc).__name__}: {exc}"
                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "review_id": review_id,
                        "prompt_version": version,
                        "prompt_sha256": prompt_hashes[version],
                        "true_filtered": labels[review_id],
                        "predicted_filtered": result.get("predicted_filtered"),
                        "confidence": result.get("confidence"),
                        "probability_filtered": result.get("probability_filtered"),
                        "model": result.get("model", model),
                        "input_tokens": result.get("input_tokens"),
                        "output_tokens": result.get("output_tokens"),
                        "elapsed_ms": result.get("elapsed_ms"),
                        "error": error,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    rows.append(record)
                    done.add(signature)
                    spend_estimate += (result.get("input_tokens") or 0) * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
                    calls_since_checkpoint += 1
                    if delay:
                        time.sleep(delay)
                    if calls_since_checkpoint >= CHECKPOINT_EVERY_CALLS:
                        _write_checkpoint(rows, old_version, new_version, change_reason,
                                          stage, spend_estimate, output_path)
                        calls_since_checkpoint = 0
        summary_path = _write_checkpoint(rows, old_version, new_version, change_reason,
                                         stage, spend_estimate, output_path)
    finally:
        store.close()
    print(summary_path.read_text())
    return summary_path


def run_single(db_path, ids_path, labels_path, prompt_path, prompt_version,
               run_id, model="jev-latest", delay=0.25):
    """Run one frozen Jev prompt on a development or locked final manifest."""
    if not _load_local_typesafe_key():
        raise RuntimeError("Set TYPESAFE_API_KEY in the local ignored .env or process environment; no Jev requests were sent.")
    ids = [int(v) for v in pd.read_csv(ids_path)["review_id"].tolist()]
    if Path(ids_path).name != "development_ids.csv":
        raise ValueError("Single-prompt model selection may run only on development_ids.csv.")
    labels = pd.read_csv(labels_path).set_index("review_id")["true_filtered"].astype(bool).to_dict()
    if set(ids) != set(labels):
        raise ValueError("Review IDs and scorer labels must match exactly.")
    instructions = Path(prompt_path).read_text()
    prompt_hash = hashlib.sha256(instructions.encode()).hexdigest()
    run_dir = OUTPUT_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{run_id}.jsonl"
    previous = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()] \
        if output_path.exists() else []
    if any(r.get("prompt_version") == prompt_version and r.get("prompt_sha256") != prompt_hash
           for r in previous):
        raise ValueError("A prompt changed under an existing version/run ID; create a new prompt version and run ID.")
    done = {r["review_id"] for r in previous
            if r.get("prompt_version") == prompt_version and r.get("prompt_sha256") == prompt_hash
            and r.get("predicted_filtered") is not None}
    rows = list(previous)
    store = YelpZipEvidence(db_path)
    calls_since_checkpoint = 0
    spend_estimate = _project_estimated_spend()
    try:
        with output_path.open("a", encoding="utf-8") as out:
            for review_id in ids:
                if review_id in done:
                    continue
                _enforce_spend_ceiling(spend_estimate)
                review = store.get_review(review_id)
                evidence = collect_evidence(store, review)
                try:
                    result = classify_review(review, evidence, model=model, instructions=instructions)
                    error = None
                except Exception as exc:
                    result, error = {}, f"{type(exc).__name__}: {exc}"
                record = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "review_id": review_id, "prompt_version": prompt_version,
                    "prompt_sha256": prompt_hash, "true_filtered": labels[review_id],
                    "predicted_filtered": result.get("predicted_filtered"),
                    "confidence": result.get("confidence"),
                    "probability_filtered": result.get("probability_filtered"),
                    "model": result.get("model", model), "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"), "elapsed_ms": result.get("elapsed_ms"),
                    "error": error,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(record)
                done.add(review_id)
                spend_estimate += (result.get("input_tokens") or 0) * PUBLIC_INPUT_PRICE_PER_MILLION / 1_000_000
                calls_since_checkpoint += 1
                if delay:
                    time.sleep(delay)
                if calls_since_checkpoint >= CHECKPOINT_EVERY_CALLS:
                    _write_single_checkpoint(rows, prompt_version, output_path, len(ids), spend_estimate)
                    calls_since_checkpoint = 0
        summary_path = _write_single_checkpoint(rows, prompt_version, output_path, len(ids), spend_estimate)
    finally:
        store.close()
    print(summary_path.read_text())
    return summary_path


def _write_checkpoint(rows, old_version, new_version, change_reason, stage, spend_estimate, output_path):
    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_artifact": str(output_path),
        "old_prompt": {"version": old_version, "metrics": _metrics(rows, old_version)},
        "candidate_prompt": {"version": new_version, "metrics": _metrics(rows, new_version)},
        "change_reason": change_reason,
        "stage": stage,
        "project_estimated_spend_usd": round(spend_estimate, 4),
        "internal_spend_ceiling_usd": PROJECT_ESTIMATED_SPEND_CEILING_USD,
        "paired_reviews": len({r["review_id"] for r in rows
                                if r["prompt_version"] in (old_version, new_version)}),
        "target_recall": 0.50,
        "target_accuracy": 0.84,
        "note": {
            "tuning": "Prompt tuning data only; this partition is not the development-selection or final holdout.",
            "development": "Development selection only; do not report as final performance.",
            "final": "Locked final comparison; do not tune prompts or thresholds from these results.",
        }[stage],
    }
    path = output_path.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2))
    return path


def _write_single_checkpoint(rows, version, output_path, planned_n, spend_estimate):
    metrics = _metrics(rows, version)
    summary = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_artifact": str(output_path), "prompt_version": version,
        "metrics": metrics, "planned_n": planned_n,
        "project_estimated_spend_usd": round(spend_estimate, 4),
        "internal_spend_ceiling_usd": PROJECT_ESTIMATED_SPEND_CEILING_USD,
        "target_recall": 0.50, "target_accuracy": 0.84,
        "note": "Development results may guide tuning. Final-holdout results must not guide changes.",
    }
    path = output_path.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--ids", default=str(OUTPUT_DIR / "tuning_ids.csv"))
    parser.add_argument("--labels", default=str(OUTPUT_DIR / "tuning_labels.csv"))
    parser.add_argument("--old-prompt", help="Optional prior prompt text file; omit for the built-in baseline")
    parser.add_argument("--new-prompt", required=True, help="Candidate prompt instructions file")
    parser.add_argument("--old-version", default="jev-v1")
    parser.add_argument("--new-version", required=True)
    parser.add_argument("--change-reason", help="Required rationale for a paired prompt change")
    parser.add_argument("--stage", choices=("tuning", "development", "final"), default="tuning")
    parser.add_argument("--run-id", required=True, help="Stable ID for resumable outputs, e.g. tune-v1-v2")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--single", action="store_true",
                        help="Run one prompt on the selected split instead of a 100-call paired tuning cycle")
    parser.add_argument("--prompt-version", help="Version for --single mode; defaults to --new-version")
    args = parser.parse_args()
    if args.single:
        run_single(args.db, args.ids, args.labels, args.new_prompt,
                   args.prompt_version or args.new_version, args.run_id, args.model, args.delay)
    else:
        if not args.change_reason:
            parser.error("--change-reason is required for every paired prompt iteration")
        run_pair(args.db, args.ids, args.labels, args.old_prompt, args.new_prompt,
                 args.old_version, args.new_version, args.run_id, args.change_reason,
                 args.stage, args.model, args.delay)


if __name__ == "__main__":
    main()
