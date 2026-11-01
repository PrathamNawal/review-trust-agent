"""Does Jev Choice add anything on top of the logistic-regression baseline?

Phase 1 (needs your own key): ask Jev Choice about every validation and test review, exactly as the
live app does (same pinned model, default instructions, all four earlier-review evidence sources).
Results are cached line by line, so an interrupted run resumes and nothing is paid for twice.
Phase 2 (no key needed): on the same reviews, compare Jev alone, the logistic model alone, and the
two stacked, and write a report.

    export TYPESAFE_API_KEY=...          # your own key; this script only reads it, never prints or stores it
    python3 -m scripts.run_jev_stacking_experiment --dry-run      # count calls, no spend
    python3 -m scripts.run_jev_stacking_experiment                # phase 1 then phase 2
    python3 -m scripts.run_jev_stacking_experiment --analyse-only # phase 2 from the cache
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dashboard.baseline_model import (
    EXTRA_FAMILIES, FEATURES, choose_cutoff_by_prevalence, time_split,
)
from dashboard.config_store import DEFAULT_JEV_EVIDENCE_SOURCES, DEFAULT_JEV_PROMPTS
from dashboard.evidence import build_time_aware_evidence
from dashboard.jev_agent import MODEL_ID
from dashboard.jev_model import JevAPIError, classify_review
from dashboard.live_agent import get_toolbox
from scripts.run_baseline_experiment import TRAIN_END, VALID_END, load_features
from scripts.run_baseline_features_experiment import fit_variant, load_extras, text_model_scores

ROOT = Path(__file__).resolve().parent.parent
PREVALENCE = 0.12
ACCURACY_FLOORS = (0.84, 0.85)
LOW_COST_PER_CALL_USD = 0.28 / 6000  # from the Jev experiment report's own estimate; a rough guide only
SEED = 7


def pending_review_ids(review_ids, cached_ids):
    """Reviews still to ask Jev about, in order, skipping any already cached."""
    done = set(cached_ids)
    return [rid for rid in review_ids if rid not in done]


def read_cache(path):
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["review_id"])] = row
    return rows


def logit(p, eps=1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def with_retries(call, attempts=5, first_wait=2.0, sleep=time.sleep):
    """Run `call`, retrying a failed request with growing waits (2, 4, 8, 16 s). One dropped connection must not
    end a run of thousands of calls. A retried request can occasionally be billed twice if the first one did get through."""
    for attempt in range(attempts):
        try:
            return call()
        except JevAPIError:
            if attempt == attempts - 1:
                raise
            sleep(first_wait * 2 ** attempt)


def ask_jev(sample_rows, cache_path, max_calls, pause):
    toolbox = get_toolbox()
    prompts = DEFAULT_JEV_PROMPTS
    made = 0
    with cache_path.open("a") as out:
        for row in sample_rows:
            if made >= max_calls:
                print(f"Reached --max-calls={max_calls}; stopping. Re-run to continue.")
                break
            review = {"review_id": int(row.review_id), "user_id": str(int(row.user_id)),
                      "prod_id": str(int(row.prod_id)), "date": str(row.date)[:10],
                      "review_text": row.review_text, "rating": float(row.rating)}
            evidence = build_time_aware_evidence(toolbox, review, DEFAULT_JEV_EVIDENCE_SOURCES)
            try:
                result = with_retries(lambda: classify_review(
                    review, evidence, model=MODEL_ID, instructions=prompts["choice"], method="choice"))
            except JevAPIError as exc:
                print(f"  review {review['review_id']}: {exc} — still failing after retries; stopping so nothing is lost. "
                      "Re-run to continue.")
                break
            out.write(json.dumps({
                "review_id": review["review_id"], "predicted_filtered": result["predicted_filtered"],
                "p_filtered": result["choice_probabilities"].get("filtered"),
                "p_recommended": result["choice_probabilities"].get("recommended"),
            }) + "\n")
            out.flush()
            made += 1
            if made % 100 == 0:
                print(f"  {made} calls made", flush=True)
            time.sleep(pause)
    return made


def paired_auc_difference(labels, scores_a, scores_b, rng, rounds=1000):
    labels, scores_a, scores_b = map(np.asarray, (labels, scores_a, scores_b))
    diffs = []
    for _ in range(rounds):
        pick = rng.integers(0, len(labels), len(labels))
        if labels[pick].min() != labels[pick].max():
            diffs.append(roc_auc_score(labels[pick], scores_a[pick]) - roc_auc_score(labels[pick], scores_b[pick]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="count the calls and estimate cost; call nothing")
    parser.add_argument("--analyse-only", action="store_true", help="skip phase 1; analyse the cache")
    parser.add_argument("--max-calls", type=int, default=100000, help="stop after this many new calls")
    parser.add_argument("--pause", type=float, default=0.05, help="seconds between calls")
    parser.add_argument("--sample", default=str(ROOT / "data" / "sample.csv"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "yelpzip_eval" / "jev_stacking_calls.jsonl"))
    parser.add_argument("--report", default=str(ROOT / "experiments" / "jev-stacking-report.md"))
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sample["date"] = pd.to_datetime(sample["date"])
    sample = sample.sort_values("date", kind="stable").reset_index(drop=True)
    labels = sample["filtered"].astype(int).to_numpy()
    train, valid, test = time_split(sample["date"], TRAIN_END, VALID_END)
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = read_cache(cache_path)

    needed = sample[valid | test]
    todo = pending_review_ids(needed["review_id"].astype(int), cached)
    print(f"Jev Choice calls needed: {len(needed)} (validation {int(valid.sum())} + test {int(test.sum())}); "
          f"already cached: {len(needed) - len(todo)}; still to make: {len(todo)}")
    print(f"Rough cost of the remaining calls: about ${len(todo) * LOW_COST_PER_CALL_USD:.2f} "
          "(scaled from the experiment report's own estimate; check your TypeSafe pricing).")
    if args.dry_run:
        return
    if not args.analyse_only and todo:
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise SystemExit("TYPESAFE_API_KEY is not set. Export your own key in this shell, or use --analyse-only.")
        rows = needed[needed["review_id"].astype(int).isin(todo)].itertuples(index=False)
        ask_jev(rows, cache_path, args.max_calls, args.pause)
        cached = read_cache(cache_path)

    have = [int(r) in cached and cached[int(r)]["p_filtered"] is not None for r in sample["review_id"]]
    if not all(h for h, m in zip(have, (valid | test)) if m):
        raise SystemExit("Not every validation/test review has a Jev answer yet; re-run to finish phase 1.")

    # ---- phase 2: same reviews, paired comparison -------------------------------------------------
    base = load_features(sample, ROOT / "data" / "yelpzip_eval" / "baseline_features.csv").set_index("review_id").loc[sample["review_id"]].reset_index(drop=True)
    extra = load_extras(sample, ROOT / "data" / "yelpzip_eval" / "baseline_extra_features.csv").set_index("review_id").loc[sample["review_id"]].reset_index(drop=True)
    table = pd.concat([base, extra], axis=1)
    oof, (valid_text, test_text) = text_model_scores(
        sample["review_text"][train], labels[train], [sample["review_text"][valid], sample["review_text"][test]])
    table["text_model_score"] = np.nan
    table.loc[train, "text_model_score"] = oof
    table.loc[valid, "text_model_score"] = valid_text
    table.loc[test, "text_model_score"] = test_text
    parts = tuple((table[m].reset_index(drop=True), labels[m]) for m in (train, valid, test))
    columns = list(FEATURES) + [f for members in EXTRA_FAMILIES.values() for f in members] + ["text_model_score"]
    lr = fit_variant(columns, parts)

    valid_y, test_y = parts[1][1], parts[2][1]
    jev_valid = np.array([cached[int(r)]["p_filtered"] for r in sample["review_id"][valid]])
    jev_test = np.array([cached[int(r)]["p_filtered"] for r in sample["review_id"][test]])
    lr_valid, lr_test = lr["valid_scores"], lr["test_scores"]

    # stacking: a tiny model on validation (2-3 numbers) that sees only the two scores
    def stack_inputs(lr_scores, jev_scores):
        return np.column_stack([logit(lr_scores), logit(jev_scores)])
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(stack_inputs(lr_valid, jev_valid), valid_y)
    stack_valid = stack.predict_proba(stack_inputs(lr_valid, jev_valid))[:, 1]
    stack_test = stack.predict_proba(stack_inputs(lr_test, jev_test))[:, 1]
    lr_only = LogisticRegression(C=1.0, max_iter=1000).fit(logit(lr_valid).reshape(-1, 1), valid_y)  # same recalibration, no Jev
    lr_only_valid = lr_only.predict_proba(logit(lr_valid).reshape(-1, 1))[:, 1]
    lr_only_test = lr_only.predict_proba(logit(lr_test).reshape(-1, 1))[:, 1]

    rng = np.random.default_rng(SEED)
    systems = {"Jev Choice alone": (jev_valid, jev_test), "Logistic regression alone": (lr_only_valid, lr_only_test),
               "Logistic regression + Jev (stacked)": (stack_valid, stack_test)}
    n_test, positives = len(test_y), int(test_y.sum())
    lines = ["# Does Jev Choice add anything on top of the logistic baseline?", "",
             "Offline experiment on the 5,000-review YelpZip evaluation sample; nothing here is wired into the live dashboard. "
             "The target is YelpZip's `filtered` label (Yelp's own filter outcome), not verified fraud.", "",
             "## Design", "",
             f"- Jev Choice (`{MODEL_ID}`, default instructions, all four earlier-review evidence sources) was asked about every validation "
             f"({int(valid.sum())}) and test ({n_test}) review — the same reviews the logistic model is scored on, so the comparison is paired.",
             "- The logistic model is the improved baseline (train ≤ 2012). The stacked model is a 2-input logistic regression fitted on "
             "**validation only** (Jev's and the logistic model's scores), then scored once on the test period. The 'alone' rows use the same "
             "one-input recalibration without Jev, so the only difference is Jev's score.",
             f"- Cutoffs chosen on validation from rates, with accuracy recomputed at a fixed {PREVALENCE:.0%} filtered share.", "",
             f"## Test result ({n_test:,} reviews, {positives} filtered)", "",
             "| System | Test AUROC |", "|---|---:|"]
    for name, (_, scores) in systems.items():
        lines.append(f"| {name} | {roc_auc_score(test_y, scores):.3f} |")
    delta_ci = paired_auc_difference(test_y, stack_test, lr_only_test, rng)
    delta = roc_auc_score(test_y, stack_test) - roc_auc_score(test_y, lr_only_test)
    lines += ["", f"**Stacked minus logistic alone: {delta:+.3f} AUROC** (95% paired bootstrap CI {delta_ci[0]:+.3f} to {delta_ci[1]:+.3f}). "
              "A range that includes zero means Jev's score did not demonstrably add information.", "",
              f"Jev's stacking weight (on logit scale): {stack.coef_[0][1]:+.2f}; logistic model's: {stack.coef_[0][0]:+.2f}.", "",
              "### Recall at the best cutoff that keeps accuracy at or above the floor", "",
              "| System | Floor | Flagged | Recall | Precision | Accuracy (test) |", "|---|---:|---:|---:|---:|---:|"]
    for name, (valid_scores, test_scores) in systems.items():
        for floor in ACCURACY_FLOORS:
            pick = choose_cutoff_by_prevalence(valid_scores, valid_y, PREVALENCE, floor)
            if pick is None:
                lines.append(f"| {name} | ≥{floor:.0%} | none qualifies | | | |")
                continue
            flagged = test_scores >= pick[0]
            tp = int((flagged & (test_y == 1)).sum()); fp = int((flagged & (test_y == 0)).sum()); tn = n_test - positives - fp
            lines.append(f"| {name} | ≥{floor:.0%} | {(tp + fp) / n_test:.1%} | {tp / positives:.1%} | {tp / max(tp + fp, 1):.1%} | {(tp + tn) / n_test:.1%} |")
    band = lr_test >= np.quantile(lr_test, 0.85)
    lines += ["", "### Inside the logistic model's top 15% (the band a second stage would re-rank)", ""]
    if test_y[band].min() != test_y[band].max():
        lines.append(f"Jev AUROC within that band: **{roc_auc_score(test_y[band], jev_test[band]):.3f}** "
                     f"({int(band.sum())} reviews, {int(test_y[band].sum())} filtered). 0.5 means Jev cannot tell filtered from not inside the band.")
    lines += ["", "## Limits", "",
              "- One sample, one split, and a trimmed history index; the test year had been seen before by earlier baseline work.",
              "- Jev's instructions were the app defaults; other prompts could behave differently.",
              "- The label is Yelp's filter output, not verified fraud.",
              "- The stacking model is fitted on a single validation year whose filtered share differs from the test year's."]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
