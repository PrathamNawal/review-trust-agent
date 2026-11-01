"""Time-split evaluation of the logistic-regression baseline on the YelpZip evaluation sample.

Train on reviews dated up to TRAIN_END, choose the regularisation and the set-aside threshold
on VALID_END, then score the later test period once. The cutoffs are fixed here, before any
result was seen. Writes a markdown report; makes no model API calls and changes no app state.

    python3 -m scripts.run_baseline_experiment
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from dashboard.baseline_model import (
    FEATURE_FAMILIES, FEATURES, build_pipeline, coefficient_table, extract_features,
    pick_set_aside_threshold, share_needed_for_recall, time_split, top_share_captured,
)
from dashboard.live_agent import get_toolbox

ROOT = Path(__file__).resolve().parent.parent
TRAIN_END = "2012-12-31"
VALID_END = "2013-12-31"
C_GRID = (0.01, 0.1, 1.0, 10.0)
MAX_SET_ASIDE_FILTERED_RATE = 0.025  # tolerated share of filtered reviews inside the set-aside lane
BOOTSTRAPS = 1000
SEED = 7


def load_features(sample, cache_path):
    if cache_path.exists():
        cached = pd.read_csv(cache_path).set_index("review_id")
        if set(sample["review_id"]) <= set(cached.index) and list(FEATURES) == [c for c in cached.columns]:
            return cached.loc[sample["review_id"]].reset_index()
    toolbox = get_toolbox()
    rows = []
    for position, row in enumerate(sample.itertuples(index=False), 1):
        review = {"review_id": int(row.review_id), "user_id": str(int(row.user_id)),
                  "prod_id": str(int(row.prod_id)), "date": str(row.date)[:10],
                  "review_text": row.review_text, "rating": float(row.rating)}
        rows.append({"review_id": int(row.review_id), **extract_features(toolbox, review)})
        if position % 500 == 0:
            print(f"  features {position}/{len(sample)}", flush=True)
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def bootstrap_auc_ci(labels, scores, rng):
    labels, scores = np.asarray(labels), np.asarray(scores)
    values = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(0, len(labels), len(labels))
        if labels[pick].min() != labels[pick].max():
            values.append(roc_auc_score(labels[pick], scores[pick]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def expected_calibration_error(labels, probs, bins=5):
    labels, probs = np.asarray(labels), np.asarray(probs)
    edges = np.quantile(probs, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    rows, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probs >= low) & (probs < high)
        if mask.any():
            rows.append((float(probs[mask].mean()), float(labels[mask].mean()), int(mask.sum())))
            ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece), rows


def fit_and_score(columns, train_x, train_y, valid_x, valid_y, test_x):
    best = None
    for c in C_GRID:
        model = build_pipeline(c).fit(train_x[columns], train_y)
        valid_auc = roc_auc_score(valid_y, model.predict_proba(valid_x[columns])[:, 1])
        if best is None or valid_auc > best[0]:
            best = (valid_auc, c, model)
    valid_auc, c, model = best
    return {"c": c, "valid_auc": valid_auc, "model": model, "columns": list(columns),
            "valid_scores": model.predict_proba(valid_x[columns])[:, 1],
            "test_scores": model.predict_proba(test_x[columns])[:, 1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=str(ROOT / "data" / "sample.csv"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "yelpzip_eval" / "baseline_features.csv"))
    parser.add_argument("--report", default=str(ROOT / "experiments" / "baseline-model-report.md"))
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sample["date"] = pd.to_datetime(sample["date"])
    sample = sample.sort_values("date", kind="stable").reset_index(drop=True)
    features = load_features(sample, Path(args.cache)).set_index("review_id").loc[sample["review_id"]].reset_index(drop=True)
    labels = sample["filtered"].astype(int).to_numpy()
    train, valid, test = time_split(sample["date"], TRAIN_END, VALID_END)

    split_rows = [(name, int(mask.sum()), int(labels[mask].sum()), float(labels[mask].mean()),
                   str(sample["date"][mask].min().date()), str(sample["date"][mask].max().date()))
                  for name, mask in (("train", train), ("validation", valid), ("test", test))]
    parts = {name: (features[mask].reset_index(drop=True), labels[mask])
             for name, mask in (("train", train), ("valid", valid), ("test", test))}
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = parts["train"], parts["valid"], parts["test"]

    full = fit_and_score(list(FEATURES), train_x, train_y, valid_x, valid_y, test_x)
    scores, rng = full["test_scores"], np.random.default_rng(SEED)
    auc = roc_auc_score(test_y, scores)
    ci = bootstrap_auc_ci(test_y, scores, rng)
    base = float(test_y.mean())

    capture = top_share_captured(scores, test_y, (0.1, 0.2, 0.3, 0.5))
    need = {r: share_needed_for_recall(scores, test_y, r) for r in (0.5, 0.8, 0.9)}
    threshold = pick_set_aside_threshold(full["valid_scores"], valid_y, MAX_SET_ASIDE_FILTERED_RATE)
    lane = {}
    if threshold is not None:
        in_valid = full["valid_scores"] <= threshold
        in_test = scores <= threshold
        lane = {"threshold": threshold, "valid_share": float(in_valid.mean()),
                "valid_rate": float(valid_y[in_valid].mean()), "test_share": float(in_test.mean()),
                "test_rate": float(test_y[in_test].mean()) if in_test.any() else float("nan"),
                "test_missed": float(test_y[in_test].sum() / test_y.sum()), "test_missed_n": int(test_y[in_test].sum())}
    ece, calibration = expected_calibration_error(test_y, scores)

    ablations = {}
    for name, drop in (("no reviewer-activity features", FEATURE_FAMILIES["reviewer activity"]),):
        columns = [f for f in FEATURES if f not in drop]
        ablations[name] = fit_and_score(columns, train_x, train_y, valid_x, valid_y, test_x)
    for family, members in FEATURE_FAMILIES.items():
        ablations[f"{family} only"] = fit_and_score(list(members), train_x, train_y, valid_x, valid_y, test_x)
    single = roc_auc_score(test_y, test_x["reviewer_prior_reviews_log"].fillna(test_x["reviewer_prior_reviews_log"].median()))
    single = max(single, 1 - single)

    test_dates = sample["date"][test].reset_index(drop=True)
    halves = []
    middle = test_dates.sort_values().iloc[len(test_dates) // 2]
    for name, mask in (("first half of test", (test_dates < middle).to_numpy()), ("second half of test", (test_dates >= middle).to_numpy())):
        halves.append((name, int(mask.sum()), float(roc_auc_score(test_y[mask], scores[mask]))))

    table = coefficient_table(full["model"], full["columns"])
    brier = brier_score_loss(test_y, scores)
    brier_base = brier_score_loss(test_y, np.full(len(test_y), train_y.mean()))

    lines = ["# Logistic-regression baseline — time-split evaluation", "",
             "Offline experiment on the 5,000-review YelpZip evaluation sample. Nothing here is wired into the "
             "live dashboard. The target is YelpZip's `filtered` label (Yelp's own filtering outcome), not verified fraud.", "",
             "## Design", "",
             f"- Features: the evidence the app already builds from **earlier reviews only** ({len(FEATURES)} features), "
             "median-imputed with missing-value flags, standardised.",
             f"- Split by review date, cutoffs fixed before any result was seen: train ≤ {TRAIN_END}, validation ≤ {VALID_END}, test after.",
             f"- Regularisation (C ∈ {C_GRID}) and the set-aside threshold were chosen on validation only; the test period was scored once.",
             f"- Set-aside rule: the largest score whose lower tail held ≤ {MAX_SET_ASIDE_FILTERED_RATE:.1%} filtered reviews on validation.", "",
             "| Split | Reviews | Filtered | Rate | Dates |", "|---|---:|---:|---:|---|"]
    lines += [f"| {n} | {r} | {f} | {rate:.1%} | {a} → {b} |" for n, r, f, rate, a, b in split_rows]
    lines += ["", "## Test result", "",
              f"- **AUROC {auc:.3f}** (95% bootstrap CI {ci[0]:.3f}–{ci[1]:.3f}); chance is 0.5. Selected C = {full['c']} (validation AUROC {full['valid_auc']:.3f}).",
              f"- Average precision {average_precision_score(test_y, scores):.3f} against a base rate of {base:.3f}.",
              f"- Single feature, reviewer's earlier-review count: AUROC {single:.3f}.",
              "- Stability: " + "; ".join(f"{n} (n={k}) AUROC {a:.3f}" for n, k, a in halves) + ".", "",
              "### Reviewing the highest-scored reviews first", "",
              "| Review the top | Filtered cases found |", "|---|---:|"]
    lines += [f"| {int(s * 100)}% | {v:.0%} |" for s, v in capture.items()]
    lines += ["", "| To find this share of filtered cases | Share of reviews to review |", "|---|---:|"]
    lines += [f"| {int(r * 100)}% | {v:.0%} |" for r, v in need.items()]
    lines += ["", "### Set-aside lane (lowest-risk reviews, validation-chosen threshold)", ""]
    if lane:
        lines += [f"- Threshold {lane['threshold']:.4f}. On validation it set aside {lane['valid_share']:.0%} of reviews with {lane['valid_rate']:.1%} filtered.",
                  f"- **On test it sets aside {lane['test_share']:.0%} of reviews; {lane['test_rate']:.1%} of them are filtered, missing {lane['test_missed_n']} cases = {lane['test_missed']:.0%} of all filtered.**"]
    else:
        lines += ["- No threshold met the limit on validation."]
    lines += ["", "### Calibration (test)", "",
              f"Brier score {brier:.4f} (predicting the training base rate: {brier_base:.4f}); expected calibration error {ece:.3f}.", "",
              "| Mean predicted | Actual filtered rate | Reviews |", "|---:|---:|---:|"]
    lines += [f"| {p:.3f} | {a:.3f} | {n} |" for p, a, n in calibration]
    lines += ["", "## Where the signal comes from (test AUROC)", "", "| Feature set | Test AUROC |", "|---|---:|", f"| all features | {auc:.3f} |"]
    for name, result in ablations.items():
        lines.append(f"| {name} | {roc_auc_score(test_y, result['test_scores']):.3f} |")
    lines += ["", "## Coefficients (standardised, largest first)", "", "| Feature | Coefficient |", "|---|---:|"]
    lines += [f"| {r.feature} | {r.coefficient:+.3f} |" for r in table.head(12).itertuples()]
    lines += ["", "## Limits", "",
              "- The label is Yelp's filter output. Reviewer-activity features probably mirror how that filter works, so this model may "
              "learn to predict Yelp's filter, not fraud.",
              "- One sample, one split. The live app uses a trimmed history index; counts there can differ from the full dataset.",
              "- The test period was used once, but this sample was part of earlier development work, so treat the result as a baseline to beat on "
              "a fresh, untouched holdout, not as a deployment decision.",
              "- Probabilities are only as calibrated as the table above shows; do not use them as fraud probabilities."]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
