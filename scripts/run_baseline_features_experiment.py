"""Better features for the logistic-regression baseline, plus a sweep of cutoff scores.

Feature sets are compared on the validation year only; the test period is scored once for the
chosen set (and for the original 17 features, for reference). Cutoffs are chosen on validation
from rates, so a year's filtered share cannot move them. Writes a markdown report.

    python3 -m scripts.run_baseline_features_experiment
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

from dashboard.baseline_model import (
    EXTRA_FAMILIES, FEATURES, accuracy_at_prevalence, build_pipeline, choose_cutoff_by_prevalence,
    coefficient_table, cutoff_at_false_positive_rate, extract_extra_features, time_split,
)
from dashboard.live_agent import get_toolbox
from scripts.run_baseline_experiment import (
    C_GRID, TRAIN_END, VALID_END, bootstrap_auc_ci, load_features,
)

ROOT = Path(__file__).resolve().parent.parent
PREVALENCE = 0.12          # accuracy is recomputed at this filtered share when choosing a cutoff
ACCURACY_FLOORS = (0.84, 0.85)
FPR_GRID = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)
SEED = 7


def load_extras(sample, cache_path):
    if cache_path.exists():
        cached = pd.read_csv(cache_path).set_index("review_id")
        if set(sample["review_id"]) <= set(cached.index):
            return cached.loc[sample["review_id"]].reset_index()
    toolbox = get_toolbox()
    rows = []
    for position, row in enumerate(sample.itertuples(index=False), 1):
        review = {"review_id": int(row.review_id), "user_id": str(int(row.user_id)),
                  "prod_id": str(int(row.prod_id)), "date": str(row.date)[:10],
                  "review_text": row.review_text, "rating": float(row.rating)}
        rows.append({"review_id": int(row.review_id), **extract_extra_features(toolbox, review)})
        if position % 1000 == 0:
            print(f"  extra features {position}/{len(sample)}", flush=True)
    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def text_model_scores(train_text, train_y, other_texts):
    """Words-only logistic model. Training rows get out-of-fold scores so the next model never sees a
    score the text model produced for a review it was fitted on."""
    def make():
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True),
                             LogisticRegression(C=1.0, max_iter=2000))
    train_text = np.asarray(train_text, dtype=object)
    oof = np.zeros(len(train_text))
    for fit_idx, score_idx in KFold(5, shuffle=True, random_state=SEED).split(train_text):
        oof[score_idx] = make().fit(train_text[fit_idx], train_y[fit_idx]).predict_proba(train_text[score_idx])[:, 1]
    full = make().fit(train_text, train_y)
    return oof, [full.predict_proba(np.asarray(t, dtype=object))[:, 1] for t in other_texts]


def fit_variant(columns, parts):
    (train_x, train_y), (valid_x, valid_y), (test_x, _) = parts
    best = None
    for c in C_GRID:
        model = build_pipeline(c).fit(train_x[columns], train_y)
        auc = roc_auc_score(valid_y, model.predict_proba(valid_x[columns])[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, model)
    auc, c, model = best
    return {"c": c, "valid_auc": auc, "model": model, "columns": list(columns),
            "valid_scores": model.predict_proba(valid_x[columns])[:, 1],
            "test_scores": model.predict_proba(test_x[columns])[:, 1]}


def confusion(flagged, labels):
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum())
    fn = int(labels.sum() - tp)
    return tp, fp, fn, int(len(labels) - tp - fp - fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=str(ROOT / "data" / "sample.csv"))
    parser.add_argument("--cache", default=str(ROOT / "data" / "yelpzip_eval" / "baseline_features.csv"))
    parser.add_argument("--extras-cache", default=str(ROOT / "data" / "yelpzip_eval" / "baseline_extra_features.csv"))
    parser.add_argument("--report", default=str(ROOT / "experiments" / "baseline-features-report.md"))
    args = parser.parse_args()

    sample = pd.read_csv(args.sample)
    sample["date"] = pd.to_datetime(sample["date"])
    sample = sample.sort_values("date", kind="stable").reset_index(drop=True)
    base = load_features(sample, Path(args.cache)).set_index("review_id").loc[sample["review_id"]].reset_index(drop=True)
    extra = load_extras(sample, Path(args.extras_cache)).set_index("review_id").loc[sample["review_id"]].reset_index(drop=True)
    table = pd.concat([base, extra], axis=1)
    labels = sample["filtered"].astype(int).to_numpy()
    train, valid, test = time_split(sample["date"], TRAIN_END, VALID_END)

    oof, (valid_text, test_text) = text_model_scores(
        sample["review_text"][train], labels[train], [sample["review_text"][valid], sample["review_text"][test]])
    table["text_model_score"] = np.nan
    table.loc[train, "text_model_score"] = oof
    table.loc[valid, "text_model_score"] = valid_text
    table.loc[test, "text_model_score"] = test_text

    parts = tuple((table[m].reset_index(drop=True), labels[m]) for m in (train, valid, test))
    test_y = parts[2][1]
    all_extra = [f for members in EXTRA_FAMILIES.values() for f in members]
    variants = {
        "original 17 features": list(FEATURES),
        "+ reviewer extras": list(FEATURES) + list(EXTRA_FAMILIES["reviewer extras"]),
        "+ business extras": list(FEATURES) + list(EXTRA_FAMILIES["business extras"]),
        "+ text style": list(FEATURES) + list(EXTRA_FAMILIES["text style"]),
        "+ all extras": list(FEATURES) + all_extra,
        "+ all extras + words-only text model": list(FEATURES) + all_extra + ["text_model_score"],
    }
    fits = {name: fit_variant(cols, parts) for name, cols in variants.items()}
    chosen_name = max(fits, key=lambda n: fits[n]["valid_auc"])  # validation only
    chosen, original = fits[chosen_name], fits["original 17 features"]

    rng = np.random.default_rng(SEED)
    test_auc = {n: roc_auc_score(test_y, f["test_scores"]) for n, f in fits.items()}
    ci = bootstrap_auc_ci(test_y, chosen["test_scores"], rng)
    n_test, positives = len(test_y), int(test_y.sum())
    test_base = float(test_y.mean())

    def row(label, scores_valid, scores_test, cutoff):
        tp, fp, fn, tn = confusion(scores_test >= cutoff, test_y)
        vt, vf, _, _ = confusion(scores_valid >= cutoff, parts[1][1])
        v_tpr, v_fpr = vt / parts[1][1].sum(), vf / (len(parts[1][1]) - parts[1][1].sum())
        return (f"| {label} | {cutoff:.3f} | {(tp + fp) / n_test:.1%} | {tp / positives:.1%} | {tp / max(tp + fp, 1):.1%} | "
                f"{(tp + tn) / n_test:.1%} | {accuracy_at_prevalence(tp / positives, fp / (n_test - positives), PREVALENCE):.1%} | "
                f"{accuracy_at_prevalence(v_tpr, v_fpr, PREVALENCE):.1%} |")

    header = ["| Cutoff rule | Score cutoff | Flagged | Recall | Precision | Accuracy (test) | Accuracy at 12% filtered (test) | same, on validation |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]

    def sweep(fit):
        lines = list(header)
        for floor in ACCURACY_FLOORS:
            pick = choose_cutoff_by_prevalence(fit["valid_scores"], parts[1][1], PREVALENCE, floor)
            if pick:
                lines.append(row(f"**best recall with accuracy ≥ {floor:.0%} (chosen on validation)**",
                                 fit["valid_scores"], fit["test_scores"], pick[0]))
            else:
                lines.append(f"| best recall with accuracy ≥ {floor:.0%} | none qualifies | | | | | | |")
        for target in FPR_GRID:
            cutoff = cutoff_at_false_positive_rate(fit["valid_scores"], parts[1][1], target)
            lines.append(row(f"flag {target:.0%} of not-filtered reviews (validation)", fit["valid_scores"], fit["test_scores"], cutoff))
        return lines

    coefficients = coefficient_table(chosen["model"], chosen["columns"])
    lines = ["# Better features for the logistic-regression baseline", "",
             "Offline experiment on the 5,000-review YelpZip evaluation sample; nothing here is wired into the live dashboard. "
             "The target is YelpZip's `filtered` label (Yelp's own filter outcome), not verified fraud.", "",
             "## Design", "",
             f"- Same date split as the first baseline: train ≤ {TRAIN_END}, validation ≤ {VALID_END}, test after. "
             "Every feature uses only reviews dated before the one being described.",
             "- Feature sets were compared **on validation AUROC only**; the test period is scored once for the chosen set "
             "(and for every set, shown for information — it did not influence the choice). The original baseline's test result "
             "had already been seen.",
             f"- Cutoffs are chosen on validation from true- and false-positive rates, with accuracy recomputed at a fixed "
             f"{PREVALENCE:.0%} filtered share. The raw filtered share moved from 16.1% (2013) to 12.0% (2014), so raw validation "
             "accuracy is not a stable way to pick a cutoff.", "",
             "## Feature sets", "",
             "| Feature set | Features | Validation AUROC | Test AUROC |", "|---|---:|---:|---:|"]
    for name, fit in fits.items():
        mark = " ← chosen" if name == chosen_name else ""
        lines.append(f"| {name}{mark} | {len(fit['columns'])} | {fit['valid_auc']:.3f} | {test_auc[name]:.3f} |")
    lines += ["", f"Chosen: **{chosen_name}**, C = {chosen['c']}. Test AUROC **{test_auc[chosen_name]:.3f}** "
              f"(95% bootstrap CI {ci[0]:.3f}–{ci[1]:.3f}); average precision {average_precision_score(test_y, chosen['test_scores']):.3f} "
              f"against a base rate of {test_base:.3f}.", "",
              f"## Cutoff scores — chosen model ({n_test:,} test reviews, {positives} filtered)", ""]
    lines += sweep(chosen)
    lines += ["", f"## Same cutoff rules — original 17 features, for comparison", ""] + sweep(original)
    lines += ["", "## Largest coefficients (standardised)", "", "| Feature | Coefficient |", "|---|---:|"]
    lines += [f"| {r.feature} | {r.coefficient:+.3f} |" for r in coefficients.head(12).itertuples()]
    lines += ["", "## Limits", "",
              "- The label is Yelp's filter output; reviewer-activity features probably mirror it, so this may predict Yelp's filter, not fraud.",
              "- One sample and one split; the live app's history index is trimmed. Confirm on a fresh, untouched holdout before any decision.",
              f"- With about {PREVALENCE:.0%} of reviews filtered, flagging nothing already scores {1 - PREVALENCE:.0%} accuracy, and "
              "every flag lowers accuracy unless it is more than 50% likely to be filtered. An accuracy floor therefore caps how many "
              "reviews can be flagged; only higher precision raises the recall available under it."]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
