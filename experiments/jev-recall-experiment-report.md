# Jev recall experiment — execution report

**Updated:** 2026-10-18

**Final evaluation:** Complete

**Decision:** At threshold 0.98, Jev improved recall over v1 by 3.0 percentage points on the 3,000-review holdout, but did not meet the target of ≥20% recall and ≥85% accuracy.

## Final result

| System | TP / FP / FN / TN | Recall (95% CI) | Accuracy (95% CI) | Precision | Queue rate |
|---|---:|---:|---:|---:|---:|
| Jev v1 | 19 / 97 / 378 / 2,506 | 4.8% (3.1–7.4%) | 84.2% (82.8–85.4%) | 16.4% | 3.9% |
| Decomposed Jev, threshold 0.98 | 31 / 137 / 366 / 2,466 | 7.8% (5.6–10.9%) | 83.2% (81.9–84.5%) | 18.5% | 5.6% |

- Dataset: 3,000 final YelpZip reviews; 397 filtered labels.
- Comparison: paired on identical reviews and evidence.
- Candidate rule: `1 - P(business_burst) >= 0.98`.
- Model: `jev-1.13.0`; 6,000/6,000 decisions completed; zero errors.
- Change vs v1: 12 more positives found, 12 fewer missed, 40 more false positives, accuracy down 0.9 points, queue rate up 1.7 points.
- No threshold or prompt tuning was performed on final labels.

## Development results

Ten paired batches covered the 500-review development set (66 filtered labels). Thresholds were chosen using earlier batches only: 0.97 for batches 0–2, then 0.98 for batches 3–9.

| Policy | TP / FP / FN / TN | Recall | Accuracy | Queue rate |
|---|---:|---:|---:|---:|
| Jev v1 | 6 / 11 / 60 / 423 | 9.1% | 85.8% | 3.4% |
| Decomposed Jev, sequential thresholds | 11 / 29 / 55 / 405 | 16.7% | 83.2% | 8.0% |

- Candidate recall was 7.6 points higher; accuracy missed the 85% floor.
- Best single development batch: 28.6% recall and 88% accuracy at threshold 0.98 (7 filtered reviews). This result did not represent the full development set.
- No tested decomposed score met both ≥20% recall and ≥85% accuracy across development.

## Choice vs. Score primitive comparison

| Primitive | TP / FP / FN / TN | Recall (95% CI) | Accuracy (95% CI) | Precision | Queue rate |
|---|---:|---:|---:|---:|---:|
| Choice | 5 / 10 / 35 / 250 | 12.5% (5.5–26.1%) | 85.0% (80.5–88.6%) | 33.3% | 5.0% |
| Score, cutoff 0.50 | 5 / 20 / 35 / 240 | 12.5% (5.5–26.1%) | 81.7% (76.9–85.6%) | 20.0% | 8.3% |

- Dataset: first 300 reviews in the frozen development partition; 40 YelpZip filtered labels.
- Paired design: both primitives received the same review and evidence in one Jev request per case; 300/300 completed without errors.
- Score cutoff was fixed at 0.50 before the run. No threshold tuning was done.
- Choice met the 85% accuracy floor on this subset; neither primitive met the ≥20% recall goal. Score found no additional positives and generated 10 more false positives.
- Interpretation: Choice is the better current development candidate. This subset is exploratory development evidence, not final performance; the confidence intervals are wide and the 3,000-review holdout remains untouched by this comparison.
- Estimated cumulative project spend after this run: **$0.3658**, below the $5 cap. Raw attempts and per-review outputs are in the Git-ignored `data/yelpzip_eval/runs/dev-choice-score-300-v1.jsonl`; summary is in its matching `.summary.json`.
- A fresh Choice answer collected in the paired Noul run below differed on one negative case (5 TP / 9 FP rather than 5 TP / 10 FP), showing small run-to-run variation. Use the paired Choice results below for the Choice-vs-Noul comparison.

## Choice vs. Noul primitive comparison

| Method | TP / FP / FN / TN | Recall (95% CI) | Accuracy (95% CI) | Precision | Queue rate |
|---|---:|---:|---:|---:|---:|
| Choice | 5 / 9 / 35 / 251 | 12.5% (5.5–26.1%) | 85.3% (80.9–88.9%) | 35.7% | 4.7% |
| Noul overall risk, cutoff 0.50 | 2 / 5 / 38 / 255 | 5.0% (1.4–16.5%) | 85.7% (81.2–89.2%) | 28.6% | 2.3% |
| Noul business non-burst policy, cutoff 0.98 | 2 / 13 / 38 / 247 | 5.0% (1.4–16.5%) | 83.0% (78.3–86.8%) | 13.3% | 5.0% |

- Dataset: the same first 300 cases in the frozen development partition; 40 YelpZip filtered labels.
- Paired design: Choice, overall-risk Noul, and business-burst Noul were queried together in one request per review; 300/300 paired requests completed.
- Choice found three more filtered cases than direct overall-risk Noul, with four additional false positives and 0.3 percentage points lower accuracy. Recall intervals are wide and overlap; this is a directional development result, not proof of a reliable generalization gain.
- Direct overall-risk Noul at 0.50 had the best accuracy of the three, while Choice had the highest recall. The 0.98 business non-burst rule did not reproduce the final-holdout performance on this development subset.
- Estimated cumulative project spend after this run: **$0.3802**, below the $5 cap. Outputs are in Git-ignored `data/yelpzip_eval/runs/dev-choice-noul-300-v1.jsonl` and its matching `.summary.json`.
- These reused development cases do not change the locked 3,000-review final result; do not select a deployment policy from this subset alone.

## Fresh holdout comparison — interim, 961 paired reviews

| Method | TP / FP / FN / TN | Recall (95% CI) | Accuracy (95% CI) | Precision | Queue rate |
|---|---:|---:|---:|---:|---:|
| Choice | 11 / 44 / 108 / 798 | 9.2% (5.2–15.8%) | 84.2% (81.7–86.4%) | 20.0% | 5.7% |
| Noul overall risk, cutoff 0.50 | 5 / 23 / 114 / 819 | 4.2% (1.8–9.5%) | **85.7% (83.4–87.8%)** | 17.9% | 2.9% |
| Noul business non-burst policy, cutoff 0.98 | 10 / 49 / 109 / 793 | 8.4% (4.6–14.8%) | 83.6% (81.1–85.8%) | 16.9% | 6.1% |

- Dataset: 961 completed cases from a newly frozen, stratified 3,000-review YelpZip holdout; 119 filtered labels among completed cases. IDs were excluded against earlier examples, manifests, and run artifacts. The full fresh set contained 397 filtered and 2,603 recommended labels.
- Paired design: Jev Choice, overall-risk Noul, and business-burst Noul were requested together with the same review and evidence. All 961 completed cases have results for all three methods.
- Choice found six more filtered cases than direct overall-risk Noul, at 1.6 percentage points lower accuracy. The 0.98 business non-burst rule had the highest recall among these methods but lower accuracy than Choice.
- Recall confidence intervals are wide and overlap. This interim result does not establish a statistically reliable winner.
- Recommendation under the current ≥85% accuracy floor: direct overall-risk Noul at 0.50 is the only method above the floor in this interim sample, but its recall is only 4.2%. Choice is the stronger recall candidate (9.2%) but falls below the accuracy floor at 84.2%. The 0.98 business policy reaches 8.4% recall at 83.6% accuracy. **No method meets the joint goal of ≥20% recall and ≥85% accuracy; do not promote a new method based on this partial run.**
- The user stopped the evaluation before the planned 3,000 cases: 961 paired reviews completed, 2,039 were not evaluated. The earlier 400-case checkpoint is superseded by these 961-case interim metrics. Treat this partial holdout as consumed for method or threshold tuning.
- Estimated cumulative project spend: **$0.427**, within the $5 cap. Partial run records and summary are in the Git-ignored `data/yelpzip_eval/runs/final-choice-noul-fresh-3000-v1.jsonl` and matching `.summary.json`; the original frozen manifest and hash are in `data/yelpzip_eval/fresh_choice_noul_3000_manifest.json`.

## Prompt-only tuning

Four paired iterations used the 50-review tuning set (7 filtered labels).

| Version | Change | Recall | Accuracy | Result |
|---|---|---:|---:|---|
| v2 | Require behavior evidence | 0/7 | 80% | No positives found |
| v3 | Favor recall; accept moderate evidence | 0/7 | 76% | No positives found; more false positives |
| v4 | Flag one meaningful signal | 3/7 | 60% | Highest tuning recall; too many false positives |
| v5 | Require corroborating signals | 0/7 | 82% | Conservative; no positives found |

Prompt-only tuning did not produce a candidate that met the project targets.

## Earlier exploratory comparison

- Sample: 50 manual-batch reviews; 6 Yelp-filtered labels.
- Jev recall: 3/6 (50%); accuracy: 82%; false positives: 6.
- Stored manual Claude baseline: 0/6 recall; accuracy: 84%; false positives: 2.
- This result is exploratory: the sample is small, overlaps prior project data, and is not a controlled comparison with the current Agno system.

## Cost and artifacts

- Estimated final-run cost: **$0.28** at published TypeSafe input-token pricing.
- Estimated cumulative project usage: **$0.352**, below the $5 budget. Actual account charges may differ.
- Per-review final outputs and summary: `data/yelpzip_eval/runs/final-v1-decomposed-threshold-098.jsonl` and `.summary.json` (Git-ignored).
- Final runner: `scripts/yelpzip_jev_final.py`.
- Full YelpZip index and split manifests are local under Git-ignored `data/yelpzip_eval/`.

## Conclusion and next step

- The candidate improved recall on the final set, but recall remains low and accuracy fell below 85%.
- YelpZip's `filtered` label is a benchmark reference, not independently verified fraud ground truth.
- Keep Jev experimental; do not use it for automatic removal.
- The 3,000-review holdout is consumed. Any revised system needs a new untouched final evaluation set.
- Recommended direction: test a supervised model using YelpZip activity and graph features; use Jev for structured evidence extraction if it adds value.

## Streamlit integration note

The Streamlit Jev adapter now uses direct Choice as its default experimental method. It
uses the bundled trimmed evidence index rather than the full YelpZip cache used for these
offline comparisons, so neither the historical 7.8%/83.2% decomposed result nor the partial
9.2%/84.2% Choice result predicts live dashboard performance. Choice probabilities are not
calibrated for Yelp fraud. Jev decisions always require human review; Agno remains available.
