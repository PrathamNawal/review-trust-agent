# Evaluation scorecard — Review Trust Agent / Jev

> **Status:** Jev prompt/workflow comparison completed on the locked 3,000-review
> YelpZip holdout. Neither version met the project target. The holdout is consumed.
> **Label caveat:** Yelp `filtered` labels are benchmark references, not verified fraud ground truth.

The dashboard's current Jev method is direct **Choice**. A separate fresh paired evaluation
was stopped at 961 of 3,000 planned reviews: Choice reached 9.2% recall and 84.2% accuracy;
direct overall-risk Noul reached 4.2% and 85.7%. These are interim results, below the joint
target, and do not establish a statistically reliable winner. Full metrics and limits are in
the [execution report](../experiments/jev-recall-experiment-report.md).

## Acceptance targets

| Measure | Target | Final result |
|---|---:|---|
| Filtered-class recall | ≥20% | Jev v1: 4.8% (19/397); decomposed: 7.8% (31/397) |
| Accuracy | ≥85% | Jev v1: 84.2%; decomposed: 83.2% |
| Completion | Report all attempts; no silent exclusions | 6,000/6,000 paired decisions; zero errors |
| Cost | Within the user's $5 budget | Estimated project total: $0.352; actual account deductions may differ |

Targets reflect the user's requested operating point. Also report precision, false-positive
rate, queue rate, F1, balanced accuracy, confidence intervals, latency, token use, and errors.

## Locked final comparison

| System | TP / FP / FN / TN | Recall (95% Wilson CI) | Accuracy (95% Wilson CI) | Precision | Queue rate | FPR |
|---|---|---:|---:|---:|---:|---:|
| Jev v1 | 19 / 97 / 378 / 2,506 | 4.8% (3.1–7.4%) | 84.2% (82.8–85.4%) | 16.4% | 3.9% | 3.7% |
| Decomposed Jev, threshold 0.98 | 31 / 137 / 366 / 2,466 | 7.8% (5.6–10.9%) | 83.2% (81.9–84.5%) | 18.5% | 5.6% | 5.3% |

- Candidate found 12 more positives and produced 40 more false positives than v1.
- Recall improved by 3.0 percentage points; accuracy declined by 0.9 points.
- Both arms used identical reviews and evidence; this isolates Jev workflow changes, not gains over Agno.
- Final set: 3,000 reviews, including 397 filtered labels. No prompt or threshold tuning used final labels.
- The holdout has been used. Do not tune against it; use a new untouched holdout for the next final evaluation.
- Full report: [`experiments/jev-recall-experiment-report.md`](../experiments/jev-recall-experiment-report.md).

## Development and tuning summary

| Stage | Result | Readout |
|---|---|---|
| Prompt tuning, 50 reviews / 7 filtered | v4 peaked at 3/7 recall (42.9%), 60% accuracy; v5 returned to 0/7 recall, 82% accuracy | Small tuning set; no prompt met both targets |
| Development, 500 reviews / 66 filtered | Decomposed sequential thresholds: 11/66 recall (16.7%), 83.2% accuracy; v1: 6/66 (9.1%), 85.8% | Recall rose, accuracy missed target |
| Final, 3,000 reviews / 397 filtered | Decomposed threshold 0.98: 7.8% recall, 83.2% accuracy | Neither target achieved; modest recall gain replicated at lower accuracy |

Full per-version metrics and threshold methodology are in the execution report. Earlier
50-review exploratory comparison (Jev 3/6 recall, 82% accuracy versus recorded manual
Claude 0/6, 84%) is preliminary and not a controlled Agno baseline.

## Current decision and next evaluation

- Keep Jev experimental; do not use it for automatic removal.
- Jev Choice is integrated as the Streamlit default experimental architecture; its
  uncalibrated decisions always route to a human. The dashboard uses a trimmed evidence
  index, so benchmark results do not predict its live metrics. Agno remains selectable.
- The 3,000-review final holdout is consumed. Do not change a prompt and re-score it as
  confirmatory evidence.
- Investigate supervised YelpZip graph/activity features; consider Jev for structured
  evidence extraction if it adds value.
- Freeze any new design before evaluation on fresh untouched IDs. Compare with Agno on
  identical examples if the question is whether Jev improves the active system.
- Keep labels and IDs out of model requests; use only evidence available before each
  review date. Include uncertainty and reviewer/business clustering in future analysis.
- Recheck token use and actual TypeSafe account charges; reported $0.352 project spend
  is a published-price estimate.

## Reproducibility

- Run code: `scripts/yelpzip_jev_final.py`, `scripts/yelpzip_jev_eval.py`.
- Final prompt/workflow: decomposed Jev with `1 - P(business_burst) >= 0.98`.
- Git-ignored per-review results: `data/yelpzip_eval/runs/final-v1-decomposed-threshold-098.jsonl`
  and matching `.summary.json`.
- Full YelpZip cache and split manifests remain local under Git-ignored `data/yelpzip_eval/`.
