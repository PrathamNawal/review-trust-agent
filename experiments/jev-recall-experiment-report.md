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

The frozen Jev policy is integrated as the Streamlit default experimental architecture.
It uses the bundled trimmed evidence index rather than the full YelpZip cache used for
the locked evaluation, so 7.8% recall and 83.2% accuracy are historical benchmark results,
not expected live dashboard metrics. Jev outputs are uncalibrated and always require human
review; Agno remains available for rollback.
