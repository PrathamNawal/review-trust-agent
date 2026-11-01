# Logistic-regression baseline — time-split evaluation

Offline experiment on the 5,000-review YelpZip evaluation sample. Nothing here is wired into the live dashboard. The target is YelpZip's `filtered` label (Yelp's own filtering outcome), not verified fraud.

## Design

- Features: the evidence the app already builds from **earlier reviews only** (17 features), median-imputed with missing-value flags, standardised.
- Split by review date, cutoffs fixed before any result was seen: train ≤ 2012-12-31, validation ≤ 2013-12-31, test after.
- Regularisation (C ∈ (0.01, 0.1, 1.0, 10.0)) and the set-aside threshold were chosen on validation only; the test period was scored once.
- Set-aside rule: the largest score whose lower tail held ≤ 2.5% filtered reviews on validation.

| Split | Reviews | Filtered | Rate | Dates |
|---|---:|---:|---:|---|
| train | 2371 | 299 | 12.6% | 2005-06-28 → 2012-12-31 |
| validation | 1140 | 184 | 16.1% | 2013-01-01 → 2013-12-31 |
| test | 1489 | 178 | 12.0% | 2014-01-01 → 2015-01-09 |

## Test result

- **AUROC 0.721** (95% bootstrap CI 0.687–0.753); chance is 0.5. Selected C = 10.0 (validation AUROC 0.764).
- Average precision 0.226 against a base rate of 0.120.
- Single feature, reviewer's earlier-review count: AUROC 0.696.
- Stability: first half of test (n=743) AUROC 0.708; second half of test (n=746) AUROC 0.738.

### Reviewing the highest-scored reviews first

| Review the top | Filtered cases found |
|---|---:|
| 10% | 18% |
| 20% | 36% |
| 30% | 58% |
| 50% | 82% |

| To find this share of filtered cases | Share of reviews to review |
|---|---:|
| 50% | 27% |
| 80% | 49% |
| 90% | 62% |

### Set-aside lane (lowest-risk reviews, validation-chosen threshold)

- Threshold 0.0396. On validation it set aside 40% of reviews with 2.4% filtered.
- **On test it sets aside 44% of reviews; 4.0% of them are filtered, missing 26 cases = 15% of all filtered.**

### Calibration (test)

Brier score 0.1003 (predicting the training base rate: 0.1053); expected calibration error 0.038.

| Mean predicted | Actual filtered rate | Reviews |
|---:|---:|---:|
| 0.001 | 0.013 | 298 |
| 0.014 | 0.057 | 298 |
| 0.068 | 0.108 | 297 |
| 0.145 | 0.205 | 298 |
| 0.249 | 0.215 | 298 |

## Where the signal comes from (test AUROC)

| Feature set | Test AUROC |
|---|---:|
| all features | 0.721 |
| no reviewer-activity features | 0.606 |
| reviewer activity only | 0.699 |
| business trend only | 0.520 |
| text reuse only | 0.678 |

## Coefficients (standardised, largest first)

| Feature | Coefficient |
|---|---:|
| reviewer_prior_reviews_log | -2.831 |
| reviewer_text_max_sim | +1.585 |
| reviewer_text_avg_sim | -0.879 |
| reviewer_max_in_7_days | +0.496 |
| own_text_len_log | -0.484 |
| reviewer_min_gap_days_log | -0.348 |
| reviewer_min_gap_days_log (missing) | -0.308 |
| business_prior_reviews_log | -0.196 |
| business_reviews_in_window | -0.172 |
| business_burst_ratio | +0.164 |
| reviewer_rating_std (missing) | +0.153 |
| reviewer_avg_rating (missing) | +0.153 |

## Limits

- The label is Yelp's filter output. Reviewer-activity features probably mirror how that filter works, so this model may learn to predict Yelp's filter, not fraud.
- One sample, one split. The live app uses a trimmed history index; counts there can differ from the full dataset.
- The test period was used once, but this sample was part of earlier development work, so treat the result as a baseline to beat on a fresh, untouched holdout, not as a deployment decision.
- Probabilities are only as calibrated as the table above shows; do not use them as fraud probabilities.
