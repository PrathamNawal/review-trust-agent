# Better features for the logistic-regression baseline

Offline experiment on the 5,000-review YelpZip evaluation sample; nothing here is wired into the live dashboard. The target is YelpZip's `filtered` label (Yelp's own filter outcome), not verified fraud.

## Design

- Same date split as the first baseline: train ≤ 2012-12-31, validation ≤ 2013-12-31, test after. Every feature uses only reviews dated before the one being described.
- Feature sets were compared **on validation AUROC only**; the test period is scored once for the chosen set (and for every set, shown for information — it did not influence the choice). The original baseline's test result had already been seen.
- Cutoffs are chosen on validation from true- and false-positive rates, with accuracy recomputed at a fixed 12% filtered share. The raw filtered share moved from 16.1% (2013) to 12.0% (2014), so raw validation accuracy is not a stable way to pick a cutoff.

## Feature sets

| Feature set | Features | Validation AUROC | Test AUROC |
|---|---:|---:|---:|
| original 17 features | 17 | 0.764 | 0.721 |
| + reviewer extras | 26 | 0.770 | 0.733 |
| + business extras | 22 | 0.775 | 0.731 |
| + text style | 24 | 0.754 | 0.720 |
| + all extras | 38 | 0.769 | 0.739 |
| + all extras + words-only text model ← chosen | 39 | 0.776 | 0.750 |

Chosen: **+ all extras + words-only text model**, C = 10.0. Test AUROC **0.750** (95% bootstrap CI 0.718–0.782); average precision 0.275 against a base rate of 0.120.

## Cutoff scores — chosen model (1,489 test reviews, 178 filtered)

| Cutoff rule | Score cutoff | Flagged | Recall | Precision | Accuracy (test) | Accuracy at 12% filtered (test) | same, on validation |
|---|---:|---:|---:|---:|---:|---:|---:|
| **best recall with accuracy ≥ 84% (chosen on validation)** | 0.348 | 8.5% | 23.6% | 33.3% | 85.2% | 85.2% | 84.0% |
| **best recall with accuracy ≥ 85% (chosen on validation)** | 0.387 | 6.0% | 16.3% | 32.6% | 86.0% | 85.9% | 85.5% |
| flag 2% of not-filtered reviews (validation) | 0.520 | 2.2% | 7.9% | 42.4% | 87.7% | 87.7% | 87.1% |
| flag 4% of not-filtered reviews (validation) | 0.422 | 4.4% | 13.5% | 36.4% | 86.8% | 86.8% | 86.6% |
| flag 6% of not-filtered reviews (validation) | 0.386 | 6.2% | 17.4% | 33.3% | 86.0% | 85.9% | 85.2% |
| flag 8% of not-filtered reviews (validation) | 0.352 | 8.2% | 23.0% | 33.6% | 85.4% | 85.3% | 84.1% |
| flag 10% of not-filtered reviews (validation) | 0.319 | 10.8% | 30.3% | 33.5% | 84.5% | 84.5% | 82.8% |
| flag 15% of not-filtered reviews (validation) | 0.264 | 15.2% | 37.1% | 29.1% | 81.7% | 81.6% | 79.5% |
| flag 20% of not-filtered reviews (validation) | 0.228 | 19.1% | 41.6% | 26.0% | 78.8% | 78.8% | 76.4% |

## Same cutoff rules — original 17 features, for comparison

| Cutoff rule | Score cutoff | Flagged | Recall | Precision | Accuracy (test) | Accuracy at 12% filtered (test) | same, on validation |
|---|---:|---:|---:|---:|---:|---:|---:|
| **best recall with accuracy ≥ 84% (chosen on validation)** | 0.276 | 5.1% | 11.2% | 26.3% | 85.6% | 85.6% | 84.5% |
| **best recall with accuracy ≥ 85% (chosen on validation)** | 0.283 | 4.5% | 10.1% | 26.9% | 86.0% | 85.9% | 85.1% |
| flag 2% of not-filtered reviews (validation) | 0.373 | 1.1% | 2.8% | 31.2% | 87.6% | 87.6% | 87.4% |
| flag 4% of not-filtered reviews (validation) | 0.328 | 2.0% | 5.1% | 30.0% | 87.2% | 87.2% | 86.2% |
| flag 6% of not-filtered reviews (validation) | 0.288 | 4.3% | 9.0% | 25.0% | 85.9% | 85.9% | 85.3% |
| flag 8% of not-filtered reviews (validation) | 0.272 | 5.4% | 12.4% | 27.5% | 85.6% | 85.6% | 84.3% |
| flag 10% of not-filtered reviews (validation) | 0.253 | 7.1% | 14.0% | 23.6% | 84.3% | 84.2% | 83.0% |
| flag 15% of not-filtered reviews (validation) | 0.227 | 10.1% | 18.5% | 21.9% | 82.3% | 82.3% | 79.4% |
| flag 20% of not-filtered reviews (validation) | 0.205 | 14.3% | 27.0% | 22.5% | 80.2% | 80.2% | 76.2% |

## Largest coefficients (standardised)

| Feature | Coefficient |
|---|---:|
| reviewer_prior_reviews_log | -1.969 |
| reviewer_text_max_sim | +1.884 |
| reviewer_text_avg_sim | -1.013 |
| own_text_len_log | -0.700 |
| reviewer_days_since_first_log | -0.641 |
| own_word_count_log | +0.551 |
| reviewer_prior_1 | -0.396 |
| text_model_score | +0.370 |
| reviewer_prior_2_3 | -0.368 |
| reviewer_prior_4_9 | -0.366 |
| business_own_rating_deviation | +0.281 |
| own_digit_share | -0.265 |

## Limits

- The label is Yelp's filter output; reviewer-activity features probably mirror it, so this may predict Yelp's filter, not fraud.
- One sample and one split; the live app's history index is trimmed. Confirm on a fresh, untouched holdout before any decision.
- With about 12% of reviews filtered, flagging nothing already scores 88% accuracy, and every flag lowers accuracy unless it is more than 50% likely to be filtered. An accuracy floor therefore caps how many reviews can be flagged; only higher precision raises the recall available under it.
