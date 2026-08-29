# Agent Tools Reference

This document explains the three evidence-gathering tools the agent (and the human
golden-set reviewer) use to investigate a review, exactly as implemented in
[scripts/tools.py](scripts/tools.py). It's written so someone new to the project
can understand what each number means without reading the source first.

All three tools query the **full 608,598-review dataset**, not just the 5,000-review
eval sample — a reviewer's or business's real history lives outside whatever subset
happens to be under evaluation. The fast lookup indices that make this possible
(`by_user`, `by_prod`) are built once by [scripts/data_index.py](scripts/data_index.py)
and cached to disk.

---

## The self-exclusion mechanism

Every tool accepts an `exclude_review_id`. This exists so a review is never
compared against itself — without it, `text_similarity_check` would always find a
"similar" review with similarity ≈ 1.0 (itself), and `reviewer_history_lookup`
would count the review under investigation as part of its own history.

You don't pass this manually when calling through the agent — [scripts/tool_schemas.py](scripts/tool_schemas.py)'s
`dispatch_tool_call()` figures it out automatically: it only excludes the review
when the tool's target (`user_id`, `prod_id`, etc.) matches the review currently
under investigation. If the agent looks up a *different* reviewer or business
(e.g., cross-checking a second account for a coordinated campaign), nothing gets
excluded — there's no self-comparison risk there.

---

## 1. `reviewer_history_lookup(user_id, exclude_review_id=None)`

**What it answers:** "Does this reviewer's overall posting behavior look normal?"

**How it works:** pulls every other review by this `user_id`, then computes:

| Field | Meaning |
|---|---|
| `other_review_count` | How many other reviews this user has. `0` triggers the singleton-reviewer case below instead of the fields below. |
| `avg_rating` | Mean star rating across their other reviews. |
| `rating_std` | Standard deviation of their ratings. A reviewer who gives only 1-star or only 5-star reviews (std ≈ 0) posts less naturally than one with a normal spread. |
| `min_gap_days_between_reviews` | The smallest number of days between any two consecutive reviews by this user. A gap of 0 means they posted more than once on the same day. |
| `max_reviews_in_any_7_day_window` | The most reviews this user posted in any rolling 7-day period across their whole history — the reviewer-side burst signal. |
| `first_review_date` / `last_review_date` | Span of their activity. |

**The singleton case:** if `other_review_count` would be 0 (a reviewer with only
one review, ever — the one under investigation), the tool returns early with a
`note` instead of computing statistics on an empty set:

```
{"user_id": "...", "other_review_count": 0,
 "note": "Singleton reviewer — no other reviews found for this user. Singleton
          accounts are themselves a known spam signal in this domain."}
```

This isn't a missing-data failure — a brand-new, one-and-done account is itself
one of the most common fake-review patterns in the literature (create an account,
post once, never return), so the tool surfaces that as a signal rather than
silently returning zeros.

---

## 2. `business_trend_lookup(prod_id, around_date, window_days=14, exclude_review_id=None)`

**What it answers:** "Was there an unusual spike in review activity at this
business around the time this review was posted?"

**How it works:**

```python
baseline_rate = other_review_count / overall_span_days      # reviews/day, all-time average
window_rate   = reviews_in_window / (2 * window_days)        # reviews/day, in the ±14-day window
burst_ratio   = window_rate / baseline_rate
```

`overall_span_days` is the gap between the business's first and last review ever.
`window_days` defaults to 14, so the window covers 28 days total (14 before and
14 after `around_date`).

| Field | Meaning |
|---|---|
| `other_review_count` | Total reviews this business has (excluding the one under investigation). |
| `avg_rating_overall` | Mean rating across the business's full history. |
| `reviews_in_window` | How many reviews landed in the ±14-day window. |
| `avg_rating_in_window` | Mean rating just within that window — compare to `avg_rating_overall` to see if the window's reviews skew unusually positive. |
| `burst_ratio` | `1.0` = normal pace for this business. `>1.0` = faster than usual (the higher, the bigger the spike — a `burst_ratio` of 3-5x+ is the classic signature of a paid/coordinated review campaign). `<1.0` = slower than usual. |

**Known limitation:** this is a simple average-based ratio, not a statistical
anomaly test. It doesn't know about legitimate reasons for a genuine spike (a
grand reopening, a viral moment, a press mention) — a human or the agent still
has to weigh `burst_ratio` against the other evidence, not treat it as a verdict
on its own.

**No-data case:** a business with zero other reviews returns `other_review_count: 0`
and a `note`, with no ratio computed (would be a divide-by-zero).

---

## 3. `text_similarity_check(review_text, compare_against, target_id, exclude_review_id=None, top_k=3)`

**What it answers:** "Does this review's language look templated or copy-pasted
relative to other reviews by the same reviewer, or at the same business?"

**How it works:**
1. `compare_against` is `"reviewer"` or `"business"` — selects whether `target_id`
   is looked up in `by_user` or `by_prod`.
2. Pulls up to the **300 most recent** reviews for that target (`MAX_COMPARISON_TEXTS`,
   a performance cap — comparing against thousands of reviews per call would be slow
   with no real gain in signal).
3. Vectorizes the review under investigation plus all candidate texts using
   **character n-grams (3-5 characters, with word-boundary awareness)** via
   scikit-learn's `TfidfVectorizer(analyzer="char_wb")`.
4. Computes cosine similarity between the review and every candidate.

**Why character n-grams, not word-level similarity:** character-level TF-IDF
catches near-duplicate phrasing even when a spammer lightly edits a template
(swapping a few words, changing punctuation) — word-level comparison is easier
to dodge with minor rewording.

| Field | Meaning |
|---|---|
| `max_similarity` | The single highest cosine similarity score (0.0-1.0) found against any candidate review. Close to 1.0 means near-identical text exists elsewhere. |
| `avg_similarity` | Mean similarity across all candidates — a high average (not just one outlier match) suggests the reviewer/business's reviews are broadly templated, not just one coincidental overlap. |
| `compared_against_n_reviews` | How many candidate reviews were actually compared (capped at 300). |
| `similar_snippets` | The top `top_k` (default 3) most similar reviews, each with its score and a 150-character preview — lets a human (or the agent) sanity-check *why* the score is high, not just trust the number blindly. |

**No-data case:** if the target has no other reviews at all, returns
`max_similarity: 0.0` with an explanatory `note` rather than crashing on an empty
comparison set.

---

## Reading the numbers together

None of these three tools is meant to be decisive alone — a single high similarity
score, a singleton reviewer, or a business burst can each have an innocent
explanation (a genuine reviewer who rarely posts, a restaurant's real grand
opening, a reviewer who happens to write in a common style). The point of the
multi-tool, multi-turn design (see [PROCESS.md](PROCESS.md) §3.1-3.2) is that the
agent — and you, in the golden-set exercise — should weigh these signals against
each other, and pull more evidence (e.g., checking a *second* reviewer at the
same business) when they don't clearly agree, rather than keying on one number in
isolation.
