# Review Trust Agent

**An agentic AI prototype for Yelp review trust and safety triage.** It gathers
reviewer, business, and text evidence, explains its recommendation, and routes Jev
decisions to a human reviewer.

[Try the live dashboard](https://review-trust-agent.streamlit.app/) ·
[Read the product case study](PRODUCT_CASE_STUDY.md) ·
[Read the experiment report](experiments/jev-recall-experiment-report.md)

**Evaluation snapshot:** 2026-10-18 · **Status:** research prototype; Jev is advisory and
requires human review.

## The problem

Reviews shape consumer choices and business reputations, but coordinated or misleading
reviews can distort that signal. A review judged in isolation may not show the pattern
that matters: a reviewer's activity, a sudden change in a business's review volume, or
reused text.

The product question is whether an AI agent can collect those signals and help a Trust &
Safety analyst investigate a case consistently, while keeping its evidence and limits
visible. YelpZip's `filtered` label is used as a benchmark reference; it is not verified
fraud ground truth.

## Product thesis

Give analysts an evidence-backed recommendation and a reviewable trail, not an
uninspectable verdict. Keep people in control until the system demonstrates reliable
performance on representative, untouched data and its confidence is calibrated for the
decision.

```text
Review → gather reviewer, business, and text evidence → model recommendation
       → show rationale and evidence → human review
```

The prototype explores agentic investigation and policy trade-offs. It is not integrated
with Yelp's production pipeline and does not automatically remove reviews.

## Product leadership demonstrated

This project makes the product decisions and evaluation work visible alongside the demo:

- **Problem framing:** balance the cost of missing manipulation against wrongly flagging
  a genuine review; accuracy alone can hide missed positives.
- **Product design:** make the investigation and evidence inspectable, and preserve a
  human decision point.
- **System choices:** build a modular Streamlit prototype where model and decision policy
  can be changed and evaluated; keep Agno and Jev workflows available.
- **Success criteria:** target at least **20% recall while maintaining at least 85%
  accuracy** against YelpZip's filtered labels.
- **Risk judgment:** keep Jev advisory and route its outputs to a human because its
  probabilities are not calibrated for Yelp fraud.
- **Evidence-led iteration:** compare methods on paired examples, keep holdout results
  separate from development tuning, document failures, and decline to claim the target
  has been achieved.

The work includes a live product prototype, evaluation design, paired experiments,
results reporting, and an explicit recommendation to keep testing rather than claim
success prematurely. See the [case study](PRODUCT_CASE_STUDY.md) for the product
hypothesis, users, trade-offs, risks, and detailed decisions.

## Evaluation evidence

Results below use different samples and methods. Treat them as separate evaluations, not
as a single leaderboard. Recall measures the share of YelpZip-filtered examples found;
accuracy measures agreement with the dataset labels.

| Method | Evaluation sample | Recall | Accuracy | What the result says |
|---|---|---:|---:|---|
| Original dashboard agent | 50 reviews; 6 filtered | 0% | 84.0% | Historical baseline; missed all 6 filtered examples. |
| Jev v1 | Locked 3,000-review holdout; 397 filtered | 4.8% | 84.2% | Completed benchmark comparison. |
| Decomposed Jev, threshold 0.98 | Same locked 3,000-review holdout | 7.8% | 83.2% | Found 12 more positives than Jev v1, with 40 more false positives. |
| Jev Choice | Fresh holdout; 961 of 3,000 planned, 119 filtered | **9.2%** | 84.2% | Interim paired evaluation; not a completed 3,000-review result. |
| Jev Noul, overall risk at 0.50 | Same 961 completed cases | 4.2% | **85.7%** | Only compared methods above the 85% floor in this interim sample. |

**Current readout:** no tested method has met both the ≥20% recall and ≥85% accuracy
targets. Choice currently provides the highest observed recall in the partial comparison,
but misses the accuracy floor. The Jev results do not establish live dashboard performance:
the app uses a bundled, trimmed evidence index, while the offline experiments used the
evaluation data and setup described in the [experiment report](experiments/jev-recall-experiment-report.md).
The 3,000-review Jev holdout is consumed; the 961-case interim holdout is also consumed
for tuning. A new candidate needs a new untouched evaluation set.

## What I learned and what comes next

- A small batch can produce an encouraging number without showing that a policy
  generalizes. Preserve a locked holdout and report sample size with every metric.
- High accuracy can coexist with poor recall on an imbalanced dataset. Report recall,
  false positives, and queue rate alongside accuracy.
- Changing the agent's prompt or decision primitive did not reach the joint target in the
  measured evaluations. The next research direction is to test supervised YelpZip
  activity and graph features, with Jev used for structured evidence extraction only if
  it adds measurable value.
- Keep every Jev decision under human review until a fresh evaluation supports a safer
  policy and confidence calibration is validated.

These are benchmark findings, not a claim about verified real-world fraud detection.
The [full report](experiments/jev-recall-experiment-report.md) includes confusion
matrices, confidence intervals, development runs, cost estimates, and limitations.

## Try the prototype

- **Human Review**, **System Config**, and **Performance Tracking** work without an API
  key.
- **Agent Review** uses experimental Jev Choice by default and requires a TypeSafe API
  key. In **Play — Tweak the Agent**, configure Jev Choice, Score, or Noul, edit
  method-specific instructions, select evidence sources, compare a candidate, and deploy
  it as a version. Score/Noul cutoffs only map model outputs to labels; all Jev results
  stay in human review. Agno is also available with Groq, OpenRouter, Gemini, or Claude
  credentials.
- Keys entered in the sidebar are session-only and are not written to project files.
- The app runs on Streamlit Community Cloud's free tier. Session judgments and deployed
  config versions may reset when the app restarts.

## Data and scope

- The research dataset is [YelpZip](https://odds.cs.stonybrook.edu/yelpzip-dataset/),
  containing about 608,000 reviews. The full dataset is not included in this repository.
- The live demo ships a trimmed index: the 5,000-review evaluation sample and limited
  reviewer and business history around those reviews. It does not ship or query the full
  YelpZip corpus.
- Dataset `filtered` labels represent Yelp's filtering outcome, not independently
  verified fraud.
- This is a research prototype, not a Yelp product, production moderation service, or
  validated autonomous decision system.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Human Review and Performance Tracking can be explored without model credentials. Add API
keys through the app's session-only sidebar inputs or Streamlit secrets for Agent Review.

## Project documents

- [Product case study](PRODUCT_CASE_STUDY.md) — problem, users, product hypothesis,
  trade-offs, risks, and walkthroughs.
- [Architecture](ARCHITECTURE.md) — system boundaries, modular components, and extension
  points.
- [Jev experiment plan](experiments/jev-recall-experiment.md) — design and safeguards.
- [Jev execution report](experiments/jev-recall-experiment-report.md) — experiment
  results, metrics, and limitations.
