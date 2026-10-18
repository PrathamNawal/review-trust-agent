# Review Trust Agent

An AI agent that investigates whether a Yelp review is fake — and a live dashboard
that lets you second-guess it, tweak its behavior, and see the tradeoffs.

![Historical dashboard screenshot showing the original agent result; the separate Jev result is summarized below](assets/dashboard-hero.png)

**[Try the live dashboard →](https://review-trust-agent.streamlit.app/)**

**Best observed recall:** the experimental Jev Choice method reached **9.2% recall**
(11/119) and 84.2% accuracy on a fresh evaluation that stopped after 961 of 3,000
planned reviews. This is an interim result, not a completed 3,000-review evaluation.

**Best completed large-holdout result:** the decomposed Jev workflow reached **7.8% recall** (31/397) and
83.2% accuracy on a separate locked 3,000-review holdout, versus 4.8% recall and 84.2%
accuracy for Jev v1. The recall improved, but it missed the 85% accuracy target and was
not a direct comparison against Agno. The Streamlit Jev adapter now uses the direct
**Choice** method. Neither observed result meets the joint ≥20% recall / ≥85% accuracy
target. Jev remains experimental and every decision goes to human review. See the
[full results and limitations](experiments/jev-recall-experiment-report.md).

**Historical baseline:** the original dashboard agent scored 0% recall on six filtered
examples in a 50-review batch (84% accuracy); an always-genuine baseline scored 88%
accuracy on that same batch. These small-batch results predate the Jev experiments and
are not the current best recall result.

**[Read the full case study →](PRODUCT_CASE_STUDY.md)** — business problem, impact,
hypothesis, users & JTBD, scope & tradeoffs, evaluation plan, top risks, plus three real
hard-case walkthroughs.

**[Read the architecture & modularization guide →](ARCHITECTURE.md)** — current system
boundaries, shared contracts, experiment plug-in points, and the safe migration plan.

**[Review the Jev recall experiment plan →](experiments/jev-recall-experiment.md)** —
design, evaluation safeguards, key handling, and run log.
**[Read the Jev experiment report →](experiments/jev-recall-experiment-report.md)** —
3,000-review final results, prompt and development tuning, limitations, and next steps.

This dashboard is a real Streamlit app running real Python — not a mockup. Every
number, chart, and control is wired to a real computation (see "What's real vs. scoped
down" below).

## Try it

- **Human Review** and **System Config** and **Performance Tracking** work
  immediately, no API key needed.
- **Agent Review** uses experimental Jev Choice by default and needs a TypeSafe
  API key, entered in the sidebar for the current session or configured in Streamlit
  secrets. Agno is also available and can use Groq, OpenRouter, Gemini, or Claude keys.
  Keys entered in the sidebar are session-only and never stored in project files.

## What's real vs. scoped down for this public demo

- **Real:** every tool call (reviewer history, business trend, text
  similarity), every LLM call, every judgment, every config deploy — all
  running against genuine Yelp review data (the
  [YelpZip](https://odds.cs.stonybrook.edu/yelpzip-dataset/) research
  dataset).
- **Scoped down:** the full dataset is 608K reviews (~400MB) — too large for
  a public GitHub repo. This demo ships a trimmed index: every review in the
  5,000-review evaluation sample, plus each of those reviewers'/businesses'
  most recent real history (up to 40 other reviews per reviewer, 60 per
  business). It's genuine data throughout, just capped in volume rather than
  full-dataset scale — see `scripts/data_index.py`.
- **Not persistent:** this runs on Streamlit Community Cloud's free tier,
  which has an ephemeral filesystem. New judgments you log and new config
  versions you deploy will reset if the app restarts (goes to sleep after
  inactivity, or redeploys). The seeded historical baselines (56% human
  accuracy, 84% original agent accuracy from the original project) always reload on
  restart since they come from the checked-in CSVs.

## Run it locally instead

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
