# Review Trust Agent — PM Dashboard (public demo)

A live, interactive demo of an agentic fraud-detection system for Yelp
reviews, built to show product-manager-level control over an AI agent: an
editable system prompt, temperature, model provider, and confidence
threshold, with one-click deploy and MLflow-backed before/after performance
tracking. This is a real Streamlit app running real Python — not a mockup.

Full project write-up (case study, decision log, failure cases, eval
methodology): see the [Review Trust Agent](https://github.com/PrathamNawal/agentic-pm)
project this was extracted from.

## Try it

- **Human Review** and **System Config** and **Performance Tracking** work
  immediately, no API key needed.
- **Agent Review** (the live agent) needs your own API key for one provider —
  paste it in the sidebar (session-only, never stored). Get a free key in
  under a minute: [Groq console](https://console.groq.com/keys) (the default
  configured provider) or [OpenRouter](https://openrouter.ai/keys).

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
  accuracy, 84% agent accuracy from the original project) always reload on
  restart since they come from the checked-in CSVs.

## Run it locally instead

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
