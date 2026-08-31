# Review Trust Agent — PM Dashboard

A real Streamlit app, not a mockup — it runs the project's actual Python modules
(`scripts/tools.py`, `scripts/tool_schemas.py`, `scripts/data_index.py`) directly,
makes real live LLM calls (Groq, OpenRouter, Gemini, or Claude — your choice, your
own API key), and logs every judgment to a local MLflow instance for real
before/after comparison across config versions.

## Run it

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Stop with `Ctrl+C`.

## What's real vs. what's local-only

- **Real:** live agent tool-calling loop, live LLM inference (any of the 4
  providers), the evidence functions, the MLflow-backed metrics, config
  versioning and rollback.
- **Local-only (by design, not a limitation to fix):** this runs as a local
  server, not a public URL. Config versions live in `data/agent_configs.json`;
  MLflow data lives in `data/mlflow.db` (SQLite) — both are plain local files,
  easy to inspect, back up, or reset.

## API keys

Enter them in the sidebar (session-only, never written to disk) or set as
environment variables before launching: `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
`GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`.

## Personas

The sidebar's **"I am a..."** picker (and the welcome dialog on first load) routes to
whichever flow matches a real end-to-end user journey, not just a nav label:

| Persona | JTBD | Lands on |
|---|---|---|
| 🕵️ **Trust & Safety Analyst** | Review the agent's already-investigated cases fast, with a visible reasoning trail | Analyst Queue |
| 🎛️ **Policy Owner / PM** | Tune the agent and see real before/after impact | Play — Tweak the Agent |
| 📝 **Reviewer / Business** | Understand why a decision was made and contest it | Check My Review |
| 🔎 **Just exploring** | The default guided tour | What is this? |

An appeal filed on **Check My Review** lands directly in the **Analyst Queue** as a
flagged case — the three personas aren't three disconnected demos, they're one loop.

## Pages

The app is a guided funnel, numbered in the order a first-time visitor should click through:

| Page | What it does |
|---|---|
| **1. What is this?** | The hook: the problem statement, three headline numbers (56% human accuracy, 84% agent accuracy, 0% agent fraud recall), and a map of what you can do here |
| **2. How it works** | Plain-language walkthrough of the evidence loop and the three tools, before any interaction is asked of you |
| **3. Try it — Human** | Act as the analyst — real reviews, real evidence, blind judgment, scored live. Ends with a CTA to see the agent judge the *same* review |
| **4. Try it — Agent** | The real agent investigates live: real tool calls, real routing decision (auto-remove / queue / ignore), explicit low-confidence fallback. Inline 3-step guide to getting a free API key if none is set |
| **5. Analyst Queue** | The real analyst workflow: the agent's already-computed verdict, confidence, and evidence trail for a case, with Accept / Override / Escalate — every decision logged, so agreement and override rate become real metrics instead of silent disagreement. Appealed cases (from page 6) surface here first. |
| **6. Check My Review** | As the reviewer/business behind a review, see the real routing decision and plain-language reasoning — ground truth is deliberately not shown, since a real reviewer wouldn't know it either — and appeal an auto-removal, which files a real, trackable case into the Analyst Queue. |
| **7. Play — Tweak the Agent** | Edit the system prompt, temperature, provider, model, and auto-remove threshold — or click a one-click "quick experiment" (more cautious / faster & cheaper / try Claude) to pre-fill the form. One click deploys a new version with an auto-filled diff reason. Roll back to any past version. |
| **8. Track Performance** | Leads with a plain-language before/after delta ("your last config change moved accuracy from X% to Y%"), then the full MLflow-backed breakdown — accuracy, precision, recall, confidence intervals, avg turns |

## Resetting to a clean state

```bash
rm data/agent_configs.json data/mlflow.db
```

Next launch reseeds MLflow with this project's real historical baselines (the
golden-set human labels and the 50-review Claude-manual batch) and starts
config versioning fresh at `v0`.
