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

## Pages

| Page | What it does |
|---|---|
| **Overview** | Top-line metrics + performance comparison across every config version deployed so far |
| **Human Review** | Act as the analyst — real reviews, real evidence, blind judgment, scored live |
| **Agent Review** | The real agent investigates live: real tool calls, real routing decision (auto-remove / queue / ignore), explicit low-confidence fallback |
| **System Config** | Edit the system prompt, temperature, provider, model, and auto-remove threshold. One click deploys a new version with an auto-filled diff reason if you leave the reason blank. Roll back to any past version. |
| **Performance Tracking** | Every judgment (human or agent), grouped by config version, pulled straight from MLflow — accuracy, precision, recall, confidence intervals, avg turns |

## Resetting to a clean state

```bash
rm data/agent_configs.json data/mlflow.db
```

Next launch reseeds MLflow with this project's real historical baselines (the
golden-set human labels and the 50-review Claude-manual batch) and starts
config versioning fresh at `v0`.
