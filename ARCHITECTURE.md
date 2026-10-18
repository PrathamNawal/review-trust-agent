# Review Trust Agent — Architecture and Modularization

This is the current implementation map and the working contract for the modularization effort. `PROCESS.md` remains useful as decision history, but some sections describe earlier prototypes and files that are no longer in the repository. For the running system, use this document together with the code paths below.

## Current system

```text
Streamlit shell/pages (dashboard/app.py)
        │
        ├── page views (dashboard/pages/) and reusable presentation components (dashboard/ui_components.py)
        ├── configuration (dashboard/config_store.py)
        ├── investigation service (dashboard/agent_service.py)
        │       └── architecture registry (dashboard/agent_architectures.py)
        │               └── Agno runner (dashboard/live_agent.py)
        │                       └── provider registry (dashboard/providers.py)
        ├── decision policy (dashboard/decision_policy.py)
        ├── metrics and run tracking (dashboard/metrics.py, dashboard/mlflow_tracking.py)
        └── data/evidence (scripts/data_index.py, scripts/tools.py, scripts/tool_schemas.py)
```

The UI reads the 5,000-review sample in `data/sample.csv`. Evidence lookups use the tracked, compressed index in `data/full_index.pkl.gz`. Historical evaluation records are seeded from the golden-set and manual-batch CSV files; live judgments and user-configured versions are stored in local MLflow/JSON files.

## Shared contracts

- **Review input:** `review_id`, `user_id`, `prod_id`, `rating`, `date`, `review_text` (`dashboard/contracts.py`).
- **Agent configuration:** `architecture`, `provider`, `model`, `temperature`, `system_prompt`, `confidence_threshold`, and `version`.
- **Trace events:** `turn`, `tool_call`, `final`, or `error`. UI pages consume the same event shape regardless of architecture.
- **Evidence tools:** tool descriptions and JSON definitions are in `scripts/tool_schemas.py`; runtime handlers register against those names, and self-exclusion is kept in the shared dispatch path.
- **Decision policy:** `compute_routing()` in `dashboard/decision_policy.py` maps a judgment and confidence threshold to `auto_remove`, `queue`, or `ignore`.
- **Metrics:** `dashboard/metrics.py` owns fraud-positive binary metrics and Wilson intervals. Undefined metrics stay `None` rather than being presented as zero.

## Current plug-in points

- **Agent architecture:** register a runner in `dashboard/agent_architectures.py`. It receives the shared review/config contract and yields trace events. `agno` is the only registered implementation today.
- **Model provider:** add a provider model class and API-key environment variable in `dashboard/providers.py`; orchestration remains in the architecture runner.
- **Prompt and settings:** prompt/config versions live in `dashboard/config_store.py`. The default prompt is sourced from `scripts/tool_schemas.py`; each deployed version preserves its own prompt and architecture selection.
- **Metrics:** add calculations in `dashboard/metrics.py`, aggregate them in `dashboard/mlflow_tracking.py`, and expose them in the Track Performance view in `dashboard/pages/performance.py`.
- **Evaluation:** `dashboard/evaluation.py` evaluates and compares configurations on identical fixed examples; the Configuration view can run an optional baseline/candidate review before deployment. These evaluations are live calls and do not write to MLflow.
- **Jev experiment adapter:** `dashboard/jev_model.py` calls TypeSafe System One using the documented `api.typesafe.ai` endpoint and `TYPESAFE_API_KEY`. `scripts/run_jev_experiment.py` runs the standalone, repeatable comparison and saves local Git-ignored row-level reports. This is an experiment path, not the active dashboard agent architecture.
- **Evidence:** implement the evidence function in `scripts/tools.py`, define its model-facing contract and register its handler in `scripts/tool_schemas.py`. The Agno adapter binds model-callable functions in `dashboard/live_agent.py`.

## Guardrails for changes

1. Keep current page flows and the current trace event shape working while extracting modules.
2. Compare any agent change on the same fixed review IDs and record provider, model, architecture, prompt/config version, and evaluation set.
3. Treat recall on the filtered/fraud class as a primary outcome. Accuracy alone is misleading on this imbalanced dataset.
4. Keep the Yelp label caveat visible: the label is the dataset's reference signal, not a verified fraud ground truth.
5. Don’t replace an existing implementation until the new path can run alongside it and the comparison is recorded.

## Remaining modularization work

- Add a candidate-versus-baseline evaluation workflow in the UI, using fixed review IDs and a review gate before deployment.
- Add durable experiment records so evaluation results survive Streamlit session resets and can be linked to deployed versions.
- Jev's paired 3,000-review holdout comparison is complete: decomposed workflow recall 7.8% and accuracy 83.2%, versus v1 at 4.8% and 84.2%. It did not meet the ≥20% recall / ≥85% accuracy target. The holdout is consumed; any Agno comparison or revised candidate needs a new untouched evaluation set. See `experiments/jev-recall-experiment-report.md`.
- Persist evaluation records and support repeatable comparisons across multiple candidate versions; the current UI can evaluate up to the full 40-review golden set and reports recall-focused metrics.
- Add tests around evidence self-exclusion, tool outputs, config migration, architecture adapters, and evaluation aggregation.
- Keep config/run storage behind small repository interfaces so local files can later be replaced without changing the UI or agent.

The current repo has characterization tests for routing, metrics, prompt-source consistency, config migration, and architecture dispatch in `tests/test_modular_boundaries.py`. These are the first safety net, not full coverage of the current app.
