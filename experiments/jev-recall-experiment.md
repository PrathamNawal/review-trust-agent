# Jev recall experiment

**Status:** Four paired tuning iterations completed; no candidate meets both target metrics; evidence redesign needed before development selection  
**Owner:** Review Trust Agent project  
**Goal:** Determine whether Jev improves detection of Yelp-filtered reviews when used as a structured decision layer alongside the current evidence-gathering agent.

## Why this experiment exists

The current 50-review manual batch had 0/6 recall on Yelp-filtered reviews. That is a useful failure signal, but six positive examples are too few to establish general performance. Jev is being considered as a second-stage structured classifier, not assumed to be a recall solution. Its performance on this dataset is unknown.

Yelp's `filtered` field is the dataset's reference label, not verified fraud ground truth. Results must be described as agreement with that reference label.

## Hypothesis

Given a review and the same evidence available to the current system, Jev can increase recall on filtered reviews while keeping precision, false-positive rate, human-review volume, completion rate, latency, and API cost within thresholds agreed before the full run.

## Candidate system

1. The local evidence pipeline gathers reviewer history, business trend, and text-similarity evidence using only reviews dated before the target review.
2. A Jev adapter receives the review and those evidence outputs. It does not receive the ground-truth `filtered` label, review ID-derived labels, or other evaluation-only data.
3. Jev returns a typed `filtered` / `recommended` decision. Confidence/probability is treated as an unvalidated model output.
4. The application maps the decision to a review/queue outcome through the project's own decision policy. Jev does not directly remove or suppress reviews during this experiment.

If Jev's API contract cannot support this input/output shape, revise the adapter contract before running; do not silently change the experiment mid-run.

## Comparisons

Run these candidates on identical review IDs and evidence:

- **Prompt baseline:** Jev with the existing v1 instructions.
- **Tuned prompt:** the selected development-set Jev instruction version.
- An Agno comparison can be a later arm if its separate provider budget is authorized and the same time-aware evidence can be supplied.

Record exact provider/model identifiers, API/schema version if exposed, prompt/rubric, decision mapping, code commit, dataset version, and run date for each candidate. Do not tune against the final held-out set.

## Verified integration notes

Checked Jev's published documentation on 2026-10-18. The documented REST endpoint is
`POST https://api.typesafe.ai/v1/systemone`, authenticated with a Bearer API key. The
request has `model`, `state`, and `questions`; Jev documents `choice`, `score`, and
`noul` question types. The response includes typed answers and usage fields, and may
include cost. The docs caution that probability/confidence are not a guarantee of
business accuracy. We should therefore use `TYPESAFE_API_KEY` only as a server-side/local
environment variable, and continue to calibrate outputs against our dataset.

TypeSafe currently publishes Jev input at $0.042 per million tokens and output as free.
The observed 50-review input volume estimates about $0.14 per 3,000 reviews. Recheck
the public rate and actual account balance before calls; account terms and deductions
may differ.

Sources: [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart).

## Evaluation plan

### Project success criteria and tuning cadence

- **Primary target (current):** at least 20% filtered-class recall and at least 85% accuracy, with higher recall than paired Jev v1. The earlier target of 50% recall is historical.
- **Accuracy guardrail (current):** at least 85%, per the project's latest instruction. Earlier prompt-only and first development iterations used 84% or 80%; those are historical.
- **Development loop (completed):** ten paired 50-review batches from the frozen 500-review development partition (1,000 Jev requests). Thresholds selected on earlier batches were 0.97 for batches 0–2 and 0.98 for batches 3–9. Across all cases under this sequential held-forward policy, candidate recall was 16.7%, accuracy 83.2%, queue rate 8.0%; v1 was 9.1%, 85.8%, and 3.4%. The candidate missed the revised target.
- **Locked final evaluation (completed once):** at the project owner's request, Jev v1 and the decomposed candidate were paired on all 3,000 final reviews. The candidate used the fixed 0.98 business non-burst threshold. Candidate performance was 7.8% recall and 83.2% accuracy, versus 4.8% and 84.2% for v1. Neither met the revised target of ≥20% recall and ≥85% accuracy. Do not tune on this consumed holdout; any revised approach needs a new untouched final set.
- **Metric noise:** development batches were small and repeated comparisons can overfit. Final results are more precise but still benchmark YelpZip's filtered labels, not verified fraud ground truth.
- **Working interpretation of “fine-tune”:** prompt/rubric and decision-threshold iteration only. The current Jev API integration does not train or update Jev model weights.

### Completed tuning log — 2026-10-18

Four paired 100-call iterations used the frozen 50-review tuning partition (7 filtered examples). The table reports the newer prompt arm; all compared arms completed 50/50 decisions.

| Version | Change recorded before calls | Recall | Accuracy | Decision |
|---|---|---:|---:|---|
| v2 | Require specific behavior evidence; prevent sentiment-only flags | 0/7 (0%) | 80% | Reduced false positives from v1's 4 to 3, but still missed all positives |
| v3 | Explicitly favor recall; accept moderate specific evidence | 0/7 (0%) | 76% | More queue cases but no recovered positives; rejected |
| v4 | Flag at least one meaningful behavior signal without requiring proof | 3/7 (42.9%) | 60% | Best recall so far, but 16 false positives; rejected against 84% accuracy guardrail |
| v5 | Require one strong or two corroborating moderate signals | 0/7 (0%) | 82% | False positives fell to 2, but all positives were missed; rejected |

No prompt-only candidate met the earlier recall/accuracy targets. On the completed development and final comparisons, the decomposed Jev workflow failed the revised joint target of ≥20% recall and ≥85% accuracy. Consider adding a supervised model using YelpZip graph and activity features, with Jev limited to structured evidence extraction, then evaluate the revised approach using a new untouched final set.

### Before live calls

- Re-check Jev's current API docs and confirm supported request/response schema, authentication, rate limits, and price before key use. The current public input rate is $0.042/M tokens, output is free; confirm account deductions in the TypeSafe console.
- Implement Jev behind a separate adapter/configuration path; leave the current Agno path as the baseline and rollback option.
- Add sanitized fixtures or mocked adapter checks for valid responses, malformed responses, timeout/rate-limit errors, and missing credentials. Never use or commit a real key in these checks.
- Confirm the evaluation runner passes only inference inputs to the candidate and keeps labels solely in the scoring path.
- Use the frozen 50-case tuning, 500-case development, and 3,000-case final partitions. These exclude the existing 5,000-review sample and documented examples used during development.
- Track precision, false-positive rate, queue rate, completion, latency, and cost as guardrails alongside the user-set recall and accuracy targets. Record any tradeoff; do not tune on the final holdout.

### Live evaluation

- Start with a small smoke run (4–8 fixed reviews) to validate request/response handling, evidence serialization, label isolation, and cost reporting. Treat it as integration validation, not evidence of efficacy.
- First compare prompt versions on the 50 tuning IDs in paired 100-call cycles; use the separate 500-case development set to select/freeze a version. Then evaluate v1 and the selected prompt paired on the untouched 3,000-case final holdout (6,000 Jev calls total). Do not change prompts during or after that final run.
- Save run metadata, per-review decisions/errors, aggregate metrics, and cost/latency. Keep credentials and sensitive request headers out of logs.
- Inspect disagreements and false negatives after the locked comparison. Any prompt/rubric change creates a new candidate version and requires a fresh evaluation set or a clearly labeled exploratory rerun.

### Metrics

Primary: filtered-class recall. Current accuracy floor: at least 85%. Earlier runs used 84% or 80%; these are historical guardrails.  
Guardrails: precision, false-positive rate/specificity, F1, balanced accuracy, completion/error rate, human-review/queue rate, latency, and API cost. Report confusion-matrix counts and confidence intervals where sample size permits. Accuracy alone is not a success criterion on this imbalanced dataset.

## API key handling

- No API key is needed for the documentation and adapter preparation stages.
- When the setup is ready, ask the project owner to provide the Jev key through an approved secret input (for example, a local environment variable or the app's password-masked, session-only key field). Never ask them to paste it into source code, a chat transcript, a checked-in `.env`, or an experiment log.
- Provide the rotated key as `TYPESAFE_API_KEY` in the process environment or the Git-ignored local `.env` file. The experiment runner reads only that variable and never prints it; do not expose it to browser code or paste it into chat/source files.
- Ensure `.env` and any local secrets are ignored by Git. Restrict `.env` permissions to the local user where possible. Redact credentials from errors and logs; do not print the key while checking setup.
- The completed experiment's estimated cumulative Jev use is about $0.352 at measured token rates, under the user's $5 cap. Actual account deductions may differ; verify in the TypeSafe console.

## Completion and decision

The planned Jev experiment is complete. The locked comparison did not meet the current target; revise the evidence/model architecture and obtain a new untouched holdout before making another final-performance claim.

Do not enable automatic removal based on Jev as an outcome of this experiment. Any later moderation deployment requires a separate review of evidence, thresholds, and operational safeguards.

## Current execution implementation

`scripts/run_jev_experiment.py` runs Jev against the fixed 50-review manual batch. It uses the same deterministic evidence tools (reviewer history, business trend, text similarity) and records each response, errors, token/cost fields if returned, elapsed time, confusion matrix metrics, and Wilson recall interval. The report output is written under `data/jev_experiment_runs/`, which is Git-ignored. Run locally with:

```bash
TYPESAFE_API_KEY=... python -m scripts.run_jev_experiment --limit 4  # integration smoke run
TYPESAFE_API_KEY=... python -m scripts.run_jev_experiment             # full 50-review exploratory run
```

The current comparison is paired by review ID against the stored manual Claude judgments. That is useful as an exploratory reference, but is **not** a controlled Agno-vs-Jev comparison: the baseline used manual Claude reasoning and is not guaranteed to have used the exact same evidence payload. A later controlled comparison should run the current Agno configuration on these same IDs with a verified provider, or create a fresh fixed evaluation set and run both systems with identical evidence.

The large-scale prompt run compares the original Jev v1 instructions with the selected tuned prompt on identical reviews and time-aware evidence. It estimates the effect of prompt changes; it does not establish gains over the current Agno system.

## End-to-end checklist

- [x] Experiment question and proposed role documented.
- [x] Baseline, candidate paths, metrics, and leakage cautions documented.
- [x] Key-handling expectations documented.
- [x] Use the TypeSafe quickstart endpoint, request/response types, and authentication method.
- [ ] Verify rate limits and actual credit/cost consumption in the TypeSafe console.
- [x] Implement an isolated Jev HTTP adapter, evidence payload, and error handling.
- [x] Add repeatable runner with per-review and aggregate report output.
- [ ] Add deterministic/mocked adapter and label-isolation checks.
- [x] Build and freeze separate tuning, development, and final-holdout manifests, excluding prior project examples.
- [x] Record initial targets and 100-call development cadence; update current target to at least 20% recall and 85% accuracy.
- [x] Run smoke evaluation and confirm TypeSafe integration.
- [x] Run the 50-review exploratory comparison; row-level and aggregate results are recorded.
- [x] Run the paired 3,000-review final comparison at the frozen 0.98 threshold; record results and keep the consumed holdout closed to tuning.
- [x] Record findings and next step in this report and `PROCESS.md`.

## Run log

| Date | Version/commit | Dataset & IDs | Candidate | Result / notes |
|---|---|---|---|---|
| 2026-10-18 | TypeSafe endpoint + `TYPESAFE_API_KEY` | First manual-batch ID | Jev smoke | Successful response; model `jev-1.13.0`, 877 input tokens, 35 output tokens. |
| 2026-10-18 | TypeSafe endpoint + `TYPESAFE_API_KEY` | Full manual batch (50 IDs) | Jev | 50/50 successful; 3/6 positives found (50% recall), 6 false positives; full metrics in `jev-recall-experiment-report.md`. |
| 2026-10-18 | TypeSafe endpoint + `TYPESAFE_API_KEY` | Frozen YelpZip final holdout (3,000 IDs) | Jev v1 vs decomposed policy at fixed 0.98 | 6,000/6,000 decisions completed; candidate 7.8% recall, 83.2% accuracy; v1 4.8%, 84.2%; estimated total project spend $0.352. Full metrics in `jev-recall-experiment-report.md`. |

The project owner supplied a credential in chat. It was not written to source files,
run reports, or outputs. Since it was exposed in chat, rotate it in the TypeSafe
console before further use; enter its replacement only in the masked local app key
field or a local environment variable.
