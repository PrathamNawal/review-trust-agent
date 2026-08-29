# Review Trust Agent — Product & Build Process

This document exists because of a specific moment in this project: the agent's system
prompt had already been written and was running against real API calls before the PM
(you) saw its text. That's backwards. In a real production build, nothing that shapes
model behavior — a prompt, a threshold, a schema, a model choice — should be invisible
to the person accountable for the product's judgment.

This document is the fix: a single place that shows (1) the full build process end to
end, (2) every prompt/config decision baked into the system so far, in plain text, and
(3) exactly which decisions are engineering execution vs. which ones require a PM's
sign-off before they go further. Anyone — a new team member, a reviewer, you in three
months — should be able to read this and understand not just *what* was built, but
*who decided what, and why*.

---

## 1. Roles: What Gets Executed vs. What Needs a PM's Sign-Off

A senior PM on an agentic AI product doesn't write the code, but does not delegate
away four categories of decision either. Splitting these explicitly is the point of
this section.

| Category | Who owns it | Examples from this project |
|---|---|---|
| **Judgment-shaping config** — anything that changes what the agent decides or how confidently | **PM reviews and approves before it runs at scale** | System prompt wording, tool descriptions (they double as instructions), the autonomy policy thresholds (Step 6), what counts as a "failure case" |
| **Architecture decisions** — what the system is made of | **PM decides the shape, engineering implements it** | 3 tools vs. 5, multi-turn loop vs. single-shot, which model/provider |
| **Methodology** — how we know if it works | **PM directs, engineering executes** | Stratified vs. balanced sampling, confidence-interval approach, eval sample size |
| **Plumbing** — execution details with no judgment content | **Engineering executes without review** | JSON parsing, retry logic, file I/O, pandas indexing, package installation |

**The mistake this document corrects:** the system prompt and tool descriptions
(Category 1) got built and run without being surfaced first — they were treated like
Category 4 plumbing. They aren't. Every prompt in Section 3 below should be read by
you before the next full eval run.

---

## 2. Build Stage Map

Each stage of [v0-plan.md](v0-plan.md)'s 9-step sequence, with status, where its
artifacts live, and — critically — whether it contains a Category 1/2/3 decision that
needs your review.

| Step | Status | Artifacts | Needs PM review? |
|---|---|---|---|
| 1. Stratified sample | ✅ Done, verified | [scripts/step1_sample.py](scripts/step1_sample.py), [data/sample.csv](data/sample.csv) | Methodology only — proportional vs. balanced stratification was a real call (see Decision Log #2) |
| 2. Tools + evidence loop | ✅ Done, verified live | [scripts/tools.py](scripts/tools.py), [scripts/data_index.py](scripts/data_index.py), [scripts/agent.py](scripts/agent.py), [scripts/agent_gemini.py](scripts/agent_gemini.py), [scripts/tool_schemas.py](scripts/tool_schemas.py) | **Yes — the system prompt and tool descriptions in `tool_schemas.py` are judgment-shaping. See Section 3.** |
| 3. Judgment schema + batch validation | Scoped, not yet implemented | — | Schema itself is Category 1 (already reviewed below); the *validation harness* is plumbing |
| 4. Full eval + statistics | Not started | — | Sample size / CI methodology is Category 3 |
| 5. Ground-truth caveat | Not started | — | **PM writes this — it's a judgment call about how much to trust your own numbers, not something to delegate** |
| 6. Autonomy policy | Not started | — | **PM decision, full stop.** This is the actual point of the project (see Section 4 of v0-plan.md) |
| 7. Real failure cases | Not started | — | PM should review which cases get told as the "3 stories" — that's narrative judgment |
| 8. Classifier baseline comparison | Not started | — | Methodology; verdict on "is the agent worth it" is a PM call |
| 9. Cross-domain transfer argument | Not started | — | **PM writes this — it's a strategic argument, not an engineering output** |

---

## 3. Prompt & Configuration Registry

Every piece of text or config currently shaping the agent's behavior, verbatim,
with where it lives and why it was written this way. This section should never
fall out of date — if a prompt changes, this section changes in the same commit.

### 3.1 System prompt
**Location:** `SYSTEM_PROMPT` in [scripts/tool_schemas.py](scripts/tool_schemas.py)
**Used by:** both `agent.py` (Claude) and `agent_gemini.py` (Gemini) — shared, not duplicated

```
You are a trust & safety investigator checking whether a Yelp review is likely
fake/manipulated (would be filtered) or genuine (would be recommended).

You have three evidence-gathering tools plus a submit_judgment tool to end the
investigation. You do not have to call all three tools, and you are not limited
to calling each one once. If the evidence so far is ambiguous, pull more of it
before deciding — for example, if the review's own text looks templated, check
a *different* reviewer at the same business to see if their language matches too
(a real coordinated campaign), or if a reviewer's history looks bursty, check the
business's trend for the same time window to see if the burst lines up.

Only call submit_judgment when you're confident you've gathered enough evidence
to justify your answer. If the signals genuinely conflict, your confidence should
reflect that — don't manufacture false certainty.
```

**Why it's written this way:** the plan's core requirement (Step 2) was that the
agent investigate rather than classify in one pass. The prompt does this by (a)
explicitly permitting repeat/cross-target tool calls, (b) giving a worked example
of *when* to pull more evidence, and (c) warning against manufactured confidence —
which directly targets Step 7's failure case #3 (signals disagree, does confidence
correctly drop?).

**What a PM should check:** does "don't manufacture false certainty" actually
produce well-calibrated confidence scores, or does it just make the agent hedge
everything to 0.5? This can only be answered empirically (Step 3/4), but it's
worth watching for in the batch validation.

### 3.2 Tool descriptions (these are instructions, not just documentation)
**Location:** `TOOL_DEFS` in [scripts/tool_schemas.py](scripts/tool_schemas.py)

Each tool's `description` field is sent to the model as part of its instructions —
it's not just internal documentation, it actively steers behavior:

| Tool | Description sent to the model |
|---|---|
| `reviewer_history_lookup` | "Look up a reviewer's posting history: review count, average rating, rating spread, and burst timing. Call this for the reviewer under investigation, **or for any other reviewer you want to cross-check** (e.g. a reviewer who left similar-looking text at the same business)." |
| `business_trend_lookup` | "Look up a business's review volume and rating trend around a given date. A sudden spike in review velocity (burst_ratio well above 1) around the review's date is a classic sign of an incentivized or coordinated campaign." |
| `text_similarity_check` | "Compare a piece of review text against a reviewer's or a business's other reviews for templated or near-duplicate phrasing. Use this on the review under investigation, **or on any other review text you've seen** (e.g. from a second reviewer) to check whether they read like the same template." |
| `submit_judgment` | "Submit your final judgment once you have enough evidence. Ends the investigation." |

**Why the bolded phrases matter:** they're what actually enables the "check a
second reviewer" behavior we saw in the live test. Without them, the model has no
signal that it's allowed to call a tool on a target other than the review at hand.
This was a deliberate design choice, not a default.

### 3.3 Per-review input format
**Location:** `format_review_prompt()` in [scripts/tool_schemas.py](scripts/tool_schemas.py)

```
Investigate this review:
- reviewer_id: {user_id}
- business_id: {prod_id}
- rating: {rating}
- date: {date}
- text: {review_text}
```

**Note:** the `filtered` label is never included here — the agent never sees the
ground truth it's being evaluated against. Worth a PM double-check before Step 4:
confirm the eval harness doesn't leak the label anywhere else either (e.g. via a
tool's return value).

### 3.4 Judgment output schema
**Location:** `submit_judgment` tool parameters in `TOOL_DEFS`

```
predicted_filtered: boolean
confidence: number, 0.0–1.0
primary_signal: one of ["reviewer_pattern", "business_trend", "text_similarity", "combination"]
reasoning: string, 1-2 sentences
```

**Why `primary_signal` is a closed enum, not free text:** forcing a choice from
these four options is what makes Step 7's audit possible later — you can group
failures by which signal the agent leaned on and see if one signal is
systematically unreliable. Free-text reasoning alone wouldn't support that
analysis without another LLM pass to categorize it.

### 3.5 Model & provider configuration
**Location:** `MODEL` constant in `agent.py` / `agent_gemini.py`; `MAX_TURNS = 6` in both

- Provider: Gemini 3.7 Flash (`gemini-3.7-flash`), switched from Claude Haiku per
  your explicit request mid-build (Decision Log #4). Claude version kept in sync,
  not deleted, in case of a later cost/quality comparison.
- Turn cap: 6 round-trips before the loop force-terminates with a `confidence: 0.0`
  fallback judgment. This is a plumbing safety valve, not a judgment call — but the
  number 6 was a guess, not derived from data. **Worth revisiting once we see actual
  turn-usage distribution in Step 3's batch validation** (the one live test used 3).

---

## 4. Decision Log

Chronological record of real decisions made in this build, in ADR-lite format:
what was decided, what else was considered, and why.

**#1 — Sampling strategy: proportional, not balanced**
Considered: an artificially balanced 50/50 sample (more filtered examples to learn
from) vs. a proportional sample matching the true 13.22% filtered rate.
Decided: proportional. Reasoning: precision/recall computed on an artificially
balanced sample doesn't reflect real-world prevalence, and would overstate how
often the agent needs to actually catch a filtered review in production.

**#2 — Dataset source: rejected two options before landing on the real one**
Considered: (a) the official Stony Brook ODDS download — blocked at the network
level from this environment; (b) an unverified single-CSV Kaggle mirror
(danaxu11/yelpzip) — matched row count but had unknown provenance and no
description. Decided: a public Google Drive mirror in the original multi-file
research format, verified against the exact published class-balance statistics
(13.22% filtered) before trusting it.

**#3 — Evidence loop: multi-turn tool use, not single-pass**
This was specified in the plan before build started (not a mid-build pivot), but
it's the single most important architectural decision in Step 2: the system
prompt and tool descriptions were explicitly written to permit calling a tool
again on a *different* target, verified working in the first live test.

**#4 — Provider switch: Claude → Gemini 3.7 Flash**
You requested this mid-build. Required rewriting the tool-calling loop against
Gemini's Interactions API (`client.interactions.create`), which uses a different
request/response shape than Anthropic's Messages API (`previous_interaction_id`
instead of a flat message list, `function_result` input items instead of
`tool_result` content blocks). Tool schemas were refactored into a shared,
provider-agnostic module (`tool_schemas.py`) so both providers read from the same
source of truth instead of maintaining two copies of the prompt.

**#5 — Gemini 3.7 Flash blocked by persistent quota exhaustion; provider choice deferred**
Step 3's batch validation run hit a `429` quota error immediately, not after
volume. Confirmed it wasn't a transient per-minute limit by retrying with proper
backoff (58s/55s/54s/54s waits) and getting the same error every time across ~4
minutes of real wall-clock time. This is a persistent block (daily cap already
exhausted, zero free-tier allocation for this model, or billing not enabled on
the key's project) that no amount of client-side retry logic can fix — it needs
action on the Google AI Studio / Cloud Console side, outside this session's
control. Decided: defer picking a provider until later rather than switching back
to Claude as a stopgap. `agent.py` (Claude) and `agent_gemini.py` (Gemini) both
remain built and ready — whichever provider gets picked next, the tools, schema,
and evidence loop don't change, only the driver module.

**#6 — Baseline methodology: dev/test split + versioned eval log, human golden set as an additional benchmark tier**
Before running any live agent eval, built the measurement infrastructure so the
first successful run becomes a real "Agent v0" baseline, not just a one-off
number. Three pieces: (1) [scripts/split_dev_test.py](scripts/split_dev_test.py)
splits the 5,000-review sample into a fixed 1,000-review dev set (for iterating
the prompt) and a 4,000-review held-out test set (touched only once, at the end)
— prevents unknowingly overfitting the prompt to the eval data. (2)
[scripts/eval_runner.py](scripts/eval_runner.py) runs any agent version against
either set, computes precision/recall/F1 with Wilson-interval confidence bounds,
and appends a versioned row to `data/eval_runs.csv` plus a full per-review detail
CSV — so every prompt iteration is a comparable, logged entry, not a number that
scrolls past in a terminal. (3) [scripts/golden_set.py](scripts/golden_set.py)
gives an on-this-exact-dataset human benchmark: a class-balanced 40-review subset
of the sample, hand-labeled by the PM with the same tool evidence the agent sees,
reported with the same Wilson-interval methodology — directly comparable to the
agent's own results on request, and to the literature-based ~50-65% human-accuracy
benchmark from deceptive-review-detection research (different dataset, same task
type). All three benchmark tiers (majority-class floor, human/literature, agent)
are described together in this session's discussion of "how do we know what
counts as good" — worth keeping that framing when writing up Step 4/8's results.

**#7 — Manual in-chat reasoning as a provider-blocked stand-in for Step 7**
With the API provider still unresolved, mined the full dataset (pure pandas/tool
queries, no LLM needed — [scripts/step7_mine_candidates.py](scripts/step7_mine_candidates.py))
for real candidates in all three of Step 7's failure categories, then reasoned
through each one inline in conversation, following the exact system prompt and
output schema from `tool_schemas.py`. This is explicitly **not** the same as an
automated agent run — no API call, no logged `eval_runs.csv` entry, just this
session's model reasoning over the same tool evidence the real agent would see.
All three real cases matched ground truth, including the coordinated-campaign
case (review_id 5217) which is a genuine, real-data demonstration of why the
business-trend tool exists: text similarity alone reads that review as clean,
and only the 4.87x burst ratio catches it. Worth re-running these same three
`review_id`s through the actual automated agent once a provider is live, to
check whether the automated version reasons the same way — if it diverges,
that's a useful signal about how much the system prompt vs. the model itself
is doing the work.

**#8 — Autonomy policy: permissive auto-remove, mitigated by an appeals path**
This was explicitly your call, not mine, per Section 1's role split. You chose a
permissive auto-remove threshold (favoring throughput/scale) over a strict one,
despite the ground-truth caveat (Decision #5-adjacent) establishing that Yelp's
own label is known to be imperfect in both directions. I flagged this tension
explicitly rather than softening it — a permissive auto-remove tier means some
genuine reviews get removed with no human ever looking at them, on the strength
of agreement with an admittedly imperfect ground truth. You kept the mitigation
(an appeals path allowing post-removal human review) rather than letting the
tradeoff stand unmitigated. Full policy and monitoring plan are in
[CASE_STUDY.md](CASE_STUDY.md). Numeric thresholds remain uncalibrated — they
require Agent v0's actual eval results, still blocked on a provider.

**#9 — Completed all provider-independent work in one pass, on your explicit instruction**
You asked me to finish everything an LLM can do without a live API key, and flag what's
left. Completed: Section 2 (failure cases as short stories), a preliminary Section 4
(agent vs. single-shot classifier, n=3, same review_ids as Step 7 — reasoned blind, no
tools, to isolate what text-only reasoning misses), and Section 5 (the Uber transfer
argument, including an honest note that the autonomy *threshold* — not just the tool
mapping — needs to shift toward caution for a domain where a false positive costs someone
their livelihood, not just a removed review). One explicit boundary respected: the golden
set labeling was NOT touched — that's the human benchmark, and my doing it would collapse
the entire point of having an independent human data point to compare the agent against.

**#10 — v1 closed at current scope; live-provider work explicitly deferred to Phase 2**
You decided to close out this version of the project rather than keep it open-ended
waiting on a provider. Everything achievable without a live API connection is finished:
full design (tools, evidence loop, policy, monitoring plan), the ground-truth caveat backed
by a real partial golden-set number (56% accuracy at n=16, matching the literature's
50-65% range), three real failure cases, a preliminary classifier comparison, and the
cross-domain transfer argument. Reframed [checklist.md](checklist.md) accordingly — items
needing a live provider are now labeled "Phase 2," not "blocked," since blocked implies
this version can't be considered done, when in fact its scope was always "design +
partial validation," with statistically-solid numbers always slated for a later phase.
Golden-set labeling (16/40) stays open at your own pace — not part of v1's finish line,
since new labels remain useful data whenever they're added.

**#11 — Groq unblocks Phase 2; generic OpenAI-compatible driver added for provider flexibility**
You supplied a Groq API key. Verified it worked with a single raw test call before
building anything (`openai/gpt-oss-120b` — the largest general-purpose model on this
key's account). Rather than writing a Groq-specific driver, built one generic
OpenAI-compatible driver (`scripts/agent_openai_compat.py`) parameterized by
`scripts/provider_config.py`, since Groq and OpenRouter both expose the same API shape —
switching between them (or adding a new one) is a one-line env var change, not a rewrite.
Wired into `eval_runner.py` alongside the existing Gemini/Claude options.

Immediately re-ran all 3 of Step 7's manually-reasoned failure cases through this real,
live agent — and it's an important, humbling result, reported honestly rather than
smoothed over: **1 out of 3 correct**, not 3 out of 3 like the manual walkthrough. It
matched on the coordinated-campaign case (review_id 5217) with the same reasoning
(business burst + singleton account outweighing low text similarity). It missed the
genuine-burst case (review_id 14) — didn't weigh the rating-variance signal that should
have prevented a false positive. And on the signal-disagreement case (review_id 418812),
its stated reasoning claimed the account was "brand-new," directly contradicting the tool
data it had just received (8 other reviews) — a reasoning-accuracy problem, not just a
threshold-calibration one. This is real, valuable Step 4-style signal that a prompt
iteration is needed, discovered before the full eval run rather than buried inside it.

**#12 — 50-review manual batch (Claude-reasoned): an important, humbling result reported
honestly, not smoothed over**
Per your direction, ran a 50-review batch through manual Claude reasoning (blind, same
discipline as the golden set) instead of the full 1,000-review Groq run, as a middle
ground between "no data" and "the full statistical eval." Built `scripts/manual_batch.py`
to draw the batch, dump blind evidence, and score against ground truth afterward.

**The result is a real finding, not a flattering one.** Headline accuracy was 84% —
sounds fine in isolation. But the batch was 6 fake / 44 genuine (matching the real ~13%
rate), and a system that always guesses "genuine" scores 88% on that same batch — higher
than the careful reasoning, while doing zero work. Checked precision/recall specifically:
**0% recall on the fraud class** — every one of the 6 actually-filtered reviews got missed,
and the 2 times "filtered" was guessed, both were wrong. This is the exact accuracy-is-a-
misleading-metric trap the case study already warned about in the abstract (Section 4's
majority-class-floor point) — now demonstrated concretely, on real reasoning, not just
argued theoretically.

Diagnosed the likely cause: the reasoning consistently wanted a business-side review spike
to corroborate a thin/singleton account before calling something fraud (the exact pattern
that correctly caught the Section 2 coordinated-campaign case) — but several real fraud
cases in this batch were thin accounts with no business-side spike at all, and got a pass
because that corroboration was missing. This is a specific, correctable finding — not a
vague "needs more work" — and it's now documented in [CASE_STUDY.md](CASE_STUDY.md) §6
exactly as it happened, including the comparison against the trivial baseline. Combined
with the earlier 3-case Groq spot-check (which had a different failure pattern — reasoning
that didn't match its own evidence), we now have two independent, honestly-reported
failure signatures to fix in the first real prompt iteration once the full automated run
happens.

**#13 — Interactive triage mockup: closing the PM's interface-intuition gap**
You gave direct, important feedback: everything built so far lived in markdown tables,
CSVs, and terminal output — for a PM whose mental model forms through interfaces, that
meant no felt intuition for how the product actually works, only an analytical
understanding. Built `mockup/index.html` — a real, interactive triage tool, not a static
diagram — using the actual data already generated in this project (the 50-review manual
batch with real tool evidence and judgments, the 16 labeled golden-set reviews), not
placeholder content. Three things it makes visible that no document could:

1. **Today vs. proposed, same review, one click apart** — toggling from "manual" (just
   the review text — what an analyst sees before doing any digging) to "agent-assisted"
   (the same review with all evidence and the judgment laid out) makes the actual value
   proposition felt, not described.
2. **The autonomy policy operating on real data, not just described in prose** — routing
   each of the 50 reviews through the illustrative policy surfaced, live, that the *only*
   review that would be auto-removed is a false positive — the exact risk Section 1 of the
   case study names in the abstract, now visible as a real, clickable example.
3. **The human-calibration mode** shows your own golden-set reasoning text next to the
   evidence and the reveal — the same 3-way comparison (human / agent / ground truth)
   that's been tracked in CSVs all along, now something to click through rather than read
   as a table.

Caught and fixed a real bug during build/test in the browser before publishing: no
`<meta charset="utf-8">` caused the "◆" brand mark to render as mojibake — fixed and
re-verified, including a dark-mode check, before publishing as an artifact.

**#14 — Controls panel: every PM-controllable knob inventoried, then made real (not dummy) in the dashboard**
Inventoried every parameter a PM could tune across the whole system — prompts, provider/model,
autonomy thresholds, tool behavior, eval/benchmark config, operational settings — organized
by category rather than answering only the two or three examples given. Then, per your explicit
"no dummy data" instruction, extended the dashboard honestly rather than faking interactivity:

- **Made genuinely live, on real data:** a "signals allowed to auto-remove" checkbox filter,
  combined with the existing confidence slider, recomputes real routing across the real
  50-review batch. Tested it directly: unchecking `combination` moved the batch's one
  false-positive auto-remove case to the queue tier and eliminated the risk entirely — a real,
  useful finding (the original plan's more conservative default policy, requiring `combination`
  specifically, would have prevented this exact error), not a canned demo.
- **Shown as real reference, not fake interactivity:** system prompt, tool descriptions,
  provider/model options, and numeric parameters (max turns, temperature — flagged as unset,
  business-trend window, similarity pool size, dev/test split, golden-set size) are pulled
  verbatim from the actual source files into a new "System Config" view. Explicitly labeled as
  non-editable here, since editing them meaningfully requires a live agent connection — chose
  honesty about that limitation over building a slider that looks functional but does nothing.

Verified via full page-text extraction (not just screenshots) that every displayed value
matches its source file exactly before publishing.

**#15 — Pivoted from the Claude Artifact sandbox to a real Streamlit dashboard, solving
both hard walls found in Decision #14/the sample-capability investigation**
The Artifact's `sample` capability turned out to be Claude-only with no temperature
control — real, load-bearing platform limits, not implementation gaps. Rather than build
around them, evaluated two alternatives: Lovable (real backend, but requires porting the
Python agent to TypeScript and a handoff I can't execute myself) and the project's own
`zero-cost-agent-toolkit` skill (Streamlit + the existing Python stack + Render + MLflow).
Chose the toolkit: it runs `scripts/tools.py`, `scripts/tool_schemas.py`, and
`scripts/data_index.py` directly, with zero rewrite, and I could build and test it myself
in this session rather than handing off to a platform I can't drive.

Built `dashboard/` — a real Streamlit app, not a mockup:
- `live_agent.py`: one unified, temperature-aware, 4-provider (Groq/OpenRouter/
  Gemini/Claude) agentic loop, reusing the real tool dispatch logic, yielding
  step-by-step trace events for live UI rendering.
- `config_store.py`: JSON-backed config version history with a real deploy/rollback
  flow and auto-filled diff-based reasons.
- `mlflow_tracking.py`: every judgment (human or agent) logged as an MLflow run,
  tagged by config version, with real precision/recall/confidence-interval
  aggregation — seeded with this project's actual historical results (the golden-set
  human labels, the 50-review Claude-manual batch) so comparisons aren't empty on
  first launch.
- `app.py`: 5 pages (Overview, Human Review, Agent Review, System Config, Performance
  Tracking), wired end-to-end.

Verified live in a real browser, not just written blind — caught and fixed two real
bugs in the process: a stale pickle cache that failed to unpickle under the current
pandas version (rebuilt fresh), and Streamlit's `use_container_width` deprecation
warnings (updated to `width="stretch"`). Ran a real live Groq investigation end-to-end
(4 turns, 3 real tool calls, a coherent judgment, correct routing) and a real config
deploy (temperature 0.2 → 0.65, auto-filled reason, new version `v1` immediately active).

Honest scope note: this runs as a local server (`streamlit run dashboard/app.py`), not
a public URL — the toolkit's Render option is there if a shareable link is ever wanted,
but standing that up requires the user's own GitHub↔Render OAuth connection, which can't
be scripted headlessly.

---

## 5. What a Real Production Build Would Add On Top of This

This project is scoped as a portfolio case study, not a shipped product — worth
naming the gap honestly rather than pretending this is launch-ready:

- **No red-teaming pass** on the system prompt (e.g., can a spammer craft review
  text that talks the agent out of flagging it?)
- **No prompt-injection defense** — reviewer text is passed directly into the
  agent's context; a malicious review could contain text aimed at the agent itself,
  not just at human readers
- **No versioning/rollback plan** for the prompt — a real system would track prompt
  versions against eval scores over time, not just "the current prompt"
- **No legal/privacy review** — this uses a public academic dataset, but a real
  reviewer-investigation system touches PII and would need that sign-off before
  any of this ships
- **No on-call/monitoring hookup** — Step 6's "drift monitoring" is a paragraph in
  a case study here, not a running dashboard

None of this blocks the case study's goal, but it's the honest answer to "is this
how you'd actually ship it" — worth having ready if asked in an interview.
