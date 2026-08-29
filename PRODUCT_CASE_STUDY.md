# Review Trust Agent — Product Case Study

**An agentic AI system for Yelp review-fraud triage — designed, built, evaluated, and
shipped as a live, self-serve dashboard.**

Live demo: [review-trust-agent.streamlit.app](https://review-trust-agent.streamlit.app/) ·
Code: [github.com/PrathamNawal/review-trust-agent](https://github.com/PrathamNawal/review-trust-agent)

This document is written for someone meeting this project cold — a hiring manager, a
peer PM, a reviewer — who wants to understand not just what was built, but the product
thinking behind it: the problem, the bet, the tradeoffs, and what was learned along the
way, including the parts that didn't work as hoped.

---

## 1. Business Problem

Any company running a review or ratings marketplace has the same structural problem:
**reviews are the trust signal the whole marketplace runs on, and that signal is
gameable.** Yelp filters roughly 13% of all submitted reviews as suspected fake or
manipulated, using its own internal, largely undisclosed algorithm. That filter is
consequential in both directions — a fake review that slips through misleads real
consumers and unfairly advantages the business behind it; a genuine review wrongly
filtered silences a real customer and can materially hurt a small business's visibility.

Today, that decision is made by an opaque, single-shot classifier with no visible
reasoning trail. When a business or reviewer disputes a removal, there's no evidence
trail explaining *why* — just a label. That's the concrete gap this project addresses:
**can an AI system make the same kind of call a trust & safety analyst would, while
producing an auditable reasoning trail and a policy that's explicit about how much
autonomy it's been given?**

## 2. Impact (Hypothetical — illustrative assumptions, not real Yelp data)

Yelp does not publish its trust & safety operational metrics, so the numbers below are
**illustrative, clearly-labeled assumptions**, meant to show the shape of the business
case a real team would build, not a claim about Yelp's actual economics.

| Assumption | Illustrative value |
|---|---|
| Reviews submitted per day (order-of-magnitude, public estimates) | ~500,000 |
| Share flagged as ambiguous / appealed (assumption) | 2% → ~10,000/day |
| Fully-loaded analyst cost | $35/hour, ~40 reviews/hour reviewed manually |
| Manual review cost for the ambiguous slice | ~$8,750/day → **~$3.2M/year** |

**The bet this project tests:** if a confidence-gated agent can *correctly* auto-clear
the confidently-genuine share and auto-flag the confidently-fake share of that ambiguous
queue — leaving only the genuinely uncertain cases for a human — even a conservative
30% reduction in human-reviewed volume is worth **~$960K/year** in this illustrative
model, before counting the harder-to-quantify value of faster resolution and a visible
reasoning trail for appeals. **The honest finding in Section 9 is that this project's
current agent is not yet ready to safely capture that value** — its recall on real fraud
was 0% in the one real batch tested — which is exactly the kind of result this
evaluation approach exists to surface before a costly production rollout, not after.

## 3. Hypothesis and Strategic Fit

**Hypothesis:** if an agent gathers the same categories of evidence a human analyst
would — reviewer posting history, business-level review velocity, and text-pattern
similarity — and reasons across them in a multi-turn loop rather than a single classifier
pass, it can produce decisions that are both more accurate *and* more explainable than an
opaque single-shot filter, at a confidence level precise enough to gate real autonomy.

**Strategic fit:** trust & safety is not a side feature for a review marketplace, it's
the product — a marketplace where either businesses or consumers stop trusting the
review signal has no product left. An agentic approach fits particularly well here
because the underlying task (weigh several independent, sometimes-conflicting signals,
know when you're unsure) is naturally suited to tool use and multi-step reasoning, not a
single forward pass over review text.

## 4. Users, Jobs-to-be-Done, and Success Outcomes

**Primary user: Trust & Safety analyst.**
JTBD: *"When a review is flagged as potentially fake, I want a consistent, evidenced
read on it fast, so I can act with confidence instead of re-deriving context from
scratch on every case."*
Success outcome: less time per case, a visible reasoning trail to point to on appeal,
fewer cases where the analyst and the tool silently disagree without either knowing why.

**Secondary user: Trust & Safety PM / policy owner.**
JTBD: *"When I need to change how cautious the system is, I want to adjust it and see
the real before/after impact, so I'm not shipping a policy change blind."*
Success outcome: this is the dashboard's core loop — edit a config, deploy it, see
accuracy/precision/recall move, roll back if it doesn't.

**Tertiary user: the reviewer/business whose content was actioned.**
JTBD: *"If my review gets removed, I want to understand why and have a real path to
contest it."*
Success outcome: an auditable reasoning trace exists per decision (see Section 7);
Section 8's autonomy policy explicitly builds in an appeals path rather than treating
"automatic" as "unreviewable."

## 5. Scope, Non-Goals, and Tradeoffs

**In scope:**
- Single-review investigation using three evidence sources (reviewer history, business
  trend, text similarity)
- A pre-registered, three-tier autonomy policy (auto-remove / queue / ignore)
- A config-versioning and before/after evaluation system (the dashboard)
- Multi-provider support (Groq, OpenRouter, Gemini, Claude) so the agent isn't locked to
  one vendor

**Explicitly out of scope (non-goals):**
- Real-time production integration into Yelp's actual review pipeline
- Multi-language review support (the dataset and prompt are English-only)
- Image/photo-based fraud signals
- Cross-account network/graph analysis (detecting *rings* of coordinated fake accounts,
  as opposed to one account's own pattern)
- Fine-tuning or training a custom model — this project deliberately evaluates
  off-the-shelf LLM reasoning against a simpler classifier baseline (Section 6 of the
  original case study) rather than assuming a bespoke model is the answer

**Tradeoffs made deliberately, not by default:**
- Chose a **permissive** auto-remove threshold, accepting some wrongful removals of
  genuine reviews in exchange for handling volume — mitigated by an appeals path, not by
  pretending the risk doesn't exist.
- Chose to evaluate against Yelp's own filter label as ground truth, while stating
  plainly that this label is itself an imperfect, undisclosed algorithm — not a gold
  standard (Section 3 of the original case study, and Section 9 below).
- Chose proportional (13.22% filtered) sampling over an artificially balanced 50/50
  sample, because precision/recall on a balanced sample would overstate real-world
  performance (Decision Log #1).

## 6. Data & Grounding

The agent is grounded in the **YelpZip dataset** (Rayana & Akoglu, KDD 2015) — 608,598
real Yelp reviews with Yelp's own filtered/recommended label attached to each one. This
is not synthetic or LLM-generated data anywhere in the evaluation pipeline. Two verified,
independent evidence sources were used before trusting the dataset (a blocked official
source and an unverified single-file mirror were both rejected first — Decision Log #2).

Each of the three tools grounds its answer in the **full 608K-review history**, not just
whatever subset happens to be under evaluation, because a reviewer's or business's real
behavioral history lives outside any evaluation sample:
- **Reviewer history** — posting frequency, rating variance, burst behavior across a
  reviewer's *entire* history.
- **Business trend** — review-velocity spikes in a ±14-day window versus the business's
  all-time baseline rate.
- **Text similarity** — character n-gram TF-IDF cosine similarity against up to 300 other
  real reviews by the same reviewer or business (catches lightly-reworded templates that
  word-level comparison would miss).

Every tool degrades honestly on missing data — a reviewer with no other history returns
an explicit "singleton reviewer" signal (itself a known spam indicator) rather than a
silent zero or a crash.

## 7. Sample Prompts

The system prompt (verbatim, from `scripts/tool_schemas.py`):

> *"You are a trust & safety investigator checking whether a Yelp review is likely
> fake/manipulated (would be filtered) or genuine (would be recommended). You have three
> evidence-gathering tools plus a submit_judgment tool to end the investigation. You do
> not have to call all three tools, and you are not limited to calling each one once. If
> the evidence so far is ambiguous, pull more of it before deciding... Only call
> submit_judgment when you're confident you've gathered enough evidence to justify your
> answer. If the signals genuinely conflict, your confidence should reflect that — don't
> manufacture false certainty."*

A real worked trace (the "smart fake campaign" case — see the full case study for all
three): the review text alone reads as an entirely normal, specific, casually-written
review of a dish. The agent calls `reviewer_history_lookup` and finds a singleton
account (no other reviews, ever); it then calls `business_trend_lookup` and finds a
review-velocity spike of nearly 5x normal in the surrounding four weeks. Neither signal
is visible in the review's own text — only the investigation surfaces them, and together
they correctly flag a coordinated campaign that a text-only classifier missed entirely.

## 8. Quality, Safety, and Policy Guardrails

- **Confidence-gated autonomy, not binary automation.** The three-tier policy
  (auto-remove / queue / ignore) means low-confidence cases always fall back to a human
  by design, not by exception-handling. Verified live in the dashboard: dropping the
  confidence threshold below a case's actual confidence visibly reroutes it to the human
  queue.
- **Appeals path as the safety net for the permissive tier.** "Automatic" means a human
  isn't required *before* action, not that one is locked out *after* it.
- **No-dummy-data as an engineering discipline, not just a demo nicety.** Every number,
  chart, and control across the dashboard is wired to a real computation — when a
  genuine limitation was hit (an earlier platform's LLM sandbox had no temperature
  control), the response was to rebuild on infrastructure where it could be made real,
  not to fake the control (Decision Log #14–15).
- **Bring-your-own-credentials in the public deployment.** Any visitor trying the live
  agent uses their own API key, entered client-side and never persisted — the project
  owner's key and quota are never exposed to public traffic.
- **A real, caught reasoning-faithfulness issue.** In one of the three hand-picked hard
  cases, the live agent's *stated* reasoning ("brand-new account") directly contradicted
  its own tool output (8 other reviews existed for that account) — a genuine unfaithful-
  reasoning failure mode, documented rather than smoothed over, and directly relevant to
  trusting any auto-remove decision made on the agent's stated rationale.

## 9. Evaluation Plan & Live Playground

The evaluation methodology and the interactive playground are the same live artifact —
the [Streamlit dashboard](https://review-trust-agent.streamlit.app/) itself, structured
as a guided six-step flow:

1. **What is this? / How it works** — the problem and mechanism, in plain language.
2. **Try it — Human** — anyone can act as the analyst on a real review with real
   evidence, scored blind against Yelp's real label. This is also the human benchmark:
   16 reviews labeled so far, **56.2% accuracy** — landing squarely in the 50–65% range
   reported in independent published research on this exact task, which is itself
   evidence the task is genuinely hard, not that the labeler was careless.
3. **Try it — Agent** — the real agent investigates live (any of 4 providers), full
   tool-call trace visible, on the same item a visitor just judged as a human.
4. **Play** — edit the system prompt, temperature, provider, or threshold and deploy a
   new version with one click; quick-experiment presets pre-fill common changes.
5. **Track Performance** — every judgment, human or agent, logged to MLflow and grouped
   by config version, with Wilson-score confidence intervals (appropriate at the small-n
   a live dashboard session actually produces) and a plain-language before/after delta.

**The headline real result, reported exactly as found:** a 50-review manual batch
(reasoning done to the same discipline as the automated agent would apply) scored **84%
accuracy** — but **0% recall** on the 6 actually-fake reviews in that batch, missing every
one, while also producing 2 false positives. On this class-imbalanced task (~13% fraud
rate), a trivial "always guess genuine" baseline scores 88% accuracy — *higher* than the
careful, evidence-weighing approach — while doing zero actual work. This is the central,
humbling finding of the project, not a footnote: **high accuracy without recall on the
class that matters is a failure, not a success**, and it's exactly what a proper eval
(rather than an accuracy-only sanity check) exists to catch before a costly rollout.

**What's not yet done, stated plainly:** a full statistically powered run (hundreds of
reviews, held-out test set, confidence intervals on precision/recall specifically) has
not been executed — the infrastructure for it (`scripts/eval_runner.py`, a fixed
dev/test split) is built and ready, gated only on sustained API access to run it at
volume.

## 10. Risks and Tradeoffs (Top 3)

1. **Recall on real fraud is the unproven, load-bearing risk.** The one real batch
   result showed 0% recall on actual fraud despite a headline-looking 84% accuracy. Until
   a full-scale run confirms this is fixable (the likely root cause — over-requiring a
   business-side spike to corroborate a thin account, per the failure analysis — is
   concrete and actionable, not vague), **this system is not ready to be trusted with any
   real autonomy**, regardless of how the dashboard's autonomy policy is configured.

2. **The ground truth itself has a ceiling on how much it can validate.** Every accuracy
   number in this project measures agreement with Yelp's own undisclosed filter — not
   objective fraud-detection accuracy, which nobody can measure directly without a better
   answer key. A human labeler working the same evidence scored only 56%, which bounds
   how much confidence any single-digit accuracy improvement should be given.

3. **Reasoning can be unfaithful to its own evidence.** The live agent's stated rationale
   contradicted its own tool output in at least one documented case. An agent that sounds
   confident and coherent while citing evidence it doesn't actually have is a distinct
   and arguably more dangerous failure mode than simply being wrong — it would pass a
   naive "does the reasoning look plausible" spot-check while still being incorrect for
   the wrong reasons, which is exactly the kind of risk an auditable, run-logged trace
   (rather than a black-box label) is meant to surface.

---

## Appendix A: Three Real Cases, Three Different Failure Modes

Not made-up examples — three real reviews from the dataset, each chosen to stress a
different way a fraud-detector can go wrong.

**The reviewer who posted seven reviews in one sitting.** Looks like a flood-and-run fake
account on posting frequency alone. But the ratings varied naturally (not uniformly
5-star, the tell of a paid campaign) and the wording differed across all seven — most
likely someone catching up on write-ups after several real visits. Yelp agreed: not
flagged. **Lesson:** timing alone is an easy false alarm; the other signals are what keep
the agent from flagging a real person just for *when* they posted.

**The fake campaign that was smart enough to not sound fake.** Read on its own, this
review is indistinguishable from a genuine one — specific, casual, no template tells. A
text-only classifier would clear it instantly (verified directly — see the classifier
comparison this project ran). What the text can't show: a singleton account with no other
history, posted during a ~5x spike in the business's review velocity. Neither fact lives
in the words. Together they correctly caught a coordinated campaign a text-only approach
missed entirely — the single clearest justification for investigating instead of just
reading.

**The case where the evidence disagreed with itself.** A posting burst (suspicious) with
no corroborating business-side spike, no rating-uniformity, no copy-paste (all pointing
the other way). Yelp did not flag it. The point isn't that the answer was right — it's
that the right response to conflicting evidence is honest uncertainty ("probably fine, not
sure"), not a confidently blended guess in either direction. An agent that manufactures
certainty when signals genuinely disagree is worse even when it happens to land correctly.

---

*The complete build decision log — every architectural and product decision made during
the build, in ADR-lite format — is in this repo at `PROCESS.md`, for anyone who wants to
dig into the engineering process.*
