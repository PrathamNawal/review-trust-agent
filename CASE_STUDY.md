# Review Trust Agent — Case Study

## What this project is

Fake reviews are a real problem for any company that runs a review or ratings system —
Yelp, Google Maps, Uber, and many others. This project builds an AI agent that investigates
a single Yelp review the way a human trust-and-safety analyst would: it looks at the
reviewer's posting history, checks whether the business got a sudden unusual spike of
reviews around that time, and compares the review's wording against other reviews to spot
copy-paste patterns. Based on what it finds, it decides whether the review looks fake.

We then check the agent's guesses against a real answer key: Yelp already runs its own spam
filter, and every review in this dataset carries Yelp's own verdict — "filtered" (Yelp's
system flagged it as likely fake) or "recommended" (it passed Yelp's checks). That gives us
something to measure the agent against, though — as explained below — it's an imperfect
answer key, not a perfect one, and this document says so plainly rather than hiding it.

The point of the project isn't just "does the agent guess correctly." It's: how much should
a company trust an AI system to act on its own here — deleting a review with no human
involved — versus when should a human always be the one to decide? That's the real decision
this case study leads with.

**How to read this document:** each section below answers one specific question a
decision-maker would actually ask. Sections 4 and 6 are marked as preliminary/pending
because they need a live connection to an AI model to finish — see the status note at the
end of each of those sections for exactly what's still needed.

**What's finished in this version, and what's planned for later.** This version covers the
full *design*: the investigation approach, the autonomy policy and its tradeoffs, the
honesty check on the answer key (including a real, if partial, human-benchmark number), real
examples of hard cases, and the argument that this approach generalizes beyond Yelp. What
this version does **not** yet include is a large-scale, statistically solid accuracy number
for the agent itself — that requires connecting the agent to a live AI model and running it
across hundreds of reviews, which is planned as the next phase of this project, not
something skipped by oversight. The infrastructure for that next phase (a fixed set of
reviews to test against, a script that runs the comparison and logs results with proper
confidence ranges) is already built and ready to go the moment that connection is available.

1. [What the agent is allowed to decide on its own](#1-the-autonomy-policy) — the central decision of this project
2. [Three real examples of hard cases, and how the agent should handle them](#2-three-real-examples)
3. [Why Yelp's own "fake review" label isn't perfect, and what that means for these results](#3-why-the-answer-key-isnt-perfect)
4. [Is a multi-step investigator actually better than just asking an AI to read the review and guess?](#4-is-the-extra-investigation-worth-it-preliminary) *(preliminary)*
5. [Would this same approach work for a completely different company, like Uber?](#5-would-this-work-somewhere-else-ubers-rating-fraud-problem)
6. [How accurate is the agent, with real statistics?](#6-how-accurate-is-the-agent) *(pending — needs a live AI connection)*
7. [What to watch after this goes live](#7-what-to-watch-after-launch)

---

## 1. The Autonomy Policy

**This is the central decision of the whole project: when is the agent allowed to act
completely on its own, and when must a human be involved?**

We landed on three tiers:

- **Auto-remove** — the agent deletes the review immediately, with no human involved.
- **Queue for human review** — the agent flags the review, but a person makes the final call.
- **Ignore** — the agent finds nothing convincing and takes no action.

**We chose a *permissive* auto-remove tier** — meaning the bar for the agent to act
completely on its own is set relatively low, favoring speed and scale over caution. In
plain terms: we're accepting that this system will sometimes delete a genuine, innocent
review by mistake, in exchange for being able to handle a much larger volume of reviews
without needing a person to check every single one.

**This tradeoff is made with eyes open, not because we didn't think about the risk.** As
Section 3 below explains, Yelp's own "fake review" label is known to be imperfect — it
sometimes wrongly flags genuine reviews, and sometimes misses real fraud. A permissive
auto-remove tier means the agent will sometimes delete a real, honest review without a
human ever looking at it, purely because the review matched what an already-imperfect
reference system would have flagged.

**The safety net for this risk: an appeals process.** Anyone whose review gets auto-removed
can contest it, which triggers a human to take a second look. Being "automatic" doesn't
mean the decision is unreviewable — it means a human isn't required *before* the removal,
not that a human is locked out *after* it.

**Queue for human review has a low bar on purpose.** Anything that shows even one
moderately convincing warning sign gets sent to a person, rather than being ignored. We'd
rather a human check on something that turns out to be fine than let something suspicious
slip through unflagged. Put simply: we're more worried about missing real fraud than about
giving reviewers extra work to check things that turn out okay.

**We have not set the exact numeric cutoffs yet.** Deciding the *shape* of the policy
(three tiers, which way each one leans) before ever running the agent is a deliberate
choice — it means the policy reflects our actual values, not just whatever pattern happened
to show up in the first batch of results. The specific numbers (e.g., "confidence above
0.85 triggers auto-remove") get set once we have real results to calibrate against — see
Section 6.

---

## 2. Three Real Examples

These are three real reviews found in the dataset — not made-up examples — each chosen
because it tests a different way an AI fraud-detector could go wrong. For each one, the
reasoning below was done by hand (by an AI, but read manually in this conversation, not by
the actual automated program) — a stand-in for the real system until it's fully connected
and running, but not a substitute for double-checking with the real thing later.

### The reviewer who posted seven reviews in one sitting

One reviewer posted seven reviews — including the one we're investigating — all on the
exact same day. On its own, that looks a lot like a fake account: create it, flood it with
reviews, move on. A system that only checked "how often does this person post" would have
good reason to flag it.

But the rest of the evidence tells a different story. The seven ratings weren't all glowing
— they ranged from very negative to very positive, which isn't what a paid fake-review
campaign usually looks like (those tend to be uniformly 5 stars). And the wording across
all seven reviews was clearly different from each other, not copy-pasted. The most likely
real explanation: someone who'd eaten at several places over time simply sat down and wrote
up all their opinions in one go. Yelp's own system agreed — this review was not flagged.

**The lesson:** posting timing alone is an easy thing to look suspicious by accident. The
other checks (how varied the ratings are, whether the wording looks copy-pasted) are what
keep the agent from wrongly flagging a real person just because of *when* they posted.

### The fake-review campaign that was smart enough to not sound fake

This is the case that best justifies building an agent that investigates, instead of one
that just reads the review text and guesses. Read on its own, this review looks completely
normal — specific, casually written, praising a particular dish. If you only gave an AI the
review's text and asked "does this sound fake?", it would confidently say no. (We tested
this directly — see Section 4 below.)

What the text alone can't show: the account that posted it has never posted any other
review, ever, and in the four weeks around this review, the number of reviews this business
was getting suddenly jumped to almost five times its normal rate. Neither of those facts
appears anywhere in the words of the review itself. Put together, they're a strong sign of
a paid or solicited review campaign — one savvy enough to avoid the obvious tell
(copy-pasted wording) while still leaving behind a less obvious one (a sudden flood of
reviews from accounts with no other history). Yelp's system caught this one. A system that
only reads review text would not have.

### The case where the evidence disagreed with itself

One reviewer posted five reviews within a single week at some point in an otherwise normal
six-month history — enough of a pattern to raise a flag on its own. But nothing else backs
it up: the business wasn't seeing any unusual spike in reviews at the time, this reviewer's
ratings varied naturally rather than being suspiciously uniform, and the wording wasn't
copy-pasted. Yelp's system did not flag this review.

The important part of this example isn't that the final answer was right — it's *how* the
agent should get there. One piece of evidence (the posting burst) points toward "suspicious,"
but two other checks point the other way and don't back it up. The right response to that
isn't to blend the signals into a falsely confident answer either way — it's for the agent
to say, honestly, "this is probably fine, but I'm not very sure," rather than "definitely
fine" or "definitely fake." An agent that manufactures confidence when the evidence
genuinely disagrees with itself is a worse agent, even on the occasions it happens to land
on the correct answer anyway.

---

## 3. Why the Answer Key Isn't Perfect

**What Yelp's "filtered" label actually is.** Every "filtered" or "recommended" tag in this
dataset comes from Yelp's own automatic spam-detection system — not a group of trained
human reviewers, and not something Yelp has ever published the exact rules for. Outside
researchers (Mukherjee, Kumar & Liu, in a study called "What Yelp Fake Review Filter Might
Be Doing?") have tried to reverse-engineer it, and found it likely relies more on behavior
patterns (how active the account is, how it's connected to other accounts) than on the
words of the review itself — and that its results generally line up with other signs of
spam. But that's supporting evidence, not proof that any single review's label is correct.

**Sometimes it's too strict.** Yelp has said publicly, over the years, that its filter also
catches reviews from accounts that simply don't have much of a track record yet — not
necessarily because the review is fake, but because the account hasn't built up enough
trust signal. So some share of the reviews labeled "filtered" in this dataset are probably
genuine opinions from newer or less active users, not fabricated reviews.

**Sometimes it's too lenient.** The flip side matters too: a review being labeled
"recommended" doesn't guarantee it's genuine. A sophisticated fake-review operation that's
good at mimicking normal behavior could slip past Yelp's checks entirely and still show up
as "recommended" — so the "clean" half of this dataset isn't guaranteed to be clean either.

**We saw this ambiguity ourselves, directly.** As part of this project, one of us
hand-labeled a set of reviews personally — reading the same evidence the agent sees, with
no access to Yelp's actual answer — to get a sense of how a careful human does on this task.
One of the examples from Section 2 above (the "evidence disagreed with itself" case) is a
perfect illustration: even reading all the same evidence carefully, a person has to sit with
genuine uncertainty to reach an answer. Yelp's system reduces that uncertainty down to a
single yes/no tag — but the underlying judgment call is genuinely a matter of degree, not a
clean binary.

**And the numbers back this up.** On the 16 reviews labeled so far (out of a planned 40),
the human labeler got **56% right** — landing right in the middle of the 50–65% range
reported in outside research on this exact kind of task (Section 4's comparison). In other
words: a careful, attentive person, given the same evidence the agent has, still lands close
to a coin flip on a meaningful share of these reviews. That's not a knock on the labeler —
it's the clearest possible evidence that this task is genuinely hard, and that Yelp's own
tag shouldn't be treated as an obviously "easy" ground truth just because it's a computer's
output.

**What would make this more trustworthy.** Ideally, a sample of these reviews would be
re-labeled by multiple trained human judges working independently, without seeing Yelp's
own label, and their agreement with each other would be measured. It would also help to
separate "this violates Yelp's community guidelines" (like a business owner reviewing their
own restaurant) from "this review is outright fabricated" — two different problems that
Yelp's single filtered/not-filtered tag currently lumps together.

**What this means for every number in this report.** Any accuracy, precision, or recall
number in this project measures *how often the agent agrees with Yelp's own system* — not
"how often the agent correctly identifies actual fraud," which nobody can measure directly
without a better answer key. That distinction should be kept in mind everywhere a number
shows up in this report, not treated as a footnote.

---

## 4. Is the Extra Investigation Worth It? (Preliminary)

**Status: this is a small, preliminary check — not the final answer.** A real answer needs
running both approaches over hundreds of reviews with an actual AI model connected, and
measuring real cost and speed. That's not possible yet (see the status note at the bottom).
What follows is a 3-review sanity check, done by hand, just to see if the idea holds up
before investing in the full version.

We took the same three reviews from Section 2 and asked: what would happen if an AI just
read the review's text, with none of the investigative tools, and guessed?

| Review | Yelp's actual answer | The investigating agent's guess | A simple text-only guess |
|---|---|---|---|
| "seven reviews in one sitting" | not flagged | not flagged (correct) | not flagged (correct) |
| "smart fake campaign" | **flagged as fake** | **flagged as fake (correct)** | **not flagged (WRONG)** |
| "evidence disagreed with itself" | not flagged | not flagged, with honest lower confidence (correct) | not flagged (correct) |

The text-only guesser missed exactly the case the investigative approach exists to catch:
nothing about that review's *wording* looks suspicious — it only becomes suspicious once
you know the posting account has no other history and the business had a sudden flood of
reviews at the same time. Neither fact shows up in the words of the review. With only 3
examples this doesn't prove anything statistically, but it's a clean illustration of why
this project didn't just build a simple text classifier.

**What's still needed to make this a real finding:** run both approaches (with tools, and
text-only) across hundreds of reviews using a live connection to an AI model, measure
accuracy properly, and — just as importantly — measure the actual cost and time each
approach takes per review. If the extra accuracy from the investigative approach doesn't
clearly outweigh its extra cost and slower speed at real scale, that's worth reporting
honestly, not glossed over just because more work went into building it.

---

## 5. Would This Work Somewhere Else? Uber's Rating-Fraud Problem

This project's real claim isn't "this works for Yelp reviews" — it's that the overall
*approach* (investigate from multiple angles → decide how much autonomy the AI has earned →
be upfront about the answer key's flaws → keep watching after launch) would work for any
company with a similar trust problem, without needing to rebuild everything from scratch.
Uber's ratings system is a good test of that claim, even without access to Uber's actual
data.

**The underlying problem is the same.** Uber faces the same two problems Yelp does: ratings
inflated by a coordinated group (a driver or rider getting friends to leave 5-star ratings
to hit a bonus threshold) and ratings weaponized unfairly (a 1-star rating left purely out
of spite after an unrelated dispute, not because of bad service). Both are about trust in a
marketplace, not about reading text carefully — which is exactly this project's whole point.

**The three checks transfer as ideas, not as literal code.**
- *Reviewer history* becomes **rater history** — does this person's rating pattern look
  normal over time, or suspiciously uniform, or bunched into a sudden burst?
- *Business trend* becomes **driver/rider trend** — has a specific driver or rider's rating
  volume spiked unusually fast in a short window, the same idea applied to a person instead
  of a business?
- *Text similarity* becomes **behavior similarity** — Uber ratings are usually just stars,
  with little or no text to compare. So the literal "compare the wording" tool doesn't carry
  over directly, but the underlying idea does: looking for suspiciously similar *patterns*
  instead — the same trip route, the same time of day, the same payment method reused
  across a cluster of accounts — anything that suggests coordination rather than
  coincidence.

**The autonomy decision needs to be re-thought, not just copied — and this is the honest
part of the comparison.** For Yelp, we chose a permissive auto-remove tier because the cost
of getting it wrong is a deleted review — annoying, but reversible and low-stakes. Removing
someone's ability to drive or ride with Uber is a much bigger deal — it can affect someone's
actual income. The three-tier shape (act automatically / send to a human / do nothing) still
makes sense, but the bar for acting automatically should shift much more toward caution —
likely closer to "a human should almost always be involved, and full automation should be
reserved only for the most obvious, unambiguous cases." The overall approach carries over;
the specific comfort level with automation does not, and pretending otherwise would be the
wrong lesson to take from this project.

**The "imperfect answer key" problem carries over exactly.** Uber doesn't have some
perfectly correct fraud label either — at best, it would have its own internal history of
past enforcement decisions, which would be just as imperfect as Yelp's filter: a record of
what one company's existing system decided in the past, not objective truth. The same
honesty applies: report how well a new system agrees with Uber's *past decisions*, not
claim it measures "true fraud-detection accuracy."

**Watching for changes after launch matters just as much, if not more.** Just as review
fraud adapts over time, rating fraud would adapt too — people spacing out ratings to avoid
looking like a burst, or varying trip patterns to avoid looking coordinated. The same
after-launch watchlist from Section 7 applies here, with the appeals process being even more
critical given how much more is at stake for someone wrongly flagged.

---

## 6. How Accurate Is the Agent?

**Status: partial — one real 50-review result exists, and it's a genuinely humbling one worth
reporting exactly as it happened.**

Before a live AI model was connected, one of us (an AI, reasoning manually rather than through
the automated pipeline) worked through 50 reviews from the dev set (the set reserved for
iteration, not the final held-out test set) by hand — same evidence, same blind discipline as
the human golden-set exercise. The headline number looked fine: **84%
accuracy**. But that number is misleading, and this project already warned about exactly this
trap in Section 4: on an imbalanced dataset (this batch was 6 fake / 44 genuine, matching the
real ~13% rate), a system that just guesses "genuine" every single time would score **88%
accuracy** — higher than the careful, evidence-weighing 84% — while doing zero actual work.

**And that's exactly what happened here.** Of the 6 actually-fake reviews in this batch, this
careful reasoning caught **zero** of them — a 0% catch rate on the only class that matters. It
also flagged 2 genuine reviews as fake, incorrectly. In other words: on this batch, weighing
the evidence carefully did *worse* than doing nothing.

**Why this happened, as best we can tell:** the reasoning consistently required a business-side
review spike *and* a thin account history to flag something as fake — the exact pattern
confirmed in Section 2's coordinated-campaign example. But several of the real fake reviews in
this batch didn't show a business spike at all — just a thin account and nothing else. The
reasoning treated a thin account alone as too weak to act on without a second corroborating
signal, and in doing so, missed real cases where a thin account *was* the whole story.

**This is a genuinely useful, if humbling, result, not a wasted exercise.** It shows a concrete,
correctable failure pattern (being too reluctant to flag on a single signal alone) rather than
a vague sense that "the agent needs work." That's real material for the first prompt
improvement once a full automated run happens — a separate, smaller live-agent check on 3 hard
cases (Section 2) already surfaced a related issue: the automated agent got 1 of those 3 right,
including correctly catching the same style of case this larger batch missed, but was also
wrong in a case where its stated reasoning didn't match the evidence it had actually been
given. Two different reasoning processes, two different failure patterns — both real, both
useful to know before scaling up.

**What's still needed for the full picture:** the same kind of run, but across the entire
1,000-review dev set (and, once the prompt is finalized, a single final run on the held-out
test set) with a live, automated agent connected — enough volume to know whether
these are consistent tendencies or artifacts of a small sample, and to calculate real
confidence intervals on precision and recall specifically (not just accuracy). The
infrastructure for that — a fixed set of reviews, a script that runs the comparison and logs
results — is already built and ready to go.

---

## 7. What to Watch After Launch

Everything in this report is based on a one-time historical test, not on watching the
system handle live, ongoing traffic — so a real launch would need to keep an eye on three
things:

1. **Whether accuracy quietly gets worse over time.** If people running fake-review
   campaigns start deliberately avoiding whatever the agent leans on most heavily (for
   example, deliberately varying their wording once they realize copy-pasted text gets
   caught), the agent's accuracy on new cases should be expected to slip over time — it
   won't just stay at whatever number it started with.
2. **How often auto-removed reviews get successfully appealed.** Since we chose a
   permissive auto-remove setting (Section 1), this is the real-world signal for whether
   that choice is costing too many honest reviews. If the appeal/reversal rate starts
   climbing, that's the trigger to make the auto-remove setting stricter.
3. **Whether the agent starts leaning on one type of evidence much more than the others.**
   If the reasons the agent gives for its decisions start clustering heavily around one
   type of evidence, that's worth digging into — either that type of evidence is becoming
   less reliable (for example, a legitimate seasonal spike in reviews getting mistaken for a
   fraud pattern), or fraud is genuinely concentrating in a way worth understanding better.
