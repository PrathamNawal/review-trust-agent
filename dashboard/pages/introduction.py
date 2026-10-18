"""Overview and explanatory pages for first-time dashboard visitors."""

import streamlit as st

from dashboard.ui_components import render_hero_stat


def render_overview(goto):
    st.title("Review Trust Agent")
    st.markdown("#### An AI agent that decides which Yelp reviews are fake — and a dashboard that lets you second-guess it.")

    st.markdown(
        "Yelp filters roughly **13% of all submitted reviews** as suspected fake or manipulated, "
        "using its own (imperfect, undisclosed) algorithm. This project builds an alternative: an "
        "AI agent that investigates each review like a trust & safety analyst — pulling real evidence "
        "before it judges — plus a dashboard that lets *you* run the same investigation, tweak the "
        "agent's behavior, and see the tradeoffs live."
    )

    c1, c2, c3 = st.columns(3)
    render_hero_stat(c1, "A trained human's accuracy", "56%",
                     "16-review golden-set benchmark, blind labeling — matches published literature (50-65%).")
    render_hero_stat(c2, "Original agent accuracy", "84%",
                     "50-review project batch; see the original recall result below.", tone="warning")
    render_hero_stat(c3, "Choice recall · interim", "9.2%",
                     "11/119 filtered labels found in 961 completed cases from a fresh 3,000-review holdout.",
                     tone="warning")

    st.info(
        "These are results from different evaluations. The original dashboard agent had 0% "
        "recall on six positives in its 50-review batch. The completed 3,000-review Jev "
        "comparison found 7.8% recall / 83.2% accuracy for the decomposed Noul policy. The "
        "current Choice method reached 9.2% recall / 84.2% accuracy on a separate fresh "
        "evaluation stopped at 961 of 3,000 reviews. That partial result is below the 20% / "
        "85% targets; Choice is experimental and every decision goes to a human."
    )
    st.caption(
        "Not sure where to start? Use the **\"I am a...\"** picker in the sidebar to jump "
        "straight to the flow built for your role — you can switch it anytime."
    )
    st.divider()
    st.subheader("What you can do here")
    st.markdown("""
1. **See how it works** — the evidence the agent gathers before it decides
2. **Try it as a human** — judge a real review blind, see if you'd have caught it
3. **Try it as the agent** — watch it investigate live, tool call by tool call
4. **Work the Analyst Queue** — review the agent's already-investigated cases: accept, override, or escalate
5. **Check a review's status** — as the reviewer/business, see why a decision was made and appeal it
6. **Play** — review the experimental Jev Choice method, switch to Agno, or compare settings
7. **Track performance** — see exactly how your changes moved accuracy, precision, and recall
""")

    if st.button("See how it works →", type="primary"):
        goto("2. How it works")


def render_how_it_works(goto):
    st.title("How it works")
    st.markdown(
        "Before the agent judges anything, it gathers real evidence — the same evidence a human "
        "trust & safety analyst would pull. Every number below comes from the actual 608K-review "
        "Yelp dataset, not a simulation."
    )
    st.markdown("#### The investigation loop")
    st.markdown("""
```
  Review comes in
        │
        ▼
  Agent picks a tool ──────► Reviewer history · Business trend · Text similarity
        │                     (it can call these zero or more times, in any order,
        │                      and re-check evidence if it's still unsure)
        ▼
  Enough evidence?  ──No──►  gather more
        │ Yes
        ▼
  Jev Choice: filtered or recommended + evidence trail
        │
        ▼
  Jev Choice result ─────────────────────► 🟡 human review required
```
""")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🕵️ Reviewer history**")
        st.caption(
            "Does this reviewer's posting pattern look normal? Checks review count, rating "
            "spread, and the most reviews they've ever posted in one 7-day window. A reviewer "
            "with only *this one* review is itself a known spam signal."
        )
    with col2:
        st.markdown("**📈 Business trend**")
        st.caption(
            "Was there a review *spike* at this business around this date? Compares the "
            "review rate in a ±14-day window to the business's all-time average. A burst 3-5x "
            "above normal is the classic signature of a paid campaign."
        )
    with col3:
        st.markdown("**🔎 Text similarity**")
        st.caption(
            "Does the wording look templated or copy-pasted? Compares this review's text against "
            "up to 300 others by the same reviewer or business, using character-level pattern "
            "matching (so it still catches lightly-reworded templates)."
        )
    st.divider()
    st.caption("Full technical reference: reviewer_history_lookup, business_trend_lookup, and "
               "text_similarity_check are implemented in scripts/tools.py.")

    if st.button("Try it yourself →", type="primary"):
        goto("3. Try it — Human")
