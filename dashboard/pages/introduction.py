"""Overview and explanatory pages for first-time dashboard visitors."""

import streamlit as st

from dashboard.ui_components import render_hero_stat, render_process_step


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
    st.markdown('<div class="rta-page-eyebrow">THE REVIEW FLOW · EXPERIMENTAL CHOICE</div>',
                unsafe_allow_html=True)
    st.title("How it works")
    st.markdown(
        "A review is paired with behavioral evidence, then Jev Choice returns an advisory label. "
        "A human makes the final call."
    )
    st.markdown("### Four steps, one human decision")
    flow = st.columns(4, gap="medium")
    render_process_step(
        flow[0], "01 · Review", "Read the case",
        "The review text, rating, and date define the case being assessed.",
    )
    render_process_step(
        flow[1], "02 · Evidence", "Build context",
        "The app prepares reviewer history, business activity, and text similarity using earlier reviews.",
    )
    render_process_step(
        flow[2], "03 · Jev Choice", "Return a label",
        "Jev selects <b>Filtered</b> or <b>Recommended</b> from the review and evidence bundle.",
    )
    render_process_step(
        flow[3], "04 · Human", "Review the result",
        "Every Jev label is advisory. A person reviews it; the app takes no automatic moderation action.",
    )

    st.markdown("### Evidence shown to Jev")
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🕵️ Reviewer activity")
            st.write("Posting volume, timing gaps, rating spread, and activity bursts across earlier reviews.")
            st.markdown('<div class="rta-evidence-meta">Signal source · reviewer history</div>',
                        unsafe_allow_html=True)
    with col2:
        with st.container(border=True):
            st.markdown("#### 📈 Business activity")
            st.write("Review volume near the case date compared with that business's earlier baseline.")
            st.markdown('<div class="rta-evidence-meta">Signal source · business trend</div>',
                        unsafe_allow_html=True)
    with col3:
        with st.container(border=True):
            st.markdown("#### 🔎 Reused wording")
            st.write("Text similarity against earlier reviews by the same reviewer and for the same business.")
            st.markdown('<div class="rta-evidence-meta">Signal source · text similarity</div>',
                        unsafe_allow_html=True)

    st.markdown(
        '<div class="rta-method-note"><b>How to read the result:</b> Choice is an experimental '
        'binary decision, not a calibrated fraud probability. On a fresh evaluation stopped at '
        '<b>961 of 3,000</b> reviews, it reached <b>9.2% recall</b> and <b>84.2% accuracy</b>. '
        'The sample was incomplete and the target (≥20% recall, ≥85% accuracy) was not met. '
        'YelpZip labels are reference labels, not verified fraud ground truth.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Evidence and evaluation details"):
        st.markdown(
            "Evidence is restricted to reviews dated before the case under investigation. "
            "The public dashboard uses a trimmed YelpZip index. The fresh paired comparison "
            "tested Choice alongside two Noul policies; see the "
            "[full experiment report](../experiments/jev-recall-experiment-report.md)."
        )

    if st.button("Try it yourself →", type="primary"):
        goto("3. Try it — Human")
