"""Overview and explanatory pages for first-time dashboard visitors."""

from pathlib import Path

import streamlit as st

from dashboard.ui_components import render_process_step


def render_overview(goto):
    st.markdown(
        '<div class="rta-page-eyebrow">WHY REVIEW CONTEXT MATTERS</div>',
        unsafe_allow_html=True,
    )
    st.title("One review rarely tells the whole story.")
    st.markdown(
        "A rating and comment can look ordinary on their own. Patterns in reviewer activity, "
        "business activity, or repeated wording can add context worth investigating — but they "
        "are signals, not proof of fraud."
    )

    image_path = Path(__file__).resolve().parents[1] / "assets" / "review-trust-comparison.jpg"
    with st.container(border=True):
        st.image(
            str(image_path),
            width="stretch",
            caption=(
                "Without context: a review stands alone. With Review Trust Agent: evidence is "
                "gathered for a person to assess; the human makes the final decision."
            ),
        )

    without, with_agent = st.columns(2, gap="large")
    with without:
        st.markdown("**WITHOUT REVIEW TRUST**")
        st.caption("A reviewer sees one post at a time, with little visibility into surrounding patterns.")
    with with_agent:
        st.markdown("**WITH REVIEW TRUST AGENT**")
        st.caption("The agent brings relevant patterns together. A human reviews its advisory label.")

    if st.button("See how the investigation works →", type="primary"):
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
