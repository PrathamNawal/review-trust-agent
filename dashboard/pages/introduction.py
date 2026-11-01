"""Overview and explanatory pages for first-time dashboard visitors."""

from pathlib import Path

import streamlit as st

from dashboard.mechanism import PLAY_PAGE, build_dot, describe_mechanism


def render_overview(goto):
    image_path = Path(__file__).resolve().parents[1] / "assets" / "review-trust-comparison.jpg"
    story, illustration = st.columns([0.88, 1.12], gap="large", vertical_alignment="center")
    with story:
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
        if st.button("See how the investigation works →", type="primary"):
            goto("2. How it works")

    with illustration.container(border=True):
        st.image(
            str(image_path),
            width="stretch",
            caption=(
                "Without context → with evidence for a human to assess. The human makes the final call."
            ),
        )


JEV_RESULT_NOTE = (
    '<div class="rta-method-note"><b>How to read the result:</b> Choice is an experimental '
    'binary decision, not a calibrated fraud probability. On a fresh evaluation stopped at '
    '<b>961 of 3,000</b> reviews, it reached <b>9.2% recall</b> and <b>84.2% accuracy</b>. '
    'The sample was incomplete and the target (≥20% recall, ≥85% accuracy) was not met. '
    'YelpZip labels are reference labels, not verified fraud ground truth.</div>'
)

JEV_DETAILS = (
    "Evidence is restricted to reviews dated before the case under investigation. "
    "The PM can omit evidence sources for controlled ablation experiments. "
    "The public dashboard uses a trimmed YelpZip index. The fresh paired comparison "
    "tested Choice alongside two Noul policies; see the "
    "[full experiment report](../experiments/jev-recall-experiment-report.md)."
)

AGNO_RESULT_NOTE = (
    '<div class="rta-method-note"><b>How to read the result:</b> The Agno agent is the original '
    'dashboard agent. On a 50-review batch with 6 filtered reviews it reached <b>0% recall</b> and '
    '<b>84.0% accuracy</b>: it missed every filtered example. Its confidence is self-reported by the '
    'model, not calibrated. YelpZip labels are reference labels, not verified fraud ground truth.</div>'
)

AGNO_DETAILS = (
    "The Agno agent's evidence tools exclude only the review being judged, so they can include "
    "reviews dated after it; the Jev architecture uses only earlier reviews. "
    "The public dashboard uses a trimmed YelpZip index. The PM can switch architecture on Play."
)


def _step_label(step):
    return step["label"] + (" · off" if step["off"] else "")


def render_how_it_works(goto, active_config):
    """Explain the active architecture, read from the deployed config and the real code."""
    try:
        mech = describe_mechanism(active_config)
    except ValueError as exc:
        st.title("How it works")
        st.info(f"{exc} A diagram for it hasn't been added to dashboard/mechanism.py yet.")
        if st.button("Try it yourself →", type="primary"):
            goto("3. Try it — Human")
        return

    st.markdown(
        f'<div class="rta-page-eyebrow">THE REVIEW FLOW · ACTIVE: {mech["name"].upper()} · '
        f'{active_config.get("version", "v0")}</div>',
        unsafe_allow_html=True,
    )
    st.title("How it works")
    st.markdown(mech["intro"])

    steps = {step["id"]: step for step in mech["steps"]}
    diagram_col, inspector_col = st.columns([1.05, 1], gap="large")

    with inspector_col:
        selected = st.pills(
            "Inspect a step", options=list(steps), format_func=lambda i: _step_label(steps[i]),
            selection_mode="single", default=mech["default_step"],
            key=f"hiw_step_{mech['architecture']}",
        ) or mech["default_step"]
        step = steps[selected]
        with st.container(border=True):
            st.markdown(f"**{step['title']}** &nbsp; `{step['kind_name']}`")
            st.markdown(f"*{step['why']}*")
            for key, value in step["rows"]:
                st.markdown(f"**{key}:** {value}")
            if step["tunables"]:
                st.markdown("**Tunable here**")
                for key, value in step["tunables"]:
                    st.markdown(f"- {key}: `{value}`")
            else:
                st.caption(step["tunable_note"])
            if step["play"] and st.button("Tune on Play →", key=f"hiw_play_{selected}"):
                goto(PLAY_PAGE)

    with diagram_col:
        st.graphviz_chart(build_dot(mech, selected), width="stretch")

    state_col, memory_col = st.columns(2, gap="medium")
    for column, card in ((state_col, mech["state_card"]), (memory_col, mech["memory_card"])):
        with column.container(border=True):
            st.markdown(f"#### {card['title']}")
            st.markdown(card["body"])
            st.caption(card["source"])

    is_jev = mech["architecture"] == "jev"
    st.markdown(JEV_RESULT_NOTE if is_jev else AGNO_RESULT_NOTE, unsafe_allow_html=True)
    with st.expander("Evidence and evaluation details"):
        st.markdown(JEV_DETAILS if is_jev else AGNO_DETAILS)

    st.caption("Want to see this run on a real review? That's the next two steps — you judge one, "
               "then see it done on the same review. This diagram is built from the active config "
               "each time the page loads, so it can't drift from what's deployed.")
    if st.button("Try it yourself →", type="primary"):
        goto("3. Try it — Human")
