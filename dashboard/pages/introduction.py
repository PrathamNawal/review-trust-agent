"""Overview and explanatory pages for first-time dashboard visitors."""

from pathlib import Path

import streamlit as st

from dashboard.mechanism import PLAY_PAGE, build_dot, build_stages, describe_mechanism


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


BADGE_COLORS = {
    "input": "gray", "evidence": "green", "tool": "green", "model": "blue",
    "output": "violet", "terminal": "violet", "rule": "orange", "human": "green",
}


def _go_to_stage(key, stage_id):
    st.session_state[key] = stage_id


def _render_rows(rows):
    for key, value in rows:
        label_col, value_col = st.columns([1, 3], gap="small")
        label_col.caption(key)
        value_col.markdown(value)


def _render_tunables(tunables, note):
    if not tunables:
        st.caption(f":material/lock: {note}")
        return
    st.caption(":material/tune: Tunable here")
    with st.container(horizontal=True):
        for key, value in tunables:
            with st.container(border=True):
                st.caption(key)
                st.markdown(f"**{value}**")


def _render_source_card(step):
    switchable = bool(step["tunables"]) and step["tunables"][0][0] == "Include this source"
    with st.container(border=True, height="stretch"):
        badge = ""
        if switchable:
            badge = " :gray-badge[off]" if step["off"] else " :green-badge[on]"
        st.markdown(f"**{step['label']}**{badge}")
        details = dict(step["rows"])
        st.markdown(details.get("Asks", step["why"]))
        extra = [(k, v) for k, v in step["rows"] if k != "Asks"]
        if extra:
            with st.expander("What it returns", icon=":material/data_object:"):
                _render_rows(extra)


def _render_stage(stage, goto, key):
    first = stage["steps"][0]
    with st.container(border=True):
        st.badge(first["kind_name"] if not stage["cluster"] else STAGE_KIND[stage["kind"]],
                 color=BADGE_COLORS[stage["kind"]])
        if not stage["cluster"]:
            st.subheader(first["title"])
            st.markdown(f"*{first['why']}*")
            _render_rows(first["rows"])
            _render_tunables(first["tunables"], first["tunable_note"])
        else:
            st.subheader(stage["label"].split(" · ")[0])
            st.markdown(f"*{first['why']}*")
            for pair in range(0, len(stage["steps"]), 2):
                columns = st.columns(2, gap="small")
                for column, step in zip(columns, stage["steps"][pair:pair + 2]):
                    with column:
                        _render_source_card(step)
        wants_play = any(step["play"] for step in stage["steps"])
        with st.container(horizontal=True, horizontal_alignment="right"):
            if wants_play and st.button("Tune on Play", icon=":material/tune:", key=f"hiw_play_{stage['id']}"):
                goto(PLAY_PAGE)


STAGE_KIND = {"evidence": "evidence · built by code", "tool": "tools · chosen by the model"}


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

    stages = build_stages(mech)
    by_id = {stage["id"]: stage for stage in stages}
    labels = {stage["id"]: f"{n}. {stage['label']}" for n, stage in enumerate(stages, 1)}
    default_stage = next(stage["id"] for stage in stages
                         if any(step["id"] == mech["default_step"] for step in stage["steps"]))
    key = f"hiw_stage_{mech['architecture']}"
    if st.session_state.get(key) not in by_id:
        st.session_state[key] = default_stage

    st.caption("Walk the flow — pick a stage to see what really happens there.")
    chosen = st.segmented_control(
        "Stage", options=list(by_id), format_func=labels.get, key=key,
        label_visibility="collapsed", width="stretch",
    ) or default_stage
    _render_stage(by_id[chosen], goto, chosen)

    position = list(by_id).index(chosen)
    if position + 1 < len(stages):
        following = stages[position + 1]
        st.button(f"Next: {following['label'].split(' · ')[0]}", icon=":material/arrow_forward:",
                  key=f"hiw_next_{chosen}", on_click=_go_to_stage, args=(key, following["id"]))

    state_col, memory_col = st.columns(2, gap="medium")
    for column, card in ((state_col, mech["state_card"]), (memory_col, mech["memory_card"])):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {card['title']}")
            st.markdown(card["body"])
            st.caption(card["source"])

    is_jev = mech["architecture"] == "jev"
    st.markdown(JEV_RESULT_NOTE if is_jev else AGNO_RESULT_NOTE, unsafe_allow_html=True)
    with st.expander("Full flow diagram", icon=":material/account_tree:"):
        selected = by_id[chosen]["steps"][0]["id"] if not by_id[chosen]["cluster"] else None
        st.graphviz_chart(build_dot(mech, selected), width="content")
    with st.expander("Evidence and evaluation details", icon=":material/info:"):
        st.markdown(JEV_DETAILS if is_jev else AGNO_DETAILS)

    st.caption("Want to see this run on a real review? That's the next two steps — you judge one, "
               "then see it done on the same review. The stages above are built from the active config "
               "each time the page loads, so they can't drift from what's deployed.")
    if st.button("Try it yourself →", type="primary"):
        goto("3. Try it — Human")
