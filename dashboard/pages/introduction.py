"""Overview and explanatory pages for first-time dashboard visitors."""

from pathlib import Path

import streamlit as st

from dashboard.mechanism import PLAY_PAGE, build_stage_dot, build_stages, describe_mechanism


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
    '**How to read the result:** Choice is an experimental '
    'binary decision, not a calibrated fraud probability. On a fresh evaluation stopped at '
    '**961 of 3,000** reviews, it reached **9.2% recall** and **84.2% accuracy**. '
    'The sample was incomplete and the target (≥20% recall, ≥85% accuracy) was not met. '
    'YelpZip labels are reference labels, not verified fraud ground truth.'
)

JEV_DETAILS = (
    "Evidence is restricted to reviews dated before the case under investigation. "
    "The PM can omit evidence sources for controlled ablation experiments. "
    "The public dashboard uses a trimmed YelpZip index. The fresh paired comparison "
    "tested Choice alongside two Noul policies; see the "
    "[full experiment report](../experiments/jev-recall-experiment-report.md)."
)

AGNO_RESULT_NOTE = (
    '**How to read the result:** The Agno agent is the original '
    'dashboard agent. On a 50-review batch with 6 filtered reviews it reached **0% recall** and '
    '**84.0% accuracy**: it missed every filtered example. Its confidence is self-reported by the '
    'model, not calibrated. YelpZip labels are reference labels, not verified fraud ground truth.'
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
STAGE_KIND = {"evidence": "evidence · built by code", "tool": "tools · chosen by the model"}
PAGE_WIDTH = 1160  # readable measure: keeps rows, tiles and the diagram from stretching on wide screens


def _go_to_stage(key, stage_id):
    st.session_state[key] = stage_id


def _render_rows(rows):
    for key, value in rows:
        label_col, value_col = st.columns([1, 4], gap="small")
        label_col.caption(key)
        value_col.markdown(value)


def _render_tunables(tunables, note):
    if not tunables:
        st.caption(f":material/lock: {note}")
        return
    st.caption(":material/tune: Tunable on Play")
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
        st.markdown(dict(step["rows"]).get("Asks", step["why"]))
        extra = [(k, v) for k, v in step["rows"] if k != "Asks"]
        if extra:
            with st.expander("What it returns", icon=":material/data_object:"):
                _render_rows(extra)


def _render_stage(stage, goto, number, next_stage, key):
    first = stage["steps"][0]
    grouped = bool(stage["cluster"])
    wants_play = any(step["play"] for step in stage["steps"])
    with st.container(border=True):
        head, action = st.columns([3, 1], vertical_alignment="center")
        with head:
            st.badge(STAGE_KIND[stage["kind"]] if grouped else first["kind_name"],
                     color=BADGE_COLORS[stage["kind"]])
            st.subheader(f"{number}. " + (stage["label"].split(" · ")[0] if grouped else first["title"]))
        if wants_play and action.button("Tune on Play", icon=":material/tune:", width="stretch",
                                        key=f"hiw_play_{stage['id']}"):
            goto(PLAY_PAGE)
        st.markdown(f"*{first['why']}*")
        if grouped:
            for pair in range(0, len(stage["steps"]), 2):
                for column, step in zip(st.columns(2, gap="small"), stage["steps"][pair:pair + 2]):
                    with column:
                        _render_source_card(step)
            if any(step["tunables"] for step in stage["steps"]):
                st.caption(":material/tune: Switch each source on or off on Play.")
        else:
            _render_rows(first["rows"])
            _render_tunables(first["tunables"], first["tunable_note"])
        if next_stage:
            with st.container(horizontal=True, horizontal_alignment="right"):
                st.button(f"Next: {next_stage['label'].split(' · ')[0]}", icon=":material/arrow_forward:",
                          key=f"hiw_next_{stage['id']}", on_click=_go_to_stage, args=(key, next_stage["id"]))


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

    with st.container(horizontal_alignment="center"):
        with st.container(width=PAGE_WIDTH):
            _render_page(mech, goto, active_config)


def _render_page(mech, goto, active_config):
    st.markdown(
        f'<div class="rta-page-eyebrow">THE REVIEW FLOW · ACTIVE: {mech["name"].upper()} · '
        f'{active_config.get("version", "v0")}</div>',
        unsafe_allow_html=True,
    )
    st.title("How it works")
    st.markdown(mech["intro"])

    stages = build_stages(mech)
    by_id = {stage["id"]: stage for stage in stages}
    default_stage = next(stage["id"] for stage in stages
                         if any(step["id"] == mech["default_step"] for step in stage["steps"]))
    key = f"hiw_stage_{mech['architecture']}"
    if st.session_state.get(key) not in by_id:
        st.session_state[key] = default_stage
    labels = {stage["id"]: f"{n}. {stage['label']}" for n, stage in enumerate(stages, 1)}

    hero = st.container(border=True)
    with hero:
        diagram_slot = st.container()
        chosen = st.segmented_control(
            "Stage", options=list(by_id), format_func=labels.get, key=key,
            label_visibility="collapsed", width="stretch",
        ) or default_stage
        st.caption("Select a stage to see what happens there.")
    with diagram_slot:
        st.graphviz_chart(build_stage_dot(mech, stages, chosen), width="stretch")

    position = list(by_id).index(chosen)
    following = stages[position + 1] if position + 1 < len(stages) else None
    _render_stage(by_id[chosen], goto, position + 1, following, key)

    state_col, memory_col = st.columns(2, gap="medium")
    for column, card in ((state_col, mech["state_card"]), (memory_col, mech["memory_card"])):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {card['title']}")
            st.markdown(card["body"])
            st.caption(card["source"])

    is_jev = mech["architecture"] == "jev"
    st.info(JEV_RESULT_NOTE if is_jev else AGNO_RESULT_NOTE, icon=":material/query_stats:")
    with st.expander("Evidence and evaluation details", icon=":material/info:"):
        st.markdown(JEV_DETAILS if is_jev else AGNO_DETAILS)

    st.caption("Want to see this run on a real review? That's the next two steps — you judge one, "
               "then see it done on the same review. The stages above are built from the active config "
               "each time the page loads, so they can't drift from what's deployed.")
    if st.button("Try it yourself →", type="primary"):
        goto("3. Try it — Human")
