"""
Review Trust Agent — PM Dashboard

Real Python, real data, real (optional) live LLM calls. Run with:
  streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard import config_store, mlflow_tracking  # noqa: E402
from dashboard.live_agent import get_toolbox, PROVIDER_PRESETS  # noqa: E402
import dashboard.jev_agent  # noqa: E402,F401 -- registers the Jev architecture
from dashboard.jev_agent import get_typesafe_api_key  # noqa: E402
from dashboard.agent_service import investigate_and_log  # noqa: E402
from dashboard.ui_components import (
    inject_css,
)  # noqa: E402
from dashboard.pages.introduction import render_overview, render_how_it_works  # noqa: E402
from dashboard.pages.configuration import render_configuration  # noqa: E402
from dashboard.pages.performance import render_performance  # noqa: E402
from dashboard.pages.review import render_human_review, render_agent_review  # noqa: E402
from dashboard.pages.workflows import render_analyst_queue, render_check_my_review  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

st.set_page_config(page_title="Review Trust Agent — PM Dashboard", layout="wide")

NAV_OPTIONS = [
    "1. What is this?",
    "2. How it works",
    "3. Try it — Human",
    "4. Try it — Agent",
    "5. Analyst Queue",
    "6. Check My Review",
    "7. Play — Tweak the Agent",
    "8. Track Performance",
]

PERSONAS = {
    "analyst": {"label": "🕵️ Trust & Safety Analyst", "landing": "5. Analyst Queue",
                "desc": "Review the agent's queued cases fast, with a visible reasoning trail."},
    "pm": {"label": "🎛️ Policy Owner / PM", "landing": "7. Play — Tweak the Agent",
           "desc": "Tune the agent and see the real before/after impact."},
    "reviewer": {"label": "📝 Reviewer / Business", "landing": "6. Check My Review",
                 "desc": "See why a decision was made on your review — and contest it."},
    "explorer": {"label": "🔎 Just exploring", "landing": "1. What is this?",
                 "desc": "Take the guided, 7-step tour of the whole system."},
}

STEP_LABELS = ["What", "How", "Human", "Agent", "Queue", "Check", "Play", "Track"]
N_STEPS = len(STEP_LABELS)


def goto(page_name, **session_updates):
    st.session_state["nav_override"] = page_name
    for k, v in session_updates.items():
        st.session_state[k] = v
    st.rerun()


def mark_step_done(step_idx):
    st.session_state.setdefault("progress", {i: False for i in range(N_STEPS)})[step_idx] = True


def render_progress(current_idx):
    prog = st.session_state.setdefault("progress", {i: False for i in range(N_STEPS)})
    if current_idx in (0, 1, 7):  # passive pages count as done once viewed
        prog[current_idx] = True
    cols = st.columns(N_STEPS)
    for i, col in enumerate(cols):
        done = prog.get(i, False)
        is_current = (i == current_idx)
        icon = "✅" if done else ("▶️" if is_current else "⚪")
        text = f"{icon} {i + 1}. {STEP_LABELS[i]}"
        col.markdown(f"**{text}**" if is_current else text)
    st.progress(sum(prog.values()) / N_STEPS)
    st.write("")


@st.dialog("👋 Welcome to Review Trust Agent")
def welcome_dialog():
    inject_css()
    st.markdown(
        "This dashboard covers the full agentic fraud-detection system for Yelp reviews — "
        "built to be explored, not just read about. It plays out differently depending on "
        "who you are. Pick the closest fit:"
    )
    persona_items = list(PERSONAS.items())
    for row_start in (0, 2):
        cols = st.columns(2)
        for col, (key, p) in zip(cols, persona_items[row_start:row_start + 2]):
            with col.container(border=True):
                st.markdown(f"**{p['label']}**")
                st.caption(p["desc"])
                if st.button("Choose", key=f"persona_welcome_{key}",
                             type="primary" if key == "explorer" else "secondary", width="stretch"):
                    st.session_state["welcomed"] = True
                    st.session_state["persona"] = key
                    goto(p["landing"])


# ---------- one-time setup ----------
inject_css()
if "seeded" not in st.session_state:
    mlflow_tracking.seed_historical_baselines()
    st.session_state["seeded"] = True

if "nav_override" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("nav_override")

if "welcomed" not in st.session_state:
    welcome_dialog()


@st.cache_data
def load_sample():
    return pd.read_csv(os.path.join(DATA_DIR, "sample.csv"))


@st.cache_resource
def toolbox():
    return get_toolbox()


def judged_review_ids(judge_type, config_version):
    runs = mlflow_tracking.get_all_runs()
    if runs.empty:
        return set()
    mask = (runs.get("tags.judge_type") == judge_type) & (runs.get("tags.config_version") == config_version)
    if "tags.review_id" not in runs.columns:
        return set()
    return set(runs.loc[mask, "tags.review_id"].astype(int))


# ---------- sidebar: persona + nav + API keys ----------
st.sidebar.title("Review Trust Agent")

st.sidebar.caption("NAVIGATION")
persona_keys = list(PERSONAS.keys())
st.session_state.setdefault("persona", "explorer")
st.sidebar.selectbox(
    "I am a...", persona_keys, format_func=lambda k: PERSONAS[k]["label"], key="persona",
)
page = st.sidebar.radio("View", NAV_OPTIONS, key="nav")

st.sidebar.divider()
st.sidebar.caption("API KEYS")
with st.sidebar.expander("🔑 Add your keys", expanded=False):
    st.caption("Keys are entered for this running app and are not written to project files. Do not paste API keys into chat or source code.")
    for env_name, label in [("GROQ_API_KEY", "Groq"), ("OPENROUTER_API_KEY", "OpenRouter"),
                             ("GOOGLE_API_KEY", "Gemini"), ("ANTHROPIC_API_KEY", "Claude"),
                             ("TYPESAFE_API_KEY", "Jev Model (TypeSafe)")]:
        if env_name == "TYPESAFE_API_KEY":
            current = get_typesafe_api_key() or ""
        else:
            current = os.environ.get(env_name, "")
        entered = st.text_input(label, value=current, type="password", key=f"key_{env_name}")
        if entered:
            os.environ[env_name] = entered

active_config = config_store.get_active_config()
st.sidebar.divider()
st.sidebar.caption("STATUS")
st.sidebar.markdown(
    f"Active config: **{active_config['version']}** "
    f"({active_config.get('architecture', 'agno')} · {active_config['provider']}/{active_config['model']})"
)

render_progress(NAV_OPTIONS.index(page))


# ============================================================
# 1. WHAT IS THIS?
# ============================================================
if page == "1. What is this?":
    render_overview(goto)


# ============================================================
# 2. HOW IT WORKS
# ============================================================
elif page == "2. How it works":
    render_how_it_works(goto)


# ============================================================
# 3. TRY IT — HUMAN
# ============================================================
elif page == "3. Try it — Human":
    render_human_review(load_sample(), toolbox(), active_config, mlflow_tracking,
                        judged_review_ids, mark_step_done, goto)


# ============================================================
# 4. TRY IT — AGENT
# ============================================================
elif page == "4. Try it — Agent":
    render_agent_review(load_sample(), active_config, PROVIDER_PRESETS, judged_review_ids,
                        investigate_and_log, mark_step_done, goto)


# ============================================================
# 5. ANALYST QUEUE
# ============================================================
elif page == "5. Analyst Queue":
    render_analyst_queue(load_sample(), toolbox(), active_config, mlflow_tracking,
                         judged_review_ids, investigate_and_log, mark_step_done)


# ============================================================
# 6. CHECK MY REVIEW
# ============================================================
elif page == "6. Check My Review":
    render_check_my_review(load_sample(), active_config, mlflow_tracking,
                           investigate_and_log, mark_step_done, goto)


# ============================================================
# 7. PLAY — TWEAK THE AGENT
# ============================================================
elif page == "7. Play — Tweak the Agent":
    render_configuration(config_store, active_config, mark_step_done, goto)


# ============================================================
# 8. TRACK PERFORMANCE
# ============================================================
elif page == "8. Track Performance":
    render_performance(mlflow_tracking)
