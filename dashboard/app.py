"""
Review Trust Agent — PM Dashboard

Real Python, real data, real (optional) live LLM calls. Run with:
  streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import config_store  # noqa: E402
import mlflow_tracking  # noqa: E402
from live_agent import run_investigation_stream, get_toolbox, PROVIDER_PRESETS  # noqa: E402
from data_index import load_index  # noqa: E402

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
    "analyst": {"label": "🕵️ Trust & Safety Analyst", "landing": "5. Analyst Queue"},
    "pm": {"label": "🎛️ Policy Owner / PM", "landing": "7. Play — Tweak the Agent"},
    "reviewer": {"label": "📝 Reviewer / Business", "landing": "6. Check My Review"},
    "explorer": {"label": "🔎 Just exploring", "landing": "1. What is this?"},
}

QUICK_EXPERIMENTS = {
    "cautious": {
        "label": "🐢 Make it more cautious",
        "help": "Lower temperature, higher confidence bar before auto-removing.",
        "changes": {"temperature": 0.1, "confidence_threshold": 0.9},
    },
    "fast": {
        "label": "⚡ Make it faster & cheaper",
        "help": "Switch to a smaller, quicker Groq model.",
        "changes": {"provider": "groq", "model": "llama-3.1-8b-instant", "temperature": 0.2},
    },
    "claude": {
        "label": "🔀 Try Claude instead",
        "help": "Swap providers entirely — same prompt, different model.",
        "changes": {"provider": "claude", "model": "claude-3-5-haiku-latest", "temperature": 0.2},
    },
}


STEP_LABELS = ["What", "How", "Human", "Agent", "Queue", "Check", "Play", "Track"]
N_STEPS = len(STEP_LABELS)


def goto(page_name, **session_updates):
    st.session_state["nav_override"] = page_name
    for k, v in session_updates.items():
        st.session_state[k] = v
    st.rerun()


def compute_routing(predicted_filtered, confidence, threshold):
    if predicted_filtered and confidence >= threshold:
        return "auto_remove"
    elif not predicted_filtered and confidence >= threshold:
        return "ignore"
    return "queue"


ROUTING_LABELS = {"auto_remove": "🔴 Auto-remove", "queue": "🟡 Routed to human queue", "ignore": "🟢 Ignored"}


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
    st.markdown(
        "This dashboard covers the full agentic fraud-detection system for Yelp reviews — "
        "built to be explored, not just read about. It plays out differently depending on "
        "who you are. Pick the closest fit:"
    )
    for key, p in PERSONAS.items():
        if st.button(p["label"], key=f"persona_welcome_{key}", type="primary" if key == "explorer" else "secondary"):
            st.session_state["welcomed"] = True
            st.session_state["persona"] = key
            goto(p["landing"])


# ---------- one-time setup ----------
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

persona_keys = list(PERSONAS.keys())
st.session_state.setdefault("persona", "explorer")
st.sidebar.selectbox(
    "I am a...", persona_keys, format_func=lambda k: PERSONAS[k]["label"], key="persona",
)

page = st.sidebar.radio("View", NAV_OPTIONS, key="nav")

st.sidebar.divider()
st.sidebar.caption("API keys — only needed for live agent investigations (steps 4–6; session only, never written to disk)")
for env_name, label in [("GROQ_API_KEY", "Groq"), ("OPENROUTER_API_KEY", "OpenRouter"),
                         ("GOOGLE_API_KEY", "Gemini"), ("ANTHROPIC_API_KEY", "Claude")]:
    current = os.environ.get(env_name, "")
    entered = st.sidebar.text_input(label, value=current, type="password", key=f"key_{env_name}")
    if entered:
        os.environ[env_name] = entered

active_config = config_store.get_active_config()
st.sidebar.divider()
st.sidebar.caption(f"Active config: **{active_config['version']}** ({active_config['provider']}/{active_config['model']})")

render_progress(NAV_OPTIONS.index(page))


# ============================================================
# 1. WHAT IS THIS?
# ============================================================
if page == "1. What is this?":
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
    c1.metric("A trained human's accuracy", "56%", help="16-review golden-set benchmark, blind labeling. Matches published literature (50-65%).")
    c2.metric("The agent's accuracy", "84%", help="50-review batch. Sounds better — but read the next number.")
    c3.metric("...but its fraud recall", "0%", help="The agent missed every single actually-fake review in that batch. High accuracy with zero recall on the class that matters — the central finding of this project.")

    st.info(
        "That gap — high accuracy, zero recall on fraud — is the whole point of this dashboard. "
        "It's not a solved problem being demoed; it's an open one being made inspectable."
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
6. **Play** — edit the prompt, temperature, or model provider and deploy a new version
7. **Track performance** — see exactly how your changes moved accuracy, precision, and recall
""")

    if st.button("See how it works →", type="primary"):
        goto("2. How it works")


# ============================================================
# 2. HOW IT WORKS
# ============================================================
elif page == "2. How it works":
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
  Judgment: fake or genuine? + confidence + reasoning
        │
        ▼
  Routing ──► high confidence, fake      → 🔴 auto-remove
         ├──► high confidence, genuine   → 🟢 ignore
         └──► low confidence (either way) → 🟡 queue for a human
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


# ============================================================
# 3. TRY IT — HUMAN
# ============================================================
elif page == "3. Try it — Human":
    st.title("Try it — act as the analyst")
    st.caption("Real reviews, real evidence, blind judgment. Your answer is logged and scored against Yelp's real label.")

    sample = load_sample()
    tb = toolbox()

    already_judged = judged_review_ids("human", active_config["version"])
    pool = sample[~sample["review_id"].isin(already_judged)]

    if "human_review_id" not in st.session_state or st.session_state["human_review_id"] not in pool["review_id"].values:
        if not pool.empty:
            st.session_state["human_review_id"] = int(pool.sample(1, random_state=None)["review_id"].iloc[0])

    if pool.empty:
        st.success("You've judged every review in the sample under this config version!")
    else:
        col_pick, col_random = st.columns([3, 1])
        with col_pick:
            review_id = st.selectbox("Review", pool["review_id"].tolist(),
                                      index=pool["review_id"].tolist().index(st.session_state["human_review_id"])
                                      if st.session_state["human_review_id"] in pool["review_id"].values else 0)
        with col_random:
            if st.button("🎲 Random review"):
                st.session_state["human_review_id"] = int(pool.sample(1)["review_id"].iloc[0])
                st.rerun()

        row = sample[sample["review_id"] == review_id].iloc[0]
        user_id, prod_id = str(int(row["user_id"])), str(int(row["prod_id"]))

        st.markdown(f"**Rating:** {'⭐' * int(row['rating'])} &nbsp;&nbsp; **Date:** {row['date']}")
        st.info(row["review_text"])

        with st.expander("Evidence — reviewer history", expanded=True):
            st.json(tb.reviewer_history_lookup(user_id, exclude_review_id=int(review_id)))
        with st.expander("Evidence — business trend", expanded=True):
            st.json(tb.business_trend_lookup(prod_id, row["date"], exclude_review_id=int(review_id)))
        with st.expander("Evidence — text similarity vs. reviewer"):
            st.json(tb.text_similarity_check(row["review_text"], "reviewer", user_id, exclude_review_id=int(review_id)))
        with st.expander("Evidence — text similarity vs. business"):
            st.json(tb.text_similarity_check(row["review_text"], "business", prod_id, exclude_review_id=int(review_id)))

        st.divider()
        with st.form("human_judgment_form"):
            predicted = st.radio("Your judgment", ["Recommended (genuine)", "Filtered (fake)"], horizontal=True)
            confidence = st.slider("Confidence", 0.0, 1.0, 0.7, 0.05)
            primary_signal = st.selectbox("Primary signal", ["reviewer_pattern", "business_trend", "text_similarity", "combination"])
            reasoning = st.text_area("Reasoning (1-2 sentences)")
            submitted = st.form_submit_button("Submit judgment")

        if submitted:
            predicted_filtered = predicted.startswith("Filtered")
            true_filtered = bool(row["filtered"])
            mlflow_tracking.log_judgment(
                judge_type="human", config_version=active_config["version"], review_id=review_id,
                true_filtered=true_filtered, predicted_filtered=predicted_filtered,
                confidence=confidence, primary_signal=primary_signal,
            )
            st.session_state["last_human_review_id"] = int(review_id)
            st.session_state["last_human_correct"] = (predicted_filtered == true_filtered)
            st.session_state["last_human_true_filtered"] = true_filtered
            st.session_state["human_review_id"] = None
            mark_step_done(2)
            st.rerun()

        if st.session_state.get("last_human_review_id") is not None:
            rid = st.session_state["last_human_review_id"]
            correct = st.session_state["last_human_correct"]
            true_filtered = st.session_state["last_human_true_filtered"]
            if correct:
                st.success(f"✓ Correct on review #{rid} — Yelp's label is **{'filtered' if true_filtered else 'recommended'}**.")
            else:
                st.error(f"✗ Not a match on review #{rid} — Yelp's label is actually **{'filtered' if true_filtered else 'recommended'}**.")
            if st.button("Now see what the agent decides on this same review →", type="primary"):
                rid_to_pass = st.session_state.pop("last_human_review_id")
                st.session_state.pop("last_human_correct", None)
                st.session_state.pop("last_human_true_filtered", None)
                goto("4. Try it — Agent", handoff_review_id=rid_to_pass)


# ============================================================
# 4. TRY IT — AGENT
# ============================================================
elif page == "4. Try it — Agent":
    st.title("Try it — the real agent, live")
    st.caption(f"Running against the active config: **{active_config['version']}** — "
               f"{active_config['provider']}/{active_config['model']}, temperature {active_config['temperature']}")

    if not os.environ.get(PROVIDER_PRESETS[active_config["provider"]]["api_key_env"]):
        key_env = PROVIDER_PRESETS[active_config["provider"]]["api_key_env"]
        provider_urls = {
            "GROQ_API_KEY": "https://console.groq.com/keys",
            "OPENROUTER_API_KEY": "https://openrouter.ai/keys",
            "GOOGLE_API_KEY": "https://aistudio.google.com/apikey",
            "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
        }
        st.warning(
            f"**No API key set for {active_config['provider']}.** Three steps to run a live investigation:\n\n"
            f"1. Get a free key → [{provider_urls.get(key_env, 'provider console')}]({provider_urls.get(key_env, '#')}) (about 30 seconds)\n"
            f"2. Paste it into the **{active_config['provider'].title()}** field in the sidebar\n"
            f"3. Come back here and click **Investigate live**"
        )

    sample = load_sample()
    already_judged = judged_review_ids("agent", active_config["version"])
    pool = sample[~sample["review_id"].isin(already_judged)]

    handoff_id = st.session_state.pop("handoff_review_id", None)
    if handoff_id is not None and handoff_id in sample["review_id"].values:
        st.session_state["agent_review_id"] = int(handoff_id)
        st.caption(f"↪ Continuing from your Human Review judgment on review #{handoff_id}")
    elif "agent_review_id" not in st.session_state or st.session_state["agent_review_id"] not in pool["review_id"].values:
        if not pool.empty:
            st.session_state["agent_review_id"] = int(pool.sample(1)["review_id"].iloc[0])

    if pool.empty and handoff_id is None:
        st.success("The agent has investigated every review in the sample under this config version!")
    else:
        options = sample["review_id"].tolist()
        col_pick, col_random = st.columns([3, 1])
        with col_pick:
            review_id = st.selectbox("Review", options,
                                      index=options.index(st.session_state["agent_review_id"])
                                      if st.session_state["agent_review_id"] in options else 0)
        with col_random:
            if st.button("🎲 Random review", key="agent_random"):
                st.session_state["agent_review_id"] = int(pool.sample(1)["review_id"].iloc[0] if not pool.empty else sample.sample(1)["review_id"].iloc[0])
                st.rerun()

        row = sample[sample["review_id"] == review_id].iloc[0]
        st.markdown(f"**Rating:** {'⭐' * int(row['rating'])} &nbsp;&nbsp; **Date:** {row['date']}")
        st.info(row["review_text"])

        if st.button("▶ Investigate live", type="primary"):
            review = {
                "review_id": int(review_id), "user_id": str(int(row["user_id"])),
                "prod_id": str(int(row["prod_id"])), "rating": float(row["rating"]),
                "date": row["date"], "review_text": row["review_text"],
            }
            trace_area = st.container()
            final_result = None
            n_tool_calls = 0

            with st.status("Investigating...", expanded=True) as status:
                for event in run_investigation_stream(review, active_config):
                    if event["type"] == "error":
                        status.update(label="Investigation failed", state="error")
                        st.error(event["message"])
                        break
                    elif event["type"] == "turn":
                        trace_area.markdown(f"**Turn {event['turn'] + 1}**")
                    elif event["type"] == "tool_call":
                        n_tool_calls += 1
                        trace_area.markdown(f"→ called `{event['tool']}`")
                        trace_area.json(event["result"])
                    elif event["type"] == "final":
                        final_result = event
                        status.update(label="Investigation complete", state="complete")

            if final_result and final_result.get("predicted_filtered") is not None:
                threshold = active_config["confidence_threshold"]
                predicted_filtered = final_result["predicted_filtered"]
                confidence = final_result["confidence"]
                true_filtered = bool(row["filtered"])

                routing = compute_routing(predicted_filtered, confidence, threshold)

                st.divider()
                verdict_col, routing_col = st.columns(2)
                with verdict_col:
                    st.metric("Verdict", "Likely fake" if predicted_filtered else "Likely genuine",
                              f"{confidence:.0%} confidence")
                with routing_col:
                    st.metric("Routing decision", ROUTING_LABELS[routing])
                    if routing == "queue":
                        st.caption("⚠️ Confidence fell below the threshold — the policy falls back to a human, "
                                   "exactly as designed, rather than guessing.")

                st.markdown(f"**Primary signal:** {final_result['primary_signal']}")
                st.markdown(f"**Reasoning:** {final_result['reasoning']}")

                correct = predicted_filtered == true_filtered
                if st.button("Reveal Yelp's actual answer"):
                    if correct:
                        st.success(f"✓ Matches — actually **{'filtered' if true_filtered else 'recommended'}**.")
                    else:
                        st.error(f"✗ Does not match — actually **{'filtered' if true_filtered else 'recommended'}**.")

                mlflow_tracking.log_judgment(
                    judge_type="agent", config_version=active_config["version"], review_id=review_id,
                    true_filtered=true_filtered, predicted_filtered=predicted_filtered, confidence=confidence,
                    primary_signal=final_result["primary_signal"], reasoning=final_result.get("reasoning"),
                    turns_used=final_result.get("turns_used"),
                    n_tool_calls=n_tool_calls, provider=active_config["provider"], model=active_config["model"],
                    temperature=active_config["temperature"], extra_tags={"framework": "agno"},
                )
                mark_step_done(3)
            elif final_result:
                st.warning(f"No judgment reached: {final_result.get('reasoning')}")

        st.divider()
        if st.button("See the Analyst Queue review this case →"):
            goto("5. Analyst Queue")


# ============================================================
# 5. ANALYST QUEUE
# ============================================================
elif page == "5. Analyst Queue":
    st.title("Analyst Queue — review the agent's work")
    st.caption("The agent already investigated these cases. Accept its verdict, override it, or "
               "escalate for deeper review — every action is logged, so agreement and override "
               "rate become real, visible metrics instead of silent disagreement.")

    sample = load_sample()
    tb = toolbox()
    pending_appeals = mlflow_tracking.get_pending_appeals(active_config["version"])
    agent_judged = sorted(judged_review_ids("agent", active_config["version"]))

    if pending_appeals:
        st.warning(f"🚨 {len(pending_appeals)} case(s) appealed by a reviewer/business — shown first below.")

    queue_ids = pending_appeals + [rid for rid in agent_judged if rid not in pending_appeals]
    if not queue_ids:
        st.info("No cases in the queue yet — the agent hasn't investigated anything under this "
                "config version. Go to **4. Try it — Agent** to run one, or check **6. Check My "
                "Review** for an appeal to land here.")
    else:
        default_id = st.session_state.get("queue_review_id", queue_ids[0])
        if default_id not in queue_ids:
            default_id = queue_ids[0]
        review_id = st.selectbox(
            "Case", queue_ids, index=queue_ids.index(default_id),
            format_func=lambda rid: f"#{rid}" + (" 🚨 appealed" if rid in pending_appeals else ""),
        )
        st.session_state["queue_review_id"] = review_id
        is_appealed = review_id in pending_appeals

        row = sample[sample["review_id"] == review_id].iloc[0]
        user_id, prod_id = str(int(row["user_id"])), str(int(row["prod_id"]))
        st.markdown(f"**Rating:** {'⭐' * int(row['rating'])} &nbsp;&nbsp; **Date:** {row['date']}")
        st.info(row["review_text"])

        with st.expander("Evidence — reviewer history"):
            st.json(tb.reviewer_history_lookup(user_id, exclude_review_id=int(review_id)))
        with st.expander("Evidence — business trend"):
            st.json(tb.business_trend_lookup(prod_id, row["date"], exclude_review_id=int(review_id)))
        with st.expander("Evidence — text similarity vs. reviewer"):
            st.json(tb.text_similarity_check(row["review_text"], "reviewer", user_id, exclude_review_id=int(review_id)))
        with st.expander("Evidence — text similarity vs. business"):
            st.json(tb.text_similarity_check(row["review_text"], "business", prod_id, exclude_review_id=int(review_id)))

        agent_judgment = mlflow_tracking.get_agent_judgment(review_id, active_config["version"])
        st.divider()
        if agent_judgment is None:
            st.info("The agent hasn't investigated this case under the active config yet.")
            if st.button("▶ Run agent investigation"):
                review = {
                    "review_id": int(review_id), "user_id": user_id, "prod_id": prod_id,
                    "rating": float(row["rating"]), "date": row["date"], "review_text": row["review_text"],
                }
                final_result = None
                n_tool_calls = 0
                with st.status("Investigating...", expanded=True) as status:
                    for event in run_investigation_stream(review, active_config):
                        if event["type"] == "error":
                            status.update(label="Investigation failed", state="error")
                            st.error(event["message"])
                            break
                        elif event["type"] == "tool_call":
                            n_tool_calls += 1
                            st.markdown(f"→ called `{event['tool']}`")
                        elif event["type"] == "final":
                            final_result = event
                            status.update(label="Investigation complete", state="complete")
                if final_result and final_result.get("predicted_filtered") is not None:
                    mlflow_tracking.log_judgment(
                        judge_type="agent", config_version=active_config["version"], review_id=review_id,
                        true_filtered=bool(row["filtered"]), predicted_filtered=final_result["predicted_filtered"],
                        confidence=final_result["confidence"], primary_signal=final_result["primary_signal"],
                        reasoning=final_result.get("reasoning"), turns_used=final_result.get("turns_used"),
                        n_tool_calls=n_tool_calls, provider=active_config["provider"], model=active_config["model"],
                        temperature=active_config["temperature"], extra_tags={"framework": "agno"},
                    )
                    st.rerun()
        else:
            predicted_filtered = agent_judgment["predicted_filtered"]
            confidence = agent_judgment["confidence"] or 0.0
            threshold = active_config["confidence_threshold"]
            routing = compute_routing(predicted_filtered, confidence, threshold)

            verdict_col, routing_col = st.columns(2)
            with verdict_col:
                st.metric("Agent's verdict", "Likely fake" if predicted_filtered else "Likely genuine",
                          f"{confidence:.0%} confidence")
            with routing_col:
                st.metric("Agent's routing", ROUTING_LABELS[routing])
            st.markdown(f"**Primary signal:** {agent_judgment['primary_signal']}")
            st.markdown(f"**Agent's reasoning:** {agent_judgment['reasoning'] or '*(not recorded)*'}")

            opposite_label = "Recommended (genuine)" if predicted_filtered else "Filtered (fake)"
            st.divider()
            with st.form(f"analyst_action_form_{review_id}"):
                action_label = st.radio(
                    "Your decision",
                    ["Accept the agent's verdict", f"Override — mark as {opposite_label} instead",
                     "Escalate for deeper review"],
                )
                reason = st.text_area("Reason (required for override or escalation)")
                action_submitted = st.form_submit_button("Submit decision")

            if action_submitted:
                is_override = action_label.startswith("Override")
                action_key = "override" if is_override else ("escalate" if action_label.startswith("Escalate") else "accept")
                if action_key != "accept" and not reason.strip():
                    st.error("A reason is required for an override or escalation.")
                else:
                    mlflow_tracking.log_analyst_action(
                        config_version=active_config["version"], review_id=review_id,
                        agent_confidence=confidence, action=action_key,
                        override_to=(not predicted_filtered) if is_override else None,
                        reason=reason, resolves_appeal=is_appealed,
                    )
                    mark_step_done(4)
                    st.success(f"Logged: **{action_label}**" + (" — appeal resolved." if is_appealed else "."))
                    st.rerun()


# ============================================================
# 6. CHECK MY REVIEW
# ============================================================
elif page == "6. Check My Review":
    st.title("Check My Review")
    st.caption("As the reviewer or business behind a review, see what happened to it and, if it "
               "was removed, contest the decision. This mirrors what a real reviewer would "
               "see — the ground truth used elsewhere in this dashboard is deliberately not "
               "shown here, since a real reviewer wouldn't know it either.")

    sample = load_sample()
    review_id = st.selectbox("Your review", sample["review_id"].tolist(), key="reviewer_review_id")
    row = sample[sample["review_id"] == review_id].iloc[0]
    st.markdown(f"**Rating:** {'⭐' * int(row['rating'])} &nbsp;&nbsp; **Date:** {row['date']}")
    st.info(row["review_text"])

    agent_judgment = mlflow_tracking.get_agent_judgment(review_id, active_config["version"])
    st.divider()

    if agent_judgment is None:
        st.info("This review hasn't been checked against the current agent config yet.")
        if st.button("Check status now", type="primary"):
            review = {
                "review_id": int(review_id), "user_id": str(int(row["user_id"])),
                "prod_id": str(int(row["prod_id"])), "rating": float(row["rating"]),
                "date": row["date"], "review_text": row["review_text"],
            }
            final_result = None
            n_tool_calls = 0
            with st.status("Checking...", expanded=False) as status:
                for event in run_investigation_stream(review, active_config):
                    if event["type"] == "error":
                        status.update(label="Check failed", state="error")
                        st.error(event["message"])
                        break
                    elif event["type"] == "tool_call":
                        n_tool_calls += 1
                    elif event["type"] == "final":
                        final_result = event
                        status.update(label="Done", state="complete")
            if final_result and final_result.get("predicted_filtered") is not None:
                mlflow_tracking.log_judgment(
                    judge_type="agent", config_version=active_config["version"], review_id=review_id,
                    true_filtered=bool(row["filtered"]), predicted_filtered=final_result["predicted_filtered"],
                    confidence=final_result["confidence"], primary_signal=final_result["primary_signal"],
                    reasoning=final_result.get("reasoning"), turns_used=final_result.get("turns_used"),
                    n_tool_calls=n_tool_calls, provider=active_config["provider"], model=active_config["model"],
                    temperature=active_config["temperature"], extra_tags={"framework": "agno"},
                )
                st.rerun()
    else:
        predicted_filtered = agent_judgment["predicted_filtered"]
        confidence = agent_judgment["confidence"] or 0.0
        threshold = active_config["confidence_threshold"]
        routing = compute_routing(predicted_filtered, confidence, threshold)

        if routing == "auto_remove":
            st.error("🔴 Your review was removed automatically.")
        elif routing == "queue":
            st.warning("🟡 Your review is flagged and waiting on a human reviewer's decision.")
        else:
            st.success("🟢 Your review is live — no action was taken.")
        st.markdown(f"**Why:** {agent_judgment['reasoning'] or '*(not recorded)*'}")
        mark_step_done(5)

        if routing == "auto_remove":
            pending = mlflow_tracking.get_pending_appeals(active_config["version"])
            just_appealed = st.session_state.get("last_appeal_review_id") == review_id

            if review_id in pending:
                st.info("You've already appealed this decision — it's waiting in the Analyst Queue for human review.")
                if just_appealed and st.button("See the Analyst Queue →", type="primary"):
                    st.session_state.pop("last_appeal_review_id", None)
                    goto("5. Analyst Queue", queue_review_id=review_id)
            else:
                st.session_state.pop("last_appeal_review_id", None)
                with st.form(f"appeal_form_{review_id}"):
                    note = st.text_area("Optional: add context for the reviewer (why you believe this is genuine)")
                    appeal_submitted = st.form_submit_button("Appeal this decision")
                if appeal_submitted:
                    mlflow_tracking.log_appeal(review_id=review_id, config_version=active_config["version"], note=note)
                    st.session_state["last_appeal_review_id"] = review_id
                    st.rerun()


# ============================================================
# 7. PLAY — TWEAK THE AGENT
# ============================================================
elif page == "7. Play — Tweak the Agent":
    st.title("Play — you're the PM now")
    st.caption("Change the agent's system prompt, temperature, provider, or auto-remove threshold. One click deploys a new version — past versions stay comparable.")

    st.markdown("**Quick experiments** — one click pre-fills the form below, review it, then deploy:")
    exp_cols = st.columns(len(QUICK_EXPERIMENTS))
    for col, (key, exp) in zip(exp_cols, QUICK_EXPERIMENTS.items()):
        with col:
            if st.button(exp["label"], help=exp["help"], key=f"exp_{key}"):
                st.session_state["config_prefill"] = exp["changes"]
                st.rerun()

    prefill = st.session_state.get("config_prefill", {})
    defaults = {**active_config, **prefill}

    st.divider()
    with st.form("deploy_form"):
        system_prompt = st.text_area("System prompt", value=defaults["system_prompt"], height=220)
        col1, col2, col3 = st.columns(3)
        with col1:
            provider = st.selectbox("Provider", list(PROVIDER_PRESETS.keys()),
                                     index=list(PROVIDER_PRESETS.keys()).index(defaults["provider"]))
        with col2:
            model = st.text_input("Model", value=defaults["model"])
        with col3:
            temperature = st.slider("Temperature", 0.0, 1.0, float(defaults["temperature"]), 0.05)

        confidence_threshold = st.slider("Auto-remove confidence threshold", 0.0, 1.0,
                                          float(defaults["confidence_threshold"]), 0.05)
        reason = st.text_input("Reason for this change (optional — auto-filled from the diff if left blank)")

        deployed = st.form_submit_button("🚀 Deploy new version", type="primary")

    if deployed:
        new_version = config_store.deploy_new_version(
            system_prompt=system_prompt, temperature=temperature, provider=provider,
            model=model, confidence_threshold=confidence_threshold, reason=reason,
        )
        st.session_state.pop("config_prefill", None)
        mark_step_done(6)
        st.session_state["just_deployed"] = new_version
        st.rerun()

    if st.session_state.get("just_deployed"):
        jd = st.session_state["just_deployed"]
        st.success(f"Deployed **{jd['version']}**: {jd['reason']}")
        if st.button("See how this changed performance →", type="primary"):
            st.session_state.pop("just_deployed", None)
            goto("8. Track Performance")

    st.divider()
    st.subheader("Version history")
    versions = config_store.list_versions()
    hist_df = pd.DataFrame([
        {"version": v["version"], "provider": v["provider"], "model": v["model"],
         "temperature": v["temperature"], "threshold": v["confidence_threshold"],
         "reason": v["reason"], "deployed_at": v["deployed_at"]}
        for v in reversed(versions)
    ])
    st.dataframe(hist_df, width="stretch")

    rollback_target = st.selectbox("Roll back to version", [v["version"] for v in reversed(versions)])
    if st.button("Roll back"):
        config_store.rollback_to(rollback_target)
        st.rerun()


# ============================================================
# 8. TRACK PERFORMANCE
# ============================================================
elif page == "8. Track Performance":
    st.title("Track Performance")
    st.caption("Every human and agent judgment, grouped by config version — logged to MLflow, not a hand-rolled CSV.")

    summary = mlflow_tracking.summarize_by_group()
    if summary.empty:
        st.info("No judgments logged yet. Go try it as a human or the agent first.")
    else:
        agent_summary = summary[summary["tags.judge_type"] == "agent"].copy()
        if len(agent_summary) >= 2:
            agent_summary["_order"] = range(len(agent_summary))
            prev_row, latest_row = agent_summary.iloc[-2], agent_summary.iloc[-1]
            delta = latest_row["accuracy"] - prev_row["accuracy"]
            arrow = "📈" if delta > 0 else ("📉" if delta < 0 else "➡️")
            st.markdown(
                f"### {arrow} Your last agent config change moved accuracy from "
                f"**{prev_row['accuracy']:.0%}** (`{prev_row['tags.config_version']}`) to "
                f"**{latest_row['accuracy']:.0%}** (`{latest_row['tags.config_version']}`)"
            )
        else:
            st.markdown("Deploy at least two config versions and run the agent under each to see a before/after delta here.")

        st.divider()
        st.subheader("Performance by judge type & config version")
        display_cols = ["tags.judge_type", "tags.config_version", "n", "accuracy", "accuracy_ci", "precision", "recall"]
        st.dataframe(summary[[c for c in display_cols if c in summary.columns]], width="stretch")

        st.subheader("Accuracy by config version")
        st.bar_chart(
            summary.pivot_table(index="tags.config_version", columns="tags.judge_type", values="accuracy"),
            width="stretch",
        )

        st.subheader("Precision / Recall by config version (agent only)")
        if not agent_summary.empty:
            pr_pivot = agent_summary.set_index("tags.config_version")[["precision", "recall"]]
            st.bar_chart(pr_pivot, width="stretch")

        with st.expander("Raw MLflow run log"):
            st.dataframe(mlflow_tracking.get_all_runs(), width="stretch")
