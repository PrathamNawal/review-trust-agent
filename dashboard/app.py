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

# ---------- one-time setup ----------
if "seeded" not in st.session_state:
    mlflow_tracking.seed_historical_baselines()
    st.session_state["seeded"] = True


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


# ---------- sidebar: nav + API keys ----------
st.sidebar.title("Review Trust Agent")
page = st.sidebar.radio("View", ["Overview", "Human Review", "Agent Review", "System Config", "Performance Tracking"])

st.sidebar.divider()
st.sidebar.caption("API keys (session only — never written to disk)")
for env_name, label in [("GROQ_API_KEY", "Groq"), ("OPENROUTER_API_KEY", "OpenRouter"),
                         ("GOOGLE_API_KEY", "Gemini"), ("ANTHROPIC_API_KEY", "Claude")]:
    current = os.environ.get(env_name, "")
    entered = st.sidebar.text_input(label, value=current, type="password", key=f"key_{env_name}")
    if entered:
        os.environ[env_name] = entered

active_config = config_store.get_active_config()
st.sidebar.divider()
st.sidebar.caption(f"Active config: **{active_config['version']}** ({active_config['provider']}/{active_config['model']})")


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Overview")

    summary = mlflow_tracking.summarize_by_group()

    c1, c2, c3, c4 = st.columns(4)
    human_rows = summary[summary["tags.judge_type"] == "human"] if not summary.empty else pd.DataFrame()
    agent_rows = summary[summary["tags.judge_type"] == "agent"] if not summary.empty else pd.DataFrame()

    c1.metric("Human judgments logged", int(human_rows["n"].sum()) if not human_rows.empty else 0)
    c2.metric("Agent judgments logged", int(agent_rows["n"].sum()) if not agent_rows.empty else 0)
    c3.metric("Config versions deployed", len(config_store.list_versions()))
    c4.metric("Active config", active_config["version"])

    st.divider()
    st.subheader("Performance by judge type & config version")
    if summary.empty:
        st.info("No judgments logged yet. Use Human Review or Agent Review to start generating data.")
    else:
        display_cols = ["tags.judge_type", "tags.config_version", "n", "accuracy", "accuracy_ci", "precision", "recall"]
        st.dataframe(summary[[c for c in display_cols if c in summary.columns]], width="stretch")

        st.bar_chart(
            summary.pivot_table(index="tags.config_version", columns="tags.judge_type", values="accuracy"),
            width="stretch",
        )

    st.divider()
    st.subheader("What each mode means")
    st.markdown("""
- **Human Review** — you act as the trust & safety analyst: read the review and real evidence, judge it blind, see the answer key after.
- **Agent Review** — the real agent investigates live, using whichever provider/model/temperature is in the **active config** below, with real tool calls.
- **System Config** — edit the system prompt, temperature, and provider; one click deploys a new version and all future Agent Review runs use it.
- **Performance Tracking** — every judgment, human or agent, is logged to MLflow, grouped by config version, so before/after comparison is real, not eyeballed.
""")


# ============================================================
# HUMAN REVIEW
# ============================================================
elif page == "Human Review":
    st.title("Human Review — act as the analyst")
    st.caption("Real reviews, real evidence, blind judgment. Your answers are logged and scored against Yelp's real label.")

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
            correct = predicted_filtered == true_filtered
            if correct:
                st.success(f"✓ Correct — Yelp's label is **{'filtered' if true_filtered else 'recommended'}**.")
            else:
                st.error(f"✗ Not a match — Yelp's label is actually **{'filtered' if true_filtered else 'recommended'}**.")
            st.session_state["human_review_id"] = None
            st.rerun()


# ============================================================
# AGENT REVIEW
# ============================================================
elif page == "Agent Review":
    st.title("Agent Review — the real agent, live")
    st.caption(f"Running against the active config: **{active_config['version']}** — "
               f"{active_config['provider']}/{active_config['model']}, temperature {active_config['temperature']}")

    if not os.environ.get(PROVIDER_PRESETS[active_config["provider"]]["api_key_env"]):
        st.warning(f"No API key set for **{active_config['provider']}**. Add it in the sidebar to run live investigations.")

    sample = load_sample()
    already_judged = judged_review_ids("agent", active_config["version"])
    pool = sample[~sample["review_id"].isin(already_judged)]

    if "agent_review_id" not in st.session_state or st.session_state["agent_review_id"] not in pool["review_id"].values:
        if not pool.empty:
            st.session_state["agent_review_id"] = int(pool.sample(1)["review_id"].iloc[0])

    if pool.empty:
        st.success("The agent has investigated every review in the sample under this config version!")
    else:
        col_pick, col_random = st.columns([3, 1])
        with col_pick:
            review_id = st.selectbox("Review", pool["review_id"].tolist(),
                                      index=pool["review_id"].tolist().index(st.session_state["agent_review_id"])
                                      if st.session_state["agent_review_id"] in pool["review_id"].values else 0)
        with col_random:
            if st.button("🎲 Random review", key="agent_random"):
                st.session_state["agent_review_id"] = int(pool.sample(1)["review_id"].iloc[0])
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

                if predicted_filtered and confidence >= threshold:
                    routing = "auto_remove"
                elif not predicted_filtered and confidence >= threshold:
                    routing = "ignore"
                else:
                    routing = "queue"

                st.divider()
                verdict_col, routing_col = st.columns(2)
                with verdict_col:
                    st.metric("Verdict", "Likely fake" if predicted_filtered else "Likely genuine",
                              f"{confidence:.0%} confidence")
                with routing_col:
                    routing_label = {"auto_remove": "🔴 Auto-remove", "queue": "🟡 Routed to human queue",
                                      "ignore": "🟢 Ignored"}[routing]
                    st.metric("Routing decision", routing_label)
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
                    primary_signal=final_result["primary_signal"], turns_used=final_result.get("turns_used"),
                    n_tool_calls=n_tool_calls, provider=active_config["provider"], model=active_config["model"],
                    temperature=active_config["temperature"],
                )
            elif final_result:
                st.warning(f"No judgment reached: {final_result.get('reasoning')}")


# ============================================================
# SYSTEM CONFIG
# ============================================================
elif page == "System Config":
    st.title("System Config")
    st.caption("Edit the live agent's configuration. Deploying creates a new version — past versions stay comparable.")

    with st.form("deploy_form"):
        system_prompt = st.text_area("System prompt", value=active_config["system_prompt"], height=220)
        col1, col2, col3 = st.columns(3)
        with col1:
            provider = st.selectbox("Provider", list(PROVIDER_PRESETS.keys()),
                                     index=list(PROVIDER_PRESETS.keys()).index(active_config["provider"]))
        with col2:
            model = st.text_input("Model", value=active_config["model"])
        with col3:
            temperature = st.slider("Temperature", 0.0, 1.0, float(active_config["temperature"]), 0.05)

        confidence_threshold = st.slider("Auto-remove confidence threshold", 0.0, 1.0,
                                          float(active_config["confidence_threshold"]), 0.05)
        reason = st.text_input("Reason for this change (optional — auto-filled from the diff if left blank)")

        deployed = st.form_submit_button("🚀 Deploy new version", type="primary")

    if deployed:
        new_version = config_store.deploy_new_version(
            system_prompt=system_prompt, temperature=temperature, provider=provider,
            model=model, confidence_threshold=confidence_threshold, reason=reason,
        )
        st.success(f"Deployed **{new_version['version']}**: {new_version['reason']}")
        st.rerun()

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
# PERFORMANCE TRACKING
# ============================================================
elif page == "Performance Tracking":
    st.title("Performance Tracking")
    st.caption("Every human and agent judgment, grouped by config version — logged to MLflow, not a hand-rolled CSV.")

    summary = mlflow_tracking.summarize_by_group()
    if summary.empty:
        st.info("No judgments logged yet.")
    else:
        st.dataframe(summary, width="stretch")

        st.subheader("Accuracy by config version")
        pivot = summary.pivot_table(index="tags.config_version", columns="tags.judge_type", values="accuracy")
        st.bar_chart(pivot, width="stretch")

        st.subheader("Precision / Recall by config version (agent only)")
        agent_summary = summary[summary["tags.judge_type"] == "agent"]
        if not agent_summary.empty:
            pr_pivot = agent_summary.set_index("tags.config_version")[["precision", "recall"]]
            st.bar_chart(pr_pivot, width="stretch")

        with st.expander("Raw MLflow run log"):
            st.dataframe(mlflow_tracking.get_all_runs(), width="stretch")
