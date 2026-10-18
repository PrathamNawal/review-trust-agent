"""Human-review and live-agent review pages."""

import os

import streamlit as st

from dashboard.jev_agent import get_typesafe_api_key
from dashboard.agent_service import investigate_and_log
from dashboard.decision_policy import compute_routing, ROUTING_LABELS, ROUTING_TONE
from dashboard.ui_components import (
    render_badge, render_evidence_card, render_evidence_grid, resolve_tool_key,
)


def render_human_review(sample, toolbox, active_config, mlflow_tracking,
                        judged_review_ids, mark_step_done, goto):
    st.title("Try it — act as the analyst")
    st.caption("Real reviews, real evidence, blind judgment. Your answer is logged and scored against Yelp's real label.")

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
        render_evidence_grid(st, toolbox, int(review_id), user_id, prod_id, row["review_text"], row["date"])

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
            review_id = st.session_state["last_human_review_id"]
            correct = st.session_state["last_human_correct"]
            true_filtered = st.session_state["last_human_true_filtered"]
            if correct:
                st.success(f"✓ Correct on review #{review_id} — Yelp's label is **{'filtered' if true_filtered else 'recommended'}**.")
            else:
                st.error(f"✗ Not a match on review #{review_id} — Yelp's label is actually **{'filtered' if true_filtered else 'recommended'}**.")
            if st.button("Now see what the agent decides on this same review →", type="primary"):
                handoff_id = st.session_state.pop("last_human_review_id")
                st.session_state.pop("last_human_correct", None)
                st.session_state.pop("last_human_true_filtered", None)
                goto("4. Try it — Agent", handoff_review_id=handoff_id)


def render_agent_review(sample, active_config, provider_presets, judged_review_ids,
                        investigate_and_log, mark_step_done, goto):
    st.title("Try it — the real agent, live")
    st.caption(f"Running against the active config: **{active_config['version']}** — "
               f"{active_config.get('architecture', 'agno')} · "
               f"{active_config['provider']}/{active_config['model']}")

    is_jev = active_config.get("architecture") == "jev"
    key_env = "TYPESAFE_API_KEY" if is_jev else provider_presets[active_config["provider"]]["api_key_env"]
    key_present = get_typesafe_api_key() if is_jev else os.environ.get(key_env)
    if not key_present:
        provider_urls = {
            "GROQ_API_KEY": "https://console.groq.com/keys",
            "OPENROUTER_API_KEY": "https://openrouter.ai/keys",
            "GOOGLE_API_KEY": "https://aistudio.google.com/apikey",
            "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
            "TYPESAFE_API_KEY": "https://docs.typesafe.ai/introduction/quickstart",
        }
        provider_name = "Jev Model (TypeSafe)" if is_jev else active_config["provider"].title()
        st.warning(
            f"**No API key set for {provider_name}.** Three steps to run a live investigation:\n\n"
            f"1. Get a free key → [{provider_urls.get(key_env, 'provider console')}]({provider_urls.get(key_env, '#')}) (about 30 seconds)\n"
            f"2. Paste it into the **{provider_name}** field in the sidebar\n"
            f"3. Come back here and click **Investigate live**"
        )

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
                chosen_pool = pool if not pool.empty else sample
                st.session_state["agent_review_id"] = int(chosen_pool.sample(1)["review_id"].iloc[0])
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
            with st.status("Investigating...", expanded=True) as status:
                for event in investigate_and_log(review, active_config, bool(row["filtered"])):
                    if event["type"] == "error":
                        status.update(label="Investigation failed", state="error")
                        st.error(event["message"])
                        break
                    if event["type"] == "turn":
                        trace_area.markdown(f"**Turn {event['turn'] + 1}**")
                    elif event["type"] == "tool_call":
                        render_evidence_card(trace_area, resolve_tool_key(event["tool"], event["input"]),
                                             event["result"])
                    elif event["type"] == "final":
                        final_result = event
                        status.update(label="Investigation complete", state="complete")

            if final_result and final_result.get("predicted_filtered") is not None:
                threshold = active_config["confidence_threshold"]
                predicted_filtered = final_result["predicted_filtered"]
                confidence = final_result["confidence"]
                true_filtered = bool(row["filtered"])
                routing = compute_routing(predicted_filtered, confidence, threshold)
                is_jev = active_config.get("architecture") == "jev"
                confidence_display = (
                    "Jev Choice · experimental" if is_jev else
                    (f"{confidence:.0%} confidence" if confidence is not None else "confidence not calibrated")
                )

                st.divider()
                verdict_col, routing_col = st.columns(2)
                with verdict_col:
                    st.metric(
                        "Verdict · Jev Choice (experimental)" if is_jev else "Verdict",
                        "Likely fake" if predicted_filtered else "Likely genuine",
                        None if is_jev else confidence_display,
                    )
                with routing_col:
                    st.caption("Routing decision")
                    render_badge(ROUTING_LABELS[routing], ROUTING_TONE[routing])
                    if routing == "queue":
                        if is_jev:
                            st.caption("Jev Choice is advisory. A human must review this result; no automatic action is taken.")
                        elif confidence is None:
                            st.caption("⚠️ This model's output is not calibrated for autonomous routing; a human must review it.")
                        else:
                            st.caption("⚠️ Confidence fell below the threshold — the policy falls back to a human, "
                                       "exactly as designed, rather than guessing.")
                st.markdown(f"**Primary signal:** {final_result['primary_signal']}")
                st.markdown(f"**Reasoning:** {final_result['reasoning']}")
                if is_jev:
                    st.info(
                        "Jev Choice selected a label from the evidence shown above. Its label "
                        "probabilities are not validated or calibrated as Yelp fraud probabilities. "
                        "YelpZip's `filtered` label is a benchmark reference, not verified fraud ground truth."
                    )
                if final_result.get("jev_choice_probabilities"):
                    st.caption("Model-reported Choice probabilities (not calibrated for fraud detection)")
                    st.json(final_result["jev_choice_probabilities"])

                correct = predicted_filtered == true_filtered
                if st.button("Reveal Yelp's actual answer"):
                    if correct:
                        st.success(f"✓ Matches — actually **{'filtered' if true_filtered else 'recommended'}**.")
                    else:
                        st.error(f"✗ Does not match — actually **{'filtered' if true_filtered else 'recommended'}**.")
                mark_step_done(3)
            elif final_result:
                st.warning(f"No judgment reached: {final_result.get('reasoning')}")

        st.divider()
        if st.button("See the Analyst Queue review this case →"):
            goto("5. Analyst Queue")
