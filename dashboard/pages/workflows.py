"""Analyst queue and reviewer self-service views."""

import streamlit as st

from dashboard.decision_policy import compute_routing, ROUTING_LABELS, ROUTING_TONE
from dashboard.ui_components import render_badge, render_evidence_card, render_evidence_grid, resolve_tool_key


def render_analyst_queue(sample, toolbox, active_config, mlflow_tracking, judged_review_ids,
                         investigate_and_log, mark_step_done):
    st.title("Analyst Queue — review the agent's work")
    st.caption("The agent already investigated these cases. Accept its verdict, override it, or "
               "escalate for deeper review — every action is logged, so agreement and override "
               "rate become real, visible metrics instead of silent disagreement.")

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

        render_evidence_grid(st, toolbox, int(review_id), user_id, prod_id, row["review_text"], row["date"])

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
                    for event in investigate_and_log(review, active_config, bool(row["filtered"])):
                        if event["type"] == "error":
                            status.update(label="Investigation failed", state="error")
                            st.error(event["message"])
                            break
                        elif event["type"] == "tool_call":
                            n_tool_calls += 1
                            render_evidence_card(st, resolve_tool_key(event["tool"], event["input"]),
                                                  event["result"])
                        elif event["type"] == "final":
                            final_result = event
                            status.update(label="Investigation complete", state="complete")
                if final_result and final_result.get("predicted_filtered") is not None:
                    st.rerun()
        else:
            predicted_filtered = agent_judgment["predicted_filtered"]
            confidence = agent_judgment["confidence"]
            threshold = active_config["confidence_threshold"]
            routing = compute_routing(predicted_filtered, confidence, threshold)
            is_jev = active_config.get("architecture") == "jev"
            jev_method = active_config.get("jev_method", "choice").title()
            confidence_display = (
                f"Jev {jev_method} · experimental" if is_jev else
                (f"{confidence:.0%} confidence" if confidence is not None else "confidence not calibrated")
            )

            verdict_col, routing_col = st.columns(2)
            with verdict_col:
                st.metric(
                    f"Agent's verdict · Jev {jev_method} (experimental)" if is_jev else "Agent's verdict",
                    "Likely fake" if predicted_filtered else "Likely genuine",
                    None if is_jev else confidence_display,
                )
            with routing_col:
                st.caption("Agent's routing")
                render_badge(ROUTING_LABELS[routing], ROUTING_TONE[routing])
                if is_jev:
                    st.caption("Advisory result; a human makes the decision.")
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


def render_check_my_review(sample, active_config, mlflow_tracking, investigate_and_log,
                           mark_step_done, goto):
    st.title("Check My Review")
    st.caption("As the reviewer or business behind a review, see what happened to it and, if it "
               "was removed, contest the decision. This mirrors what a real reviewer would "
               "see — the ground truth used elsewhere in this dashboard is deliberately not "
               "shown here, since a real reviewer wouldn't know it either.")

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
                for event in investigate_and_log(review, active_config, bool(row["filtered"])):
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
                st.rerun()
    else:
        predicted_filtered = agent_judgment["predicted_filtered"]
        confidence = agent_judgment["confidence"]
        threshold = active_config["confidence_threshold"]
        routing = compute_routing(predicted_filtered, confidence, threshold)

        if routing == "auto_remove":
            st.error("🔴 Your review was removed automatically.")
        elif routing == "queue":
            if active_config.get("architecture") == "jev":
                st.warning("🟡 Jev's result is advisory and waiting for a human reviewer's decision; no automatic action was taken.")
            else:
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
