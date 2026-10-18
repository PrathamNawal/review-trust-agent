"""Configuration, experiment, version history, and rollback view."""

import pandas as pd
import streamlit as st
import os

from dashboard.agent_architectures import available_architectures
from dashboard.evaluation import compare_configurations, load_evaluation_reviews
from dashboard.live_agent import PROVIDER_PRESETS


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


def _deploy(config_store, candidate):
    return config_store.deploy_new_version(
        system_prompt=candidate["system_prompt"], temperature=candidate["temperature"],
        provider=candidate["provider"], model=candidate["model"],
        confidence_threshold=candidate["confidence_threshold"],
        reason=candidate.get("reason", ""), architecture=candidate["architecture"],
    )


def render_configuration(config_store, active_config, mark_step_done, goto):
    st.title("Play — you're the PM now")
    st.caption("Change the agent's system prompt, temperature, provider, or auto-remove threshold. One click deploys a new version — past versions stay comparable.")

    st.markdown("**Quick experiments** — one click pre-fills the form below. You can deploy directly or evaluate first:")
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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            architectures = list(available_architectures()) or ["agno"]
            current_architecture = defaults.get("architecture", "agno")
            architecture = st.selectbox("Agent architecture", architectures,
                                        index=(architectures.index(current_architecture)
                                               if current_architecture in architectures else 0))
        with col2:
            providers = list(PROVIDER_PRESETS)
            provider = st.selectbox("Provider", providers,
                                     index=providers.index(defaults["provider"]))
        with col3:
            model = st.text_input("Model", value=defaults["model"])
        with col4:
            temperature = st.slider("Temperature", 0.0, 1.0, float(defaults["temperature"]), 0.05)

        confidence_threshold = st.slider("Auto-remove confidence threshold", 0.0, 1.0,
                                          float(defaults["confidence_threshold"]), 0.05)
        reason = st.text_input("Reason for this change (optional — auto-filled from the diff if left blank)")
        evaluate_candidate = st.form_submit_button("🔬 Evaluate candidate")
        deploy_now = st.form_submit_button("🚀 Deploy new version", type="primary")

    if deploy_now:
        direct_candidate = {
            "system_prompt": system_prompt, "temperature": temperature,
            "provider": provider, "model": model,
            "confidence_threshold": confidence_threshold,
            "architecture": architecture, "reason": reason,
        }
        new_version = _deploy(config_store, direct_candidate)
        st.session_state.pop("candidate_config", None)
        st.session_state.pop("candidate_evaluation", None)
        st.session_state.pop("config_prefill", None)
        mark_step_done(6)
        st.session_state["just_deployed"] = new_version
        st.rerun()

    if evaluate_candidate:
        st.session_state["candidate_config"] = {
            "system_prompt": system_prompt, "temperature": temperature,
            "provider": provider, "model": model,
            "confidence_threshold": confidence_threshold,
            "architecture": architecture, "version": "candidate", "reason": reason,
        }
        st.session_state.pop("candidate_evaluation", None)

    candidate = st.session_state.get("candidate_config")
    if candidate:
        st.divider()
        st.subheader("Candidate evaluation")
        st.caption("The candidate and active version run on the same fixed Yelp-labeled examples. "
                   "Each review makes one live agent investigation per version, so this uses API calls.")
        golden_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "golden_set.csv")
        golden_reviews = load_evaluation_reviews(golden_path)
        sample_sizes = sorted({4, 8, 16, 32, len(golden_reviews)})
        n_reviews = st.selectbox(
            "Evaluation examples", sample_sizes, index=0,
            format_func=lambda n: f"Full set ({n})" if n == len(golden_reviews) else str(n),
            help="A larger set gives a steadier comparison and makes more model calls.",
            key="candidate_eval_count",
        )
        if st.button("Run baseline and candidate evaluation", type="primary"):
            fixed_reviews = golden_reviews[:n_reviews]
            with st.spinner(f"Running {n_reviews * 2} investigations on the same examples…"):
                st.session_state["candidate_evaluation"] = compare_configurations(
                    active_config, candidate, fixed_reviews
                )

        evaluation = st.session_state.get("candidate_evaluation")
        if evaluation:
            before = evaluation["baseline"]["metrics"]
            after = evaluation["candidate"]["metrics"]
            compare = pd.DataFrame([
                {"version": "Active", **{k: before[k] for k in ("completed", "completion_rate", "recall", "precision", "f1", "balanced_accuracy", "accuracy")}},
                {"version": "Candidate", **{k: after[k] for k in ("completed", "completion_rate", "recall", "precision", "f1", "balanced_accuracy", "accuracy")}},
                {"version": "Change", "completed": after["completed"] - before["completed"],
                 **evaluation["delta"]},
            ])
            st.dataframe(compare, hide_index=True, width="stretch")
            per_review = evaluation["baseline"]["rows"].merge(
                evaluation["candidate"]["rows"], on=["review_id", "true_filtered"],
                suffixes=("_active", "_candidate"),
            )
            st.caption("Per-review outcomes (filtered = Yelp's reference label)")
            st.dataframe(per_review[[
                "review_id", "true_filtered", "predicted_filtered_active",
                "predicted_filtered_candidate", "correct_active", "correct_candidate",
                "error_active", "error_candidate",
            ]], hide_index=True, width="stretch")
            st.caption("Fraud recall is the share of Yelp-filtered examples the agent flagged. Yelp labels are a reference signal, not verified fraud ground truth.")
            expected = len(evaluation["review_ids"])
            if before["completed"] < expected or after["completed"] < expected:
                st.warning(f"Active completed {before['completed']}/{expected}; candidate completed {after['completed']}/{expected}. Resolve provider errors or incomplete runs before deploying.")
            else:
                recall_delta = evaluation["delta"]["recall"]
                st.metric("Fraud recall change", f"{after['recall']:.0%}" if after["recall"] is not None else "Undefined",
                          f"{recall_delta:+.0%}" if recall_delta is not None else None)
                if st.button("🚀 Deploy evaluated candidate", type="primary"):
                    new_version = _deploy(config_store, candidate)
                    st.session_state.pop("candidate_config", None)
                    st.session_state.pop("candidate_evaluation", None)
                    st.session_state.pop("config_prefill", None)
                    mark_step_done(6)
                    st.session_state["just_deployed"] = new_version
                    st.rerun()

    if st.session_state.get("just_deployed"):
        deployed_config = st.session_state["just_deployed"]
        st.success(f"Deployed **{deployed_config['version']}**: {deployed_config['reason']}")
        if st.button("See how this changed performance →", type="primary"):
            st.session_state.pop("just_deployed", None)
            goto("8. Track Performance")

    st.divider()
    st.subheader("Version history")
    versions = config_store.list_versions()
    history = pd.DataFrame([
        {"version": version["version"], "architecture": version.get("architecture", "agno"),
         "provider": version["provider"], "model": version["model"],
         "temperature": version["temperature"], "threshold": version["confidence_threshold"],
         "reason": version["reason"], "deployed_at": version["deployed_at"]}
        for version in reversed(versions)
    ])
    st.dataframe(history, width="stretch")

    rollback_target = st.selectbox("Roll back to version", [version["version"] for version in reversed(versions)])
    if st.button("Roll back"):
        config_store.rollback_to(rollback_target)
        st.rerun()
