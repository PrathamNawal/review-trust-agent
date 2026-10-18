"""Configuration, experiment, version history, and rollback view."""

import pandas as pd
import streamlit as st
import os

from dashboard.agent_architectures import available_architectures
from dashboard.config_store import DEFAULT_JEV_EVIDENCE_SOURCES, DEFAULT_JEV_PROMPTS
from dashboard.evaluation import compare_configurations, load_evaluation_reviews
from dashboard.live_agent import PROVIDER_PRESETS


JEV_EVIDENCE_LABELS = {
    "reviewer_history": "Reviewer activity",
    "business_trend": "Business review-volume trend",
    "reviewer_text_similarity": "Reviewer text reuse",
    "business_text_similarity": "Business text reuse",
}

QUICK_EXPERIMENTS = {
    "jev_choice": {
        "label": "🔎 Jev Choice",
        "help": "Use Jev's direct filtered/recommended label. No threshold is applied; all results stay in human review.",
        "changes": {
            "architecture": "jev", "provider": "typesafe", "model": "jev-1.13.0",
            "jev_method": "choice", "reason": "Try Jev Choice as a human-reviewed advisory label",
        },
    },
    "jev_score": {
        "label": "📏 Jev Score",
        "help": "Explore Jev's 0–1 ordinal score. The threshold is experimental and does not control human review.",
        "changes": {"architecture": "jev", "provider": "typesafe", "model": "jev-1.13.0",
                    "jev_method": "score", "jev_threshold": 0.5,
                    "reason": "Explore Jev Score with a 0.50 research threshold"},
    },
    "jev_noul": {
        "label": "🧭 Jev Noul",
        "help": "Explore Jev's 0–1 Noul output. It is not calibrated; the threshold is experimental.",
        "changes": {"architecture": "jev", "provider": "typesafe", "model": "jev-1.13.0",
                    "jev_method": "noul", "jev_threshold": 0.5,
                    "reason": "Explore Jev Noul with a 0.50 research threshold"},
    },
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
        jev_method=candidate.get("jev_method", "choice"),
        jev_threshold=candidate.get("jev_threshold", 0.5),
        jev_prompts=candidate.get("jev_prompts", DEFAULT_JEV_PROMPTS),
        jev_evidence_sources=candidate.get("jev_evidence_sources", DEFAULT_JEV_EVIDENCE_SOURCES),
    )


def render_configuration(config_store, active_config, mark_step_done, goto):
    st.title("Play — you're the PM now")
    st.caption("Configure a candidate, compare it on the same labeled examples, then deploy it as a version with one click. Earlier versions remain available for rollback.")

    prefill = st.session_state.get("config_prefill", {})
    defaults = {**active_config, **prefill}

    architectures = list(available_architectures()) or ["agno"]
    current_architecture = defaults.get("architecture", "agno")
    architecture = st.selectbox(
        "Agent architecture", architectures,
        index=architectures.index(current_architecture) if current_architecture in architectures else 0,
        key="selected_agent_architecture",
    )
    if architecture != current_architecture and prefill:
        st.session_state.pop("config_prefill", None)
        defaults = {**active_config}
    if architecture == "agno" and defaults.get("provider") == "typesafe":
        defaults.update({"provider": "groq", "model": "openai/gpt-oss-120b", "temperature": 0.2})

    experiments = ({key: value for key, value in QUICK_EXPERIMENTS.items()
                    if key.startswith("jev_")} if architecture == "jev" else
                   {key: value for key, value in QUICK_EXPERIMENTS.items()
                    if key in ("cautious", "fast", "claude")})
    st.markdown("**Guided starting points** — these prefill the candidate; they do not deploy until you choose a deploy button.")
    exp_cols = st.columns(len(experiments))
    for col, (key, exp) in zip(exp_cols, experiments.items()):
        with col:
            if st.button(exp["label"], help=exp["help"], key=f"exp_{key}"):
                st.session_state["config_prefill"] = {**prefill, **exp["changes"]}
                if "jev_method" in exp["changes"]:
                    st.session_state["jev_config_generation"] = st.session_state.get("jev_config_generation", 0) + 1
                st.rerun()

    if architecture == "jev":
        st.info(
            "Jev is experimental and advisory. Its outputs are not calibrated probabilities; "
            "every result stays in human review. Choice returns a label directly. Score and Noul "
            "use a research threshold to map a 0–1 output to a label. Changing a threshold does "
            "not change the human-review requirement. The tested model is pinned to jev-1.13.0. "
            "On Streamlit Community Cloud, deployed config versions may reset when the app restarts."
        )
        with st.expander("Measured reference results and how to use them"):
            st.markdown(
                "| Paired comparison | Method | Reviews | Recall | Accuracy |\n"
                "|---|---|---:|---:|---:|\n"
                "| Fresh interim holdout | Choice | 961 | 9.2% | 84.2% |\n"
                "| Fresh interim holdout | Noul, overall risk at 0.50 | 961 | 4.2% | 85.7% |\n"
                "| Development subset | Choice | 300 | 12.5% | 85.0% |\n"
                "| Development subset | Score at 0.50 | 300 | 12.5% | 81.7% |\n\n"
                "The two comparisons use different data partitions; compare methods only within a row pair. "
                "No method has met the joint ≥20% recall / ≥85% accuracy target on a completed, untouched evaluation. "
                "YelpZip's `filtered` label is a reference label, not verified fraud ground truth."
            )

    # Keep this selector outside the form: Streamlit forms do not rerun when a
    # widget changes, so method-specific controls would otherwise stay stale
    # until the user submits Evaluate or Deploy.
    if architecture == "jev":
        method_options = ["choice", "score", "noul"]
        jev_method = st.selectbox(
            "Decision method",
            method_options,
            index=method_options.index(defaults.get("jev_method", "choice")),
            format_func=lambda value: {"choice": "Choice · direct label",
                                       "score": "Score · ordinal 0–1",
                                       "noul": "Noul · model estimate 0–1"}[value],
            key=f"selected_jev_method_{st.session_state.get('jev_config_generation', 0)}",
            help="Compare methods on the same examples before deploying. These primitive comparisons are exploratory, not proof of a general performance winner.",
        )
        method_help = {
            "choice": "Jev selects Filtered or Recommended. Keep the label meanings fixed; edit the instructions that guide the decision.",
            "score": "Jev returns an ordinal score from the genuine-evidence to filtered-evidence rubric. The cutoff below is an experimental decision rule, not calibrated confidence.",
            "noul": "Jev returns a model-estimated value for the question below. It is not a calibrated fraud probability; the cutoff is an experimental decision rule.",
        }
        st.caption(method_help[jev_method])

    st.divider()
    with st.form("deploy_form"):
        if architecture == "jev":
            system_prompt = defaults["system_prompt"]
            provider, model, temperature = "typesafe", "jev-1.13.0", 0.0
            confidence_threshold = float(defaults["confidence_threshold"])
            st.caption("Model: `jev-1.13.0` (pinned for reproducibility). The current Jev client does not expose a temperature control.")
            prompts = {**DEFAULT_JEV_PROMPTS, **defaults.get("jev_prompts", {})}
            jev_prompts = dict(prompts)
            jev_prompts[jev_method] = st.text_area(
                "Decision instructions",
                value=prompts[jev_method],
                height=130,
                help="Tell Jev what evidence to weigh and how to treat missing or conflicting evidence. Change one idea at a time, compare on fixed examples, and avoid instructions that imply unsupported certainty.",
                key=f"jev_prompt_{active_config.get('version', 'v0')}_{jev_method}",
            )
            if jev_method == "choice":
                jev_threshold = float(defaults.get("jev_threshold", 0.5))
                st.caption("Choice has no cutoff. Jev's probabilities are shown as model output only and are not calibrated for Yelp fraud.")
            else:
                jev_threshold = st.slider(
                    "Research cutoff",
                    min_value=0.0, max_value=1.0,
                    value=float(defaults.get("jev_threshold", 0.5)), step=0.01,
                    help="Values at or above this cutoff map to Filtered. This changes the label mapping only; human review remains mandatory.",
                )

            st.markdown("**Evidence Jev receives**")
            st.caption("Unselected evidence is omitted from the model request. Missing evidence is treated as unknown; this is useful for controlled ablation, but can reduce signal.")
            configured_sources = set(defaults.get("jev_evidence_sources", DEFAULT_JEV_EVIDENCE_SOURCES))
            source_columns = st.columns(2)
            jev_evidence_sources = []
            for index, (source, label) in enumerate(JEV_EVIDENCE_LABELS.items()):
                with source_columns[index % 2]:
                    if st.checkbox(
                        label, value=source in configured_sources,
                        key=f"jev_source_{active_config.get('version', 'v0')}_{source}",
                    ):
                        jev_evidence_sources.append(source)
            if not jev_evidence_sources:
                st.warning("Select at least one evidence source before evaluating or deploying.")
        else:
            jev_method = defaults.get("jev_method", "choice")
            jev_threshold = float(defaults.get("jev_threshold", 0.5))
            jev_prompts = {**DEFAULT_JEV_PROMPTS, **defaults.get("jev_prompts", {})}
            jev_evidence_sources = list(defaults.get("jev_evidence_sources", DEFAULT_JEV_EVIDENCE_SOURCES))
            system_prompt = st.text_area("System prompt", value=defaults["system_prompt"], height=220)
            col1, col2, col3 = st.columns(3)
            with col1:
                providers = list(PROVIDER_PRESETS)
                default_provider = defaults.get("provider")
                if default_provider not in providers:
                    default_provider = "groq"
                provider = st.selectbox("Provider", providers,
                                         index=providers.index(default_provider))
            with col2:
                model = st.text_input("Model", value=defaults["model"])
            with col3:
                temperature = st.slider("Temperature", 0.0, 1.0, float(defaults["temperature"]), 0.05)

            confidence_threshold = st.slider(
                "Auto-remove confidence threshold", 0.0, 1.0,
                float(defaults["confidence_threshold"]), 0.05,
            )
        reason = st.text_input(
            "Reason for this change (optional — auto-filled from the diff if left blank)",
            value=defaults.get("reason", ""),
        )
        evaluate_candidate = st.form_submit_button("🔬 Evaluate candidate")
        deploy_now = st.form_submit_button("🚀 Deploy new version", type="primary")

    if (deploy_now or evaluate_candidate) and architecture == "jev" and not jev_evidence_sources:
        st.error("Select at least one evidence source before evaluating or deploying.")
        st.stop()
    if (deploy_now or evaluate_candidate) and architecture == "jev" and not jev_prompts.get(jev_method, "").strip():
        st.error("Add decision instructions before evaluating or deploying.")
        st.stop()

    if deploy_now:
        direct_candidate = {
            "system_prompt": system_prompt, "temperature": temperature,
            "provider": provider, "model": model,
            "confidence_threshold": confidence_threshold,
            "architecture": architecture, "reason": reason,
            "jev_method": jev_method, "jev_threshold": jev_threshold,
            "jev_prompts": jev_prompts, "jev_evidence_sources": jev_evidence_sources,
        }
        new_version = _deploy(config_store, direct_candidate)
        st.session_state["selected_agent_architecture"] = architecture
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
            "jev_method": jev_method, "jev_threshold": jev_threshold,
            "jev_prompts": jev_prompts, "jev_evidence_sources": jev_evidence_sources,
        }
        st.session_state.pop("candidate_evaluation", None)

    candidate = st.session_state.get("candidate_config")
    if candidate:
        st.divider()
        st.subheader("Candidate evaluation")
        st.caption("The candidate and active version run on the same fixed Yelp-labeled examples. "
                   "Each review makes one live agent investigation per version, so this uses API calls.")
        st.info("This small golden set is useful for a quick directional comparison, not final validation. "
                "Use a new untouched evaluation set before making broader performance claims.")
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
                    st.session_state["selected_agent_architecture"] = candidate["architecture"]
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
         "decision method": version.get("jev_method", "choice") if version.get("architecture") == "jev" else "Agno",
         "cutoff": ("—" if version.get("jev_method", "choice") == "choice" else version.get("jev_threshold"))
                   if version.get("architecture") == "jev" else version["confidence_threshold"],
         "Jev evidence": ", ".join(JEV_EVIDENCE_LABELS.get(source, source)
                                    for source in version.get("jev_evidence_sources", []))
                         if version.get("architecture") == "jev" else "—",
         "Jev prompt": version.get("jev_prompts", {}).get(version.get("jev_method", "choice"), "")[:90]
                      if version.get("architecture") == "jev" else "—",
         "reason": version["reason"], "deployed_at": version["deployed_at"]}
        for version in reversed(versions)
    ])
    st.dataframe(history, width="stretch")

    rollback_target = st.selectbox("Roll back to version", [version["version"] for version in reversed(versions)])
    if st.button("Roll back"):
        config_store.rollback_to(rollback_target)
        rolled_back = config_store.get_active_config()
        st.session_state["selected_agent_architecture"] = rolled_back.get("architecture", "agno")
        st.rerun()
