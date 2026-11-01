"""Jev architecture adapter for Streamlit's shared investigation contract."""

import os
from pathlib import Path

from dashboard.agent_architectures import register_architecture
from dashboard.config_store import DEFAULT_JEV_EVIDENCE_SOURCES, DEFAULT_JEV_PROMPTS
from dashboard.evidence import MAX_SIMILARITY_TEXTS, build_time_aware_evidence
from dashboard.jev_model import DECISION_METHODS, JevAPIError, classify_review
from dashboard.live_agent import get_toolbox


MODEL_ID = "jev-1.13.0"


def get_typesafe_api_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("TYPESAFE_API_KEY")
        if key:
            return key
    except Exception:
        pass
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("TYPESAFE_API_KEY="):
                return stripped.split("=", 1)[1].strip().strip("\"'") or None
    return None


def run_jev_investigation_stream(review, config):
    """Use Jev Choice as an advisory label; every result routes to a human."""
    api_key = get_typesafe_api_key()
    if not api_key:
        yield {"type": "error", "message": "TYPESAFE_API_KEY is not set. Add it in the sidebar or Streamlit secrets."}
        return

    try:
        yield {"type": "turn", "turn": 0}
        toolbox = get_toolbox()
        selected_sources = config.get("jev_evidence_sources", DEFAULT_JEV_EVIDENCE_SOURCES)
        if not selected_sources:
            raise ValueError("Select at least one Jev evidence source in System Config.")
        all_evidence = build_time_aware_evidence(toolbox, review, selected_sources)
        unknown_sources = set(selected_sources).difference(all_evidence)
        if unknown_sources:
            raise ValueError("The deployed Jev config contains an unknown evidence source.")
        evidence = {key: all_evidence[key] for key in selected_sources}
        event_meta = {
            "reviewer_history": ("reviewer_history_lookup", {"user_id": review["user_id"]}),
            "business_trend": ("business_trend_lookup", {
                "prod_id": review["prod_id"], "around_date": review["date"]}),
            "reviewer_text_similarity": ("text_similarity_check", {
                "compare_against": "reviewer", "target_id": review["user_id"]}),
            "business_text_similarity": ("text_similarity_check", {
                "compare_against": "business", "target_id": review["prod_id"]}),
        }
        for key, (tool_name, tool_input) in event_meta.items():
            if key not in evidence:
                continue
            yield {"type": "tool_call", "turn": 0, "tool": tool_name,
                   "input": tool_input, "result": evidence[key]}

        method = config.get("jev_method", "choice")
        if method not in DECISION_METHODS:
            raise ValueError("The deployed Jev config has an unsupported decision method.")
        prompts = {**DEFAULT_JEV_PROMPTS, **config.get("jev_prompts", {})}
        threshold = float(config.get("jev_threshold", 0.5))
        result = classify_review(
            review, evidence, api_key=api_key, model=MODEL_ID,
            instructions=prompts[method], method=method, threshold=threshold,
        )
        predicted = result["predicted_filtered"]
        choice_probabilities = result.get("choice_probabilities", {})
        method_label = method.title()
        if method == "choice":
            decision_summary = f"Jev Choice selected {'filtered' if predicted else 'recommended'}"
        else:
            decision_summary = (
                f"Jev {method_label} returned {result['decision_value']:.3f}; the configured "
                f"threshold is {threshold:.2f}"
            )
        reasoning = (
            f"{decision_summary} after reviewing {len(evidence)} selected evidence source(s). "
            "This experimental output is not a calibrated fraud probability; a human must review it."
        )
        yield {
            "type": "final",
            "predicted_filtered": predicted,
            "confidence": None,
            "confidence_calibrated": False,
            "primary_signal": f"Jev {method_label}",
            "reasoning": reasoning,
            "turns_used": 1,
            "jev_choice": ("filtered" if predicted else "recommended") if method == "choice" else None,
            "jev_choice_probabilities": choice_probabilities,
            "jev_method": method,
            "jev_decision_value": result.get("decision_value"),
            "jev_threshold": result.get("decision_threshold"),
            "model": result.get("model", MODEL_ID),
        }
    except JevAPIError as exc:
        yield {"type": "error", "message": str(exc)}
    except Exception as exc:
        yield {"type": "error", "message": f"Jev investigation failed ({type(exc).__name__})."}


_time_aware_evidence = build_time_aware_evidence  # backwards-compatible alias


register_architecture("jev", run_jev_investigation_stream)
