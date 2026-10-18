"""
Persisted agent-config version history. Each "deploy" appends a new version
and moves the active pointer — real persistence to disk (JSON), not a
session-only variable, so config survives across dashboard restarts.
"""

import json
import os
from datetime import datetime, timezone

from scripts.tool_schemas import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from dashboard.jev_model import (
    DECISION_METHODS,
    DEFAULT_DECISION_INSTRUCTIONS,
    DEFAULT_NOUL_INSTRUCTIONS,
    DEFAULT_SCORE_INSTRUCTIONS,
)


DEFAULT_JEV_PROMPTS = {
    "choice": DEFAULT_DECISION_INSTRUCTIONS,
    "score": DEFAULT_SCORE_INSTRUCTIONS,
    "noul": DEFAULT_NOUL_INSTRUCTIONS,
}
DEFAULT_JEV_EVIDENCE_SOURCES = [
    "reviewer_history", "business_trend", "reviewer_text_similarity",
    "business_text_similarity",
]
JEV_EVIDENCE_SOURCE_KEYS = set(DEFAULT_JEV_EVIDENCE_SOURCES)

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agent_configs.json")

DEFAULT_CONFIG = {
    "version": "v0",
    "architecture": "jev",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.0,
    "provider": "typesafe",
    "model": "jev-1.13.0",
    "confidence_threshold": 0.75,
    "jev_method": "choice",
    "jev_threshold": 0.5,
    "jev_prompts": DEFAULT_JEV_PROMPTS,
    "jev_evidence_sources": DEFAULT_JEV_EVIDENCE_SOURCES,
    "reason": "Default to experimental Jev Choice; all outputs require human review.",
    "deployed_at": None,
}


def _load_raw():
    if not os.path.exists(STORE_PATH):
        return {"versions": [DEFAULT_CONFIG], "active": "v0"}
    with open(STORE_PATH) as f:
        return json.load(f)


def _save_raw(data):
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_active_config():
    data = _load_raw()
    active_version = data["active"]
    for v in data["versions"]:
        if v["version"] == active_version:
            return _normalize_config(v)
    fallback = data["versions"][-1]
    return _normalize_config(fallback)


def list_versions():
    return [_normalize_config(v) for v in _load_raw()["versions"]]


def _normalize_config(version):
    config = {**DEFAULT_CONFIG, **version, "architecture": version.get("architecture", "agno")}
    config["jev_prompts"] = {**DEFAULT_JEV_PROMPTS, **version.get("jev_prompts", {})}
    config["jev_evidence_sources"] = list(
        version.get("jev_evidence_sources", DEFAULT_JEV_EVIDENCE_SOURCES)
    )
    return config


def _summarize_diff(old, new):
    """Auto-fill a deploy reason from what actually changed, if the user left it blank."""
    changes = []
    for key, label in [
        ("system_prompt", "system prompt"),
        ("architecture", "architecture"),
        ("temperature", "temperature"),
        ("provider", "provider"),
        ("model", "model"),
        ("confidence_threshold", "confidence threshold"),
        ("jev_method", "Jev decision method"),
        ("jev_threshold", "Jev score threshold"),
        ("jev_prompts", "Jev instructions"),
        ("jev_evidence_sources", "Jev evidence sources"),
    ]:
        if old.get(key) != new.get(key):
            if key == "system_prompt":
                changes.append("system prompt edited")
            else:
                changes.append(f"{label} {old.get(key)} -> {new.get(key)}")
    return "; ".join(changes) if changes else "No parameter changes."


def deploy_new_version(system_prompt, temperature, provider, model, confidence_threshold,
                       reason="", architecture="agno", jev_method="choice",
                       jev_threshold=0.5, jev_prompts=None,
                       jev_evidence_sources=None):
    """Appends a new config version and makes it active. Returns the new version dict."""
    if architecture == "jev":
        if jev_method not in DECISION_METHODS:
            raise ValueError(f"Unsupported Jev decision method: {jev_method}")
        selected_sources = set(
            DEFAULT_JEV_EVIDENCE_SOURCES if jev_evidence_sources is None
            else jev_evidence_sources
        )
        if not selected_sources or not selected_sources.issubset(JEV_EVIDENCE_SOURCE_KEYS):
            raise ValueError("Select one or more supported Jev evidence sources.")
        instructions = {**DEFAULT_JEV_PROMPTS, **(jev_prompts or {})}.get(jev_method, "")
        if not instructions.strip():
            raise ValueError("Jev decision instructions cannot be blank.")
        if not 0 <= float(jev_threshold) <= 1:
            raise ValueError("Jev research cutoff must be between 0 and 1.")

    data = _load_raw()
    current = get_active_config()

    next_num = len(data["versions"])
    new_version = {
        "version": f"v{next_num}",
        "architecture": architecture,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "provider": provider,
        "model": model,
        "confidence_threshold": confidence_threshold,
        "jev_method": jev_method,
        "jev_threshold": float(jev_threshold),
        "jev_prompts": {**DEFAULT_JEV_PROMPTS, **(jev_prompts or {})},
        "jev_evidence_sources": list(
            DEFAULT_JEV_EVIDENCE_SOURCES if jev_evidence_sources is None
            else jev_evidence_sources
        ),
        "reason": reason.strip() or _summarize_diff(current, {
            "system_prompt": system_prompt, "architecture": architecture, "temperature": temperature,
            "provider": provider, "model": model, "confidence_threshold": confidence_threshold,
            "jev_method": jev_method, "jev_threshold": float(jev_threshold),
            "jev_prompts": {**DEFAULT_JEV_PROMPTS, **(jev_prompts or {})},
            "jev_evidence_sources": list(
                DEFAULT_JEV_EVIDENCE_SOURCES if jev_evidence_sources is None
                else jev_evidence_sources
            ),
        }),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }

    data["versions"].append(new_version)
    data["active"] = new_version["version"]
    _save_raw(data)
    return new_version


def rollback_to(version):
    data = _load_raw()
    if not any(v["version"] == version for v in data["versions"]):
        raise ValueError(f"Unknown version: {version}")
    data["active"] = version
    _save_raw(data)
