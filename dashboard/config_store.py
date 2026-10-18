"""
Persisted agent-config version history. Each "deploy" appends a new version
and moves the active pointer — real persistence to disk (JSON), not a
session-only variable, so config survives across dashboard restarts.
"""

import json
import os
from datetime import datetime, timezone

from scripts.tool_schemas import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agent_configs.json")

DEFAULT_CONFIG = {
    "version": "v0",
    "architecture": "jev",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.0,
    "provider": "typesafe",
    "model": "jev-1.13.0",
    "confidence_threshold": 0.75,
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
            return {**DEFAULT_CONFIG, **v, "architecture": v.get("architecture", "agno")}
    fallback = data["versions"][-1]
    return {**DEFAULT_CONFIG, **fallback, "architecture": fallback.get("architecture", "agno")}


def list_versions():
    return [{**DEFAULT_CONFIG, **v, "architecture": v.get("architecture", "agno")}
            for v in _load_raw()["versions"]]


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
    ]:
        if old.get(key) != new.get(key):
            if key == "system_prompt":
                changes.append("system prompt edited")
            else:
                changes.append(f"{label} {old.get(key)} -> {new.get(key)}")
    return "; ".join(changes) if changes else "No parameter changes."


def deploy_new_version(system_prompt, temperature, provider, model, confidence_threshold,
                       reason="", architecture="agno"):
    """Appends a new config version and makes it active. Returns the new version dict."""
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
        "reason": reason.strip() or _summarize_diff(current, {
            "system_prompt": system_prompt, "architecture": architecture, "temperature": temperature,
            "provider": provider, "model": model, "confidence_threshold": confidence_threshold,
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
