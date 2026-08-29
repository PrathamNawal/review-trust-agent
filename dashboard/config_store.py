"""
Persisted agent-config version history. Each "deploy" appends a new version
and moves the active pointer — real persistence to disk (JSON), not a
session-only variable, so config survives across dashboard restarts.
"""

import json
import os
from datetime import datetime, timezone

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agent_configs.json")

DEFAULT_SYSTEM_PROMPT = """You are a trust & safety investigator checking whether a Yelp review is likely \
fake/manipulated (would be filtered) or genuine (would be recommended).

You have three evidence-gathering tools plus a submit_judgment tool to end the investigation. \
You do not have to call all three tools, and you are not limited to calling each one once. If \
the evidence so far is ambiguous, pull more of it before deciding — for example, if the review's \
own text looks templated, check a *different* reviewer at the same business to see if their \
language matches too (a real coordinated campaign), or if a reviewer's history looks bursty, \
check the business's trend for the same time window to see if the burst lines up.

Only call submit_judgment when you're confident you've gathered enough evidence to justify your \
answer. If the signals genuinely conflict, your confidence should reflect that — don't manufacture \
false certainty."""

DEFAULT_CONFIG = {
    "version": "v0",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.2,
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "confidence_threshold": 0.75,
    "reason": "Initial baseline configuration.",
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
            return v
    return data["versions"][-1]


def list_versions():
    return _load_raw()["versions"]


def _summarize_diff(old, new):
    """Auto-fill a deploy reason from what actually changed, if the user left it blank."""
    changes = []
    for key, label in [
        ("system_prompt", "system prompt"),
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


def deploy_new_version(system_prompt, temperature, provider, model, confidence_threshold, reason=""):
    """Appends a new config version and makes it active. Returns the new version dict."""
    data = _load_raw()
    current = get_active_config()

    next_num = len(data["versions"])
    new_version = {
        "version": f"v{next_num}",
        "system_prompt": system_prompt,
        "temperature": temperature,
        "provider": provider,
        "model": model,
        "confidence_threshold": confidence_threshold,
        "reason": reason.strip() or _summarize_diff(current, {
            "system_prompt": system_prompt, "temperature": temperature,
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
