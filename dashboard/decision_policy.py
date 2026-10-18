"""Pure decision-policy rules shared by the dashboard and evaluation code."""

ROUTING_LABELS = {
    "auto_remove": "🔴 Auto-remove",
    "queue": "🟡 Routed to human queue",
    "ignore": "🟢 Ignored",
}

ROUTING_TONE = {"auto_remove": "danger", "queue": "warning", "ignore": "safe"}


def compute_routing(predicted_filtered, confidence, threshold):
    """Map an agent judgment to the existing three-tier autonomy policy."""
    if predicted_filtered and confidence >= threshold:
        return "auto_remove"
    if not predicted_filtered and confidence >= threshold:
        return "ignore"
    return "queue"
