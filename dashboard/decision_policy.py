"""Pure decision-policy rules shared by the dashboard and evaluation code."""

import ast

ROUTING_LABELS = {
    "auto_remove": "🔴 Auto-remove",
    "queue": "🟡 Routed to human queue",
    "ignore": "🟢 Ignored",
}

ROUTING_TONE = {"auto_remove": "danger", "queue": "warning", "ignore": "safe"}


def compute_routing(predicted_filtered, confidence, threshold):
    """Map an agent judgment to the existing three-tier autonomy policy."""
    if confidence is None or confidence != confidence:
        return "queue"
    if predicted_filtered and confidence >= threshold:
        return "auto_remove"
    if not predicted_filtered and confidence >= threshold:
        return "ignore"
    return "queue"


def _as_unit_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0.0 <= number <= 1.0 else None  # NaN fails the comparison too


def filtered_score(method, choice_probabilities=None, decision_value=None):
    """Jev's score that a review is 'filtered' (0-1), read back from what was logged, or None.

    Choice logs the probability of each label, so use the 'filtered' entry — the logged
    decision value is the probability of whichever label Jev picked, which would rank
    'recommended' picks backwards. Score and Noul log a 0-1 value that is already
    the filtered score. The score is uncalibrated: it is for ordering only.
    """
    if method == "choice":
        if isinstance(choice_probabilities, str):
            try:
                choice_probabilities = ast.literal_eval(choice_probabilities)
            except (ValueError, SyntaxError):
                return None
        if not isinstance(choice_probabilities, dict):
            return None
        return _as_unit_float(choice_probabilities.get("filtered"))
    if method in ("score", "noul"):
        return _as_unit_float(decision_value)
    return None


def order_queue(pending_appeals, judged_ids, scores):
    """Appeals first, as given; then the rest by score, highest first, unscored last, ties by id."""
    appealed = set(pending_appeals)
    rest = [rid for rid in judged_ids if rid not in appealed]
    rest.sort(key=lambda rid: (scores.get(rid) is None, -(scores.get(rid) or 0.0), rid))
    return list(pending_appeals) + rest
