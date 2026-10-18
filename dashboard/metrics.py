"""Pure binary-classification metrics used by evaluation and tracking."""


def binary_classification_metrics(true_labels, predicted_labels):
    """Compute fraud-positive metrics from aligned boolean-like labels.

    The filtered/fraud class is treated as positive. Undefined denominators are
    returned as None so callers can display them honestly instead of as zero.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal length")

    tp = tn = fp = fn = 0
    for actual, predicted in zip(true_labels, predicted_labels):
        actual, predicted = bool(actual), bool(predicted)
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1

    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1_denom = 2 * tp + fp + fn
    f1 = (2 * tp / f1_denom) if f1_denom else None
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None else None
    )
    return {
        "n": n,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def wilson_interval(successes, n, z=1.96):
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
