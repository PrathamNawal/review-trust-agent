"""Minimal server-side client for Jev Model's typed decision API."""

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class JevAPIError(RuntimeError):
    pass


DEFAULT_DECISION_INSTRUCTIONS = (
    "Using the review and evidence, estimate whether Yelp would filter this review. "
    "Judge the evidence, not writing style alone. Treat missing evidence as unknown. "
    "Do not infer from any dataset labels or identifiers."
)

DEFAULT_SCORE_INSTRUCTIONS = (
    "Score how strongly the supplied behavioral evidence supports filtering this review. "
    "Use the evidence, not writing style, rating, or sentiment alone. Treat missing evidence "
    "as unknown. Do not infer from dataset labels or identifiers."
)

DEFAULT_NOUL_INSTRUCTIONS = (
    "Estimate whether Yelp would filter this review based on the supplied behavioral evidence. "
    "Treat the value as a model estimate, not a calibrated fraud probability. Treat missing "
    "evidence as unknown. Do not infer from dataset labels or identifiers."
)

DECISION_METHODS = ("choice", "score", "noul")


DECOMPOSED_QUESTIONS = {
    "overall_risk": {
        "type": "noul",
        "instructions": (
            "Does the combined evidence support that Yelp would filter this review as "
            "potentially manipulated? Use only specific behavioral evidence in `evidence`. "
            "Do not infer from rating or sentiment alone. Treat missing evidence as unknown."
        ),
    },
    "reviewer_activity": {
        "type": "noul",
        "instructions": (
            "Does `evidence.reviewer_history` show an unusually dense or coordinated "
            "review-posting cadence? Consider the review count, minimum gaps, and maximum "
            "reviews in any seven-day window. Missing history is not suspicious by itself."
        ),
    },
    "business_burst": {
        "type": "noul",
        "instructions": (
            "Does `evidence.business_trend` show an abrupt review-volume increase around "
            "this review date, considering both `burst_ratio` and `reviews_in_window`? "
            "Do not treat a small count or ordinary sustained activity as a burst."
        ),
    },
    "reviewer_text_reuse": {
        "type": "noul",
        "instructions": (
            "Does `evidence.reviewer_text_similarity` show materially repeated wording "
            "across this review and the reviewer's earlier reviews? Consider similarity "
            "and the comparison count; missing comparisons are unknown."
        ),
    },
    "business_text_reuse": {
        "type": "noul",
        "instructions": (
            "Does `evidence.business_text_similarity` show materially repeated wording "
            "across this review and earlier reviews for the business? Consider similarity "
            "and the comparison count; missing comparisons are unknown."
        ),
    },
}


def _post_system_one(state, questions, api_key, model, timeout):
    payload = {"model": model, "state": state, "questions": questions}
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body, round((time.monotonic() - started) * 1000)
        except HTTPError as exc:
            if exc.code in (429, 529) and attempt < 3:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise JevAPIError(f"Jev API returned HTTP {exc.code}.") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise JevAPIError(f"Jev request failed ({type(exc).__name__}).") from None
    raise JevAPIError("Jev API remained rate limited after retries.")


def classify_review(review, evidence, api_key=None, model="jev-latest", timeout=60,
                    instructions=None, method="choice", threshold=0.5):
    """Return a configured Jev decision and usage metadata; never log credentials."""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise JevAPIError("TYPESAFE_API_KEY is not configured.")
    state = {
        "review": {k: review.get(k) for k in ("rating", "date", "review_text")},
        "evidence": evidence,
    }
    if method not in DECISION_METHODS:
        raise ValueError(f"Unsupported Jev decision method: {method}")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("Jev decision threshold must be between 0 and 1.")

    if method == "choice":
        question = {
            "type": "choice",
            "instructions": instructions or DEFAULT_DECISION_INSTRUCTIONS,
            "criteria": {
                "filtered": "Evidence supports that this review would be filtered as potentially fake or manipulated.",
                "recommended": "Evidence supports that this review would be recommended as likely genuine.",
            },
        }
    elif method == "score":
        question = {
            "type": "score",
            "instructions": instructions or DEFAULT_SCORE_INSTRUCTIONS,
            "criteria": [
                "The evidence supports recommending this review as likely genuine.",
                "The evidence supports filtering this review as potentially fake or manipulated.",
            ],
        }
    else:
        question = {
            "type": "noul",
            "instructions": instructions or DEFAULT_NOUL_INSTRUCTIONS,
        }
    questions = {"review_disposition": question}
    body, elapsed_ms = _post_system_one(state, questions, api_key, model, timeout)
    answer = body.get("answers", {}).get("review_disposition", {})
    probabilities = {}
    if method == "choice":
        choice = answer.get("choice")
        if answer.get("type") != "choice" or choice not in ("filtered", "recommended"):
            raise JevAPIError("Jev returned an unexpected Choice response.")
        predicted_filtered = choice == "filtered"
        probabilities = answer.get("probabilities", {})
        decision_value = probabilities.get(choice)
    else:
        value_key = "score" if method == "score" else "noul"
        decision_value = answer.get(value_key)
        if (answer.get("type") != method or not isinstance(decision_value, (int, float))
                or not 0 <= decision_value <= 1):
            raise JevAPIError(f"Jev returned an unexpected {method.title()} response.")
        predicted_filtered = float(decision_value) >= float(threshold)
        probabilities = answer.get("probabilities", {})
    usage = body.get("usage", {})
    return {
        "predicted_filtered": predicted_filtered,
        "confidence": None,
        "choice_probabilities": probabilities if method == "choice" else {},
        "decision_method": method,
        "decision_value": float(decision_value) if isinstance(decision_value, (int, float)) else None,
        "decision_threshold": float(threshold) if method != "choice" else None,
        "model": body.get("model", model),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": usage.get("cost_usd", usage.get("cost")),
        "elapsed_ms": body.get("elapsedMs", elapsed_ms),
    }


def compare_choice_and_score(review, evidence, api_key=None, model="jev-1.13.0", timeout=60):
    """Ask Jev for a binary Choice and ordinal Score over the same review state."""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise JevAPIError("TYPESAFE_API_KEY is not configured.")
    state = {
        "review": {k: review.get(k) for k in ("rating", "date", "review_text")},
        "evidence": evidence,
    }
    questions = {
        "review_disposition": {
            "type": "choice",
            "instructions": DEFAULT_DECISION_INSTRUCTIONS,
            "criteria": {
                "filtered": "Evidence supports that this review would be filtered as potentially fake or manipulated.",
                "recommended": "Evidence supports that this review would be recommended as likely genuine.",
            },
        },
        "risk_score": {
            "type": "score",
            "instructions": (
                "Score how strongly the supplied behavioral evidence supports filtering this review. "
                "Use the rubric endpoints only; do not infer from sentiment or rating alone. "
                "Treat missing evidence as unknown."
            ),
            "criteria": [
                "The evidence supports recommending this review as likely genuine.",
                "The evidence supports filtering this review as potentially fake or manipulated.",
            ],
        },
    }
    body, elapsed_ms = _post_system_one(state, questions, api_key, model, timeout)
    answers = body.get("answers", {})
    choice_answer = answers.get("review_disposition", {})
    choice = choice_answer.get("choice")
    if choice_answer.get("type") != "choice" or choice not in ("filtered", "recommended"):
        raise JevAPIError("Jev returned an unexpected Choice response.")
    score_answer = answers.get("risk_score", {})
    score = score_answer.get("score")
    if score_answer.get("type") != "score" or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        raise JevAPIError("Jev returned an unexpected Score response.")
    usage = body.get("usage", {})
    return {
        "choice_filtered": choice == "filtered",
        "choice_confidence": choice_answer.get("confidence"),
        "choice_probabilities": choice_answer.get("probabilities", {}),
        "score": float(score),
        "score_probabilities": score_answer.get("probabilities", {}),
        "score_confidence": score_answer.get("confidence"),
        "model": body.get("model", model),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": usage.get("cost_usd", usage.get("cost")),
        "elapsed_ms": body.get("elapsedMs", elapsed_ms),
    }


def compare_choice_and_noul(review, evidence, api_key=None, model="jev-1.13.0", timeout=60):
    """Ask Choice plus direct-risk and business-burst Noul questions on one state."""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise JevAPIError("TYPESAFE_API_KEY is not configured.")
    state = {
        "review": {k: review.get(k) for k in ("rating", "date", "review_text")},
        "evidence": evidence,
    }
    questions = {
        "review_disposition": {
            "type": "choice",
            "instructions": DEFAULT_DECISION_INSTRUCTIONS,
            "criteria": {
                "filtered": "Evidence supports that this review would be filtered as potentially fake or manipulated.",
                "recommended": "Evidence supports that this review would be recommended as likely genuine.",
            },
        },
        "overall_risk": DECOMPOSED_QUESTIONS["overall_risk"],
        "business_burst": DECOMPOSED_QUESTIONS["business_burst"],
    }
    body, elapsed_ms = _post_system_one(state, questions, api_key, model, timeout)
    answers = body.get("answers", {})
    choice_answer = answers.get("review_disposition", {})
    choice = choice_answer.get("choice")
    if choice_answer.get("type") != "choice" or choice not in ("filtered", "recommended"):
        raise JevAPIError("Jev returned an unexpected Choice response.")
    noul_values = {}
    for question_id in ("overall_risk", "business_burst"):
        answer = answers.get(question_id, {})
        value = answer.get("noul") if answer.get("type") == "noul" else None
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise JevAPIError(f"Jev returned an unexpected {question_id} response.")
        noul_values[question_id] = float(value)
    usage = body.get("usage", {})
    return {
        "choice_filtered": choice == "filtered",
        "choice_confidence": choice_answer.get("confidence"),
        "choice_probabilities": choice_answer.get("probabilities", {}),
        "noul_overall_risk": noul_values["overall_risk"],
        "noul_business_burst": noul_values["business_burst"],
        "model": body.get("model", model),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": usage.get("cost_usd", usage.get("cost")),
        "elapsed_ms": body.get("elapsedMs", elapsed_ms),
    }


def classify_review_dimensions(review, evidence, api_key=None, model="jev-1.13.0", timeout=60):
    """Return one overall Noul and four independent evidence-factor probabilities."""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise JevAPIError("TYPESAFE_API_KEY is not configured.")
    state = {
        "review": {k: review.get(k) for k in ("rating", "date", "review_text")},
        "evidence": evidence,
    }
    payload = {"model": model, "state": state, "questions": DECOMPOSED_QUESTIONS}
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            if exc.code in (429, 529) and attempt < 3:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise JevAPIError(f"Jev API returned HTTP {exc.code}.") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise JevAPIError(f"Jev request failed ({type(exc).__name__}).") from None
    else:
        raise JevAPIError("Jev API remained rate limited after retries.")

    answers = body.get("answers", {})
    probabilities = {}
    for question_id in DECOMPOSED_QUESTIONS:
        answer = answers.get(question_id, {})
        value = answer.get("noul") if answer.get("type") == "noul" else None
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise JevAPIError(f"Jev returned an unexpected {question_id} decision response.")
        probabilities[question_id] = float(value)
    usage = body.get("usage", {})
    signals = {k: v for k, v in probabilities.items() if k != "overall_risk"}
    return {
        "probabilities": probabilities,
        "max_signal_probability": max(signals.values()),
        "mean_signal_probability": sum(signals.values()) / len(signals),
        "model": body.get("model", model),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "elapsed_ms": body.get("elapsedMs", round((time.monotonic() - started) * 1000)),
    }
