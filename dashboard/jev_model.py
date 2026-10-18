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


def classify_review(review, evidence, api_key=None, model="jev-latest", timeout=60,
                    instructions=None):
    """Return a typed binary decision and usage metadata; never log credentials."""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise JevAPIError("TYPESAFE_API_KEY is not configured.")
    state = {
        "review": {k: review.get(k) for k in ("rating", "date", "review_text")},
        "evidence": evidence,
    }
    payload = {
        "model": model,
        "state": state,
        "questions": {
            "review_disposition": {
                "type": "choice",
                "instructions": instructions or DEFAULT_DECISION_INSTRUCTIONS,
                "criteria": {
                    "filtered": "Evidence supports that this review would be filtered as potentially fake or manipulated.",
                    "recommended": "Evidence supports that this review would be recommended as likely genuine.",
                },
            }
        },
    }
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
            # Do not surface response body: it can contain request details.
            raise JevAPIError(f"Jev API returned HTTP {exc.code}.") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise JevAPIError(f"Jev request failed ({type(exc).__name__}).") from None
    else:
        raise JevAPIError("Jev API remained rate limited after retries.")

    answer = body.get("answers", {}).get("review_disposition", {})
    choice = answer.get("choice")
    if answer.get("type") != "choice" or choice not in ("filtered", "recommended"):
        raise JevAPIError("Jev returned an unexpected decision response.")
    probabilities = answer.get("probabilities", {})
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        confidence = probabilities.get(choice)
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        confidence = 0.0
    usage = body.get("usage", {})
    return {
        "predicted_filtered": choice == "filtered",
        "confidence": float(confidence),
        "probability_filtered": probabilities.get("filtered"),
        "model": body.get("model", model),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cost_usd": usage.get("cost_usd", usage.get("cost")),
        "elapsed_ms": body.get("elapsedMs", round((time.monotonic() - started) * 1000)),
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
