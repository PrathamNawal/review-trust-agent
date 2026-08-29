"""
Provider-agnostic tool definitions (plain JSON Schema). Each agent driver
(agent.py for Claude, agent_gemini.py for Gemini) adapts these into whatever
shape its SDK expects.
"""

TOOL_DEFS = [
    {
        "name": "reviewer_history_lookup",
        "description": (
            "Look up a reviewer's posting history: review count, average rating, "
            "rating spread, and burst timing. Call this for the reviewer under "
            "investigation, or for any other reviewer you want to cross-check "
            "(e.g. a reviewer who left similar-looking text at the same business)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "business_trend_lookup",
        "description": (
            "Look up a business's review volume and rating trend around a given date. "
            "A sudden spike in review velocity (burst_ratio well above 1) around the "
            "review's date is a classic sign of an incentivized or coordinated campaign."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prod_id": {"type": "string"},
                "around_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["prod_id", "around_date"],
        },
    },
    {
        "name": "text_similarity_check",
        "description": (
            "Compare a piece of review text against a reviewer's or a business's other "
            "reviews for templated or near-duplicate phrasing. Use this on the review "
            "under investigation, or on any other review text you've seen (e.g. from a "
            "second reviewer) to check whether they read like the same template."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "review_text": {"type": "string"},
                "compare_against": {"type": "string", "enum": ["reviewer", "business"]},
                "target_id": {"type": "string"},
            },
            "required": ["review_text", "compare_against", "target_id"],
        },
    },
    {
        "name": "submit_judgment",
        "description": "Submit your final judgment once you have enough evidence. Ends the investigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_filtered": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "primary_signal": {
                    "type": "string",
                    "enum": ["reviewer_pattern", "business_trend", "text_similarity", "combination"],
                },
                "reasoning": {"type": "string", "description": "1-2 sentences."},
            },
            "required": ["predicted_filtered", "confidence", "primary_signal", "reasoning"],
        },
    },
]

SYSTEM_PROMPT = """You are a trust & safety investigator checking whether a Yelp review is likely \
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


def dispatch_tool_call(toolbox, review, tool_name, tool_input):
    """Shared dispatch logic: excludes the review under investigation from its own
    comparison set, but only when the model's tool target IS that same review's
    reviewer/business — a lookup on a different id is never excluded."""
    exclude = review["review_id"]

    if tool_name == "reviewer_history_lookup":
        excl = exclude if tool_input["user_id"] == review["user_id"] else None
        return toolbox.reviewer_history_lookup(tool_input["user_id"], exclude_review_id=excl)

    if tool_name == "business_trend_lookup":
        excl = exclude if tool_input["prod_id"] == review["prod_id"] else None
        return toolbox.business_trend_lookup(
            tool_input["prod_id"], tool_input["around_date"], exclude_review_id=excl
        )

    if tool_name == "text_similarity_check":
        same_target = (
            (tool_input["compare_against"] == "reviewer" and tool_input["target_id"] == review["user_id"])
            or (tool_input["compare_against"] == "business" and tool_input["target_id"] == review["prod_id"])
        )
        excl = exclude if same_target else None
        return toolbox.text_similarity_check(
            tool_input["review_text"],
            tool_input["compare_against"],
            tool_input["target_id"],
            exclude_review_id=excl,
        )

    raise ValueError(f"Unknown tool: {tool_name}")


def format_review_prompt(review):
    return (
        "Investigate this review:\n"
        f"- reviewer_id: {review['user_id']}\n"
        f"- business_id: {review['prod_id']}\n"
        f"- rating: {review['rating']}\n"
        f"- date: {review['date']}\n"
        f"- text: {review['review_text']}"
    )
