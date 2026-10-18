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


def get_tool_definition(name):
    """Return the canonical model-facing tool contract by stable name."""
    for definition in TOOL_DEFS:
        if definition["name"] == name:
            return definition
    raise KeyError(f"Unknown tool definition: {name}")

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


def _reviewer_history(toolbox, review, args):
    exclude = review["review_id"] if args["user_id"] == review["user_id"] else None
    return toolbox.reviewer_history_lookup(args["user_id"], exclude_review_id=exclude)


def _business_trend(toolbox, review, args):
    exclude = review["review_id"] if args["prod_id"] == review["prod_id"] else None
    return toolbox.business_trend_lookup(
        args["prod_id"], args["around_date"], exclude_review_id=exclude
    )


def _text_similarity(toolbox, review, args):
    same_target = (
        (args["compare_against"] == "reviewer" and args["target_id"] == review["user_id"])
        or (args["compare_against"] == "business" and args["target_id"] == review["prod_id"])
    )
    exclude = review["review_id"] if same_target else None
    return toolbox.text_similarity_check(
        args["review_text"], args["compare_against"], args["target_id"],
        exclude_review_id=exclude,
    )


# Runtime handlers are separate from the provider-neutral JSON schemas. New
# evidence tools can register behavior without adding another dispatch branch.
TOOL_HANDLERS = {
    "reviewer_history_lookup": _reviewer_history,
    "business_trend_lookup": _business_trend,
    "text_similarity_check": _text_similarity,
}


def register_tool_handler(name, handler):
    if name not in {tool["name"] for tool in TOOL_DEFS}:
        raise ValueError(f"No schema registered for tool: {name}")
    TOOL_HANDLERS[name] = handler


def dispatch_tool_call(toolbox, review, tool_name, tool_input):
    """Dispatch a schema-defined evidence tool with consistent self-exclusion."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        raise ValueError(f"Unknown or non-evidence tool: {tool_name}")
    return handler(toolbox, review, tool_input)


def format_review_prompt(review):
    return (
        "Investigate this review:\n"
        f"- reviewer_id: {review['user_id']}\n"
        f"- business_id: {review['prod_id']}\n"
        f"- rating: {review['rating']}\n"
        f"- date: {review['date']}\n"
        f"- text: {review['review_text']}"
    )
