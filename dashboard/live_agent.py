"""
Agno-based multi-provider live agent runner. Replaces the earlier hand-rolled,
per-provider loop (three separate _run_* functions handling each provider's
different request/response shape — see PROCESS.md Decision #4 for why that
was originally necessary, and the framework-switch decision log entry for
why it no longer is).

Reuses the project's real tool logic unchanged: the self-exclusion dispatch
in scripts/tool_schemas.py's dispatch_tool_call(), the real evidence
functions in scripts/tools.py, and the exact system prompt / tool
descriptions already in use. Only the loop mechanics changed providers.

Key implementation notes, learned from live testing against Groq before
this was trusted:
  - Groq (and likely other providers) reject combining JSON-mode structured
    output with tool/function calling in one request — the same constraint
    that made the original design use a `submit_judgment` tool instead of a
    structured output schema. That design carries over unchanged here.
  - Agno's @tool(stop_after_tool_call=True) ends the run immediately after
    that tool executes, without an extra (wasted, and potentially
    fidelity-risking) model call to restate the judgment in prose — verified
    live: without it, the agent makes a 3rd model call after submit_judgment
    just to summarize what it already decided.
  - Agno's ToolExecution.result is always a str(...) of whatever the tool
    function returned, not the original dict — confirmed live: a result
    containing a None value renders as Python's `None` (not JSON's `null`),
    which broke the dashboard's st.json() display even though the agent
    itself received and correctly reasoned over the real data (only the
    trace UI was affected). Since it's always a Python literal repr, not
    JSON, ast.literal_eval() (not json.loads()) is what reverses it.
"""

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from tool_schemas import format_review_prompt, dispatch_tool_call  # noqa: E402
from tools import ToolBox  # noqa: E402
from data_index import load_index  # noqa: E402

from agno.agent import Agent  # noqa: E402
from agno.tools import tool as agno_tool  # noqa: E402
from agno.models.groq import Groq  # noqa: E402
from agno.models.anthropic import Claude  # noqa: E402
from agno.models.google import Gemini  # noqa: E402
from agno.models.openrouter import OpenRouter  # noqa: E402

MAX_TURNS = 6  # safety cap on model calls per investigation, same bound as before

PROVIDER_PRESETS = {
    "groq": {"api_key_env": "GROQ_API_KEY", "model_cls": Groq},
    "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "model_cls": OpenRouter},
    "gemini": {"api_key_env": "GOOGLE_API_KEY", "model_cls": Gemini},
    "claude": {"api_key_env": "ANTHROPIC_API_KEY", "model_cls": Claude},
}


class MissingAPIKey(Exception):
    pass


_toolbox_cache = None


def get_toolbox():
    global _toolbox_cache
    if _toolbox_cache is None:
        idx = load_index()
        _toolbox_cache = ToolBox(idx)
    return _toolbox_cache


def _require_key(env_name):
    key = os.environ.get(env_name)
    if not key:
        raise MissingAPIKey(f"{env_name} is not set. Add it in the sidebar or your environment.")
    return key


def _build_tools(toolbox, review):
    """Builds Agno tool functions bound to this specific review under
    investigation. Each one delegates to dispatch_tool_call() — the single
    place the self-exclusion logic lives — rather than reimplementing it, so
    this file can't drift out of sync with the rest of the project's tool
    behavior. Docstrings double as the tool descriptions Agno sends the
    model, so they're kept word-for-word aligned with tool_schemas.TOOL_DEFS."""

    def reviewer_history_lookup(user_id: str) -> dict:
        """Look up a reviewer's posting history: review count, average rating,
        rating spread, and burst timing. Call this for the reviewer under
        investigation, or for any other reviewer you want to cross-check
        (e.g. a reviewer who left similar-looking text at the same business)."""
        return dispatch_tool_call(toolbox, review, "reviewer_history_lookup", {"user_id": user_id})

    def business_trend_lookup(prod_id: str, around_date: str) -> dict:
        """Look up a business's review volume and rating trend around a given
        date (format YYYY-MM-DD). A sudden spike in review velocity
        (burst_ratio well above 1) around the review's date is a classic sign
        of an incentivized or coordinated campaign."""
        return dispatch_tool_call(
            toolbox, review, "business_trend_lookup",
            {"prod_id": prod_id, "around_date": around_date},
        )

    def text_similarity_check(review_text: str, compare_against: str, target_id: str) -> dict:
        """Compare a piece of review text against a reviewer's or a business's
        other reviews for templated or near-duplicate phrasing. compare_against
        must be exactly 'reviewer' or 'business'. Use this on the review under
        investigation, or on any other review text you've seen (e.g. from a
        second reviewer) to check whether they read like the same template."""
        return dispatch_tool_call(
            toolbox, review, "text_similarity_check",
            {"review_text": review_text, "compare_against": compare_against, "target_id": target_id},
        )

    @agno_tool(stop_after_tool_call=True)
    def submit_judgment(predicted_filtered: bool, confidence: float, primary_signal: str, reasoning: str) -> str:
        """Submit your final judgment once you have enough evidence. Ends the
        investigation. predicted_filtered: true if the review is likely fake.
        confidence: a number from 0.0 to 1.0. primary_signal: one of
        reviewer_pattern, business_trend, text_similarity, combination.
        reasoning: 1-2 sentences justifying the answer."""
        return "submitted"

    return [reviewer_history_lookup, business_trend_lookup, text_similarity_check, submit_judgment]


def _parse_tool_result(result):
    """Agno's ToolExecution.result is always str(...) of the tool's actual
    return value, not the dict/list itself — reverse that with
    ast.literal_eval (it's a Python literal repr, not JSON, so json.loads
    would fail on None/single-quoted strings). Falls back to the raw string
    if it's ever not a literal (defensive, not expected in normal operation)."""
    if not isinstance(result, str):
        return result
    try:
        return ast.literal_eval(result)
    except (ValueError, SyntaxError):
        return result


def _build_model(provider, model_id, api_key, temperature):
    preset = PROVIDER_PRESETS[provider]
    return preset["model_cls"](id=model_id, api_key=api_key, temperature=temperature)


def run_investigation_stream(review, config):
    """
    review: dict with review_id, user_id, prod_id, rating, date, review_text
    config: dict with provider, model, temperature, system_prompt

    Yields the same trace-event shape as the original hand-rolled
    implementation, so the dashboard's UI code needed zero changes:
      {"type": "turn", "turn": n}
      {"type": "tool_call", "turn": n, "tool": name, "input": {...}, "result": {...}}
      {"type": "final", "predicted_filtered":..., "confidence":..., "primary_signal":..., "reasoning":..., "turns_used": n}
      {"type": "error", "message": str}
    """
    provider = config["provider"]
    preset = PROVIDER_PRESETS.get(provider)
    if preset is None:
        yield {"type": "error", "message": f"Unknown provider: {provider}"}
        return

    try:
        api_key = _require_key(preset["api_key_env"])
    except MissingAPIKey as e:
        yield {"type": "error", "message": str(e)}
        return

    toolbox = get_toolbox()
    tools = _build_tools(toolbox, review)
    model = _build_model(provider, config["model"], api_key, config["temperature"])

    agent = Agent(
        model=model, tools=tools, instructions=config["system_prompt"],
        retries=3, delay_between_retries=3, exponential_backoff=True,
    )

    turn = -1
    try:
        for event in agent.run(format_review_prompt(review), stream=True, stream_events=True):
            cls = type(event).__name__
            if cls == "RunErrorEvent":
                yield {"type": "error", "message": str(event.content)}
                return
            elif cls == "ModelRequestStartedEvent":
                turn += 1
                yield {"type": "turn", "turn": turn}
                if turn >= MAX_TURNS:
                    break
            elif cls == "ToolCallCompletedEvent":
                t = event.tool
                if t.tool_name == "submit_judgment":
                    yield {"type": "final", **t.tool_args, "turns_used": turn + 1}
                    return
                yield {"type": "tool_call", "turn": turn, "tool": t.tool_name,
                       "input": t.tool_args, "result": _parse_tool_result(t.result)}
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return

    yield {"type": "final", "turns_used": turn + 1, "_incomplete": True,
           "reasoning": f"Exceeded max turns ({MAX_TURNS}) without a judgment."}
