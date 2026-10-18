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
from copy import deepcopy
import os

from scripts.tool_schemas import format_review_prompt, dispatch_tool_call, TOOL_DEFS
from scripts.tools import ToolBox
from scripts.data_index import load_index

from agno.agent import Agent  # noqa: E402
from agno.tools import tool as agno_tool  # noqa: E402
from dashboard.agent_architectures import register_architecture
from dashboard.providers import PROVIDER_PRESETS, build_model

MAX_TURNS = 6  # safety cap on model calls per investigation, same bound as before


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
    """Adapt canonical JSON-schema tools to Agno without duplicating wrappers."""
    tools = []

    def make_evidence_adapter(tool_name):
        def adapter(**arguments):
            return dispatch_tool_call(toolbox, review, tool_name, arguments)
        return adapter

    for definition in TOOL_DEFS:
        name = definition["name"]

        if name == "submit_judgment":
            def adapter(**arguments):
                return "submitted"
            stop_after = True
        else:
            adapter = make_evidence_adapter(name)
            stop_after = False

        adapter.__name__ = name
        adapter.__doc__ = definition["description"]
        adapted = agno_tool(
            name=name,
            description=definition["description"],
            stop_after_tool_call=stop_after,
        )(adapter)
        # Agno builds a Python-oriented schema from the callable. Replace that
        # schema with the canonical provider-neutral contract for exact parity.
        adapted.parameters = deepcopy(definition["parameters"])
        tools.append(adapted)
    return tools


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
    """Compatibility wrapper; provider construction lives in providers.py."""
    return build_model(provider, model_id, api_key, temperature)


def run_agno_investigation_stream(review, config):
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


register_architecture("agno", run_agno_investigation_stream)

# Preserve the previous import path for callers outside the dashboard.
run_investigation_stream = run_agno_investigation_stream
