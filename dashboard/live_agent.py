"""
One unified, temperature-aware, multi-provider live agent runner. Reuses the
real tool definitions and dispatch logic from scripts/tool_schemas.py and the
real evidence functions from scripts/tools.py — no duplicated agent logic.

Unlike the earlier per-provider scripts (agent.py, agent_gemini.py,
agent_openai_compat.py), this one:
  - accepts temperature as a real, functional parameter (a documented gap in
    the original scripts — closed here since we own the raw API calls)
  - yields step-by-step trace events for live UI rendering, instead of only
    returning a final result
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from tool_schemas import TOOL_DEFS, dispatch_tool_call, format_review_prompt  # noqa: E402
from tools import ToolBox  # noqa: E402
from data_index import load_index  # noqa: E402

MAX_TURNS = 6

_OPENAI_TOOLS = [
    {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
    for t in TOOL_DEFS
]

PROVIDER_PRESETS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY", "kind": "openai_compat"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "kind": "openai_compat"},
    "gemini": {"api_key_env": "GOOGLE_API_KEY", "kind": "gemini"},
    "claude": {"api_key_env": "ANTHROPIC_API_KEY", "kind": "claude"},
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


def run_investigation_stream(review, config):
    """
    review: dict with review_id, user_id, prod_id, rating, date, review_text
    config: dict with provider, model, temperature, system_prompt

    Yields trace-event dicts as the investigation proceeds:
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

    try:
        if preset["kind"] == "openai_compat":
            yield from _run_openai_compat(review, config, preset, api_key, toolbox)
        elif preset["kind"] == "claude":
            yield from _run_claude(review, config, api_key, toolbox)
        elif preset["kind"] == "gemini":
            yield from _run_gemini(review, config, api_key, toolbox)
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


def _run_openai_compat(review, config, preset, api_key, toolbox):
    import openai

    client = openai.OpenAI(base_url=preset["base_url"], api_key=api_key)
    messages = [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": format_review_prompt(review)},
    ]

    for turn in range(MAX_TURNS):
        yield {"type": "turn", "turn": turn}
        response = _retry_openai(client, model=config["model"], messages=messages,
                                  tools=_OPENAI_TOOLS, temperature=config["temperature"], max_tokens=1024)
        message = response.choices[0].message

        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append({"role": "user", "content": "Please call submit_judgment to finish."})
            continue

        messages.append({
            "role": "assistant", "content": message.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                            for tc in message.tool_calls],
        })

        submit_call = next((tc for tc in message.tool_calls if tc.function.name == "submit_judgment"), None)
        if submit_call:
            args = json.loads(submit_call.function.arguments)
            yield {"type": "final", **args, "turns_used": turn + 1}
            return

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = dispatch_tool_call(toolbox, review, tc.function.name, args)
            yield {"type": "tool_call", "turn": turn, "tool": tc.function.name, "input": args, "result": result}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    yield {"type": "final", "predicted_filtered": None, "confidence": 0.0, "primary_signal": "combination",
           "reasoning": f"Exceeded max turns ({MAX_TURNS}) without a judgment.", "turns_used": MAX_TURNS}


def _retry_openai(client, max_retries=4, **kwargs):
    import re
    delay_re = re.compile(r"retry.*?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "rate" in msg.lower()
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            m = delay_re.search(msg)
            delay = float(m.group(1)) + 1 if m else 10
            time.sleep(min(delay, 30))


def _run_claude(review, config, api_key, toolbox):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    claude_tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in TOOL_DEFS]
    messages = [{"role": "user", "content": format_review_prompt(review)}]

    for turn in range(MAX_TURNS):
        yield {"type": "turn", "turn": turn}
        response = client.messages.create(
            model=config["model"], max_tokens=1024, temperature=config["temperature"],
            system=config["system_prompt"], tools=claude_tools, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            messages.append({"role": "user", "content": "Please call submit_judgment to finish."})
            continue

        submit_block = next((b for b in tool_uses if b.name == "submit_judgment"), None)
        if submit_block:
            yield {"type": "final", **submit_block.input, "turns_used": turn + 1}
            return

        tool_results = []
        for block in tool_uses:
            result = dispatch_tool_call(toolbox, review, block.name, block.input)
            yield {"type": "tool_call", "turn": turn, "tool": block.name, "input": block.input, "result": result}
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})

    yield {"type": "final", "predicted_filtered": None, "confidence": 0.0, "primary_signal": "combination",
           "reasoning": f"Exceeded max turns ({MAX_TURNS}) without a judgment.", "turns_used": MAX_TURNS}


def _run_gemini(review, config, api_key, toolbox):
    from google import genai

    client = genai.Client(api_key=api_key)
    gemini_tools = [{"type": "function", **t} for t in TOOL_DEFS]
    kwargs = dict(model=config["model"], system_instruction=config["system_prompt"], tools=gemini_tools,
                  generation_config={"temperature": config["temperature"]})

    interaction = client.interactions.create(input=format_review_prompt(review), **kwargs)

    for turn in range(MAX_TURNS):
        yield {"type": "turn", "turn": turn}
        fc_steps = [s for s in interaction.steps if s.type == "function_call"]

        if not fc_steps:
            interaction = client.interactions.create(
                input="Please call submit_judgment to finish.", previous_interaction_id=interaction.id, **kwargs)
            continue

        submit_step = next((s for s in fc_steps if s.name == "submit_judgment"), None)
        if submit_step:
            yield {"type": "final", **submit_step.arguments, "turns_used": turn + 1}
            return

        function_results = []
        for step in fc_steps:
            result = dispatch_tool_call(toolbox, review, step.name, step.arguments)
            yield {"type": "tool_call", "turn": turn, "tool": step.name, "input": step.arguments, "result": result}
            function_results.append({"type": "function_result", "name": step.name, "call_id": step.id,
                                      "result": [{"type": "text", "text": json.dumps(result)}]})

        interaction = client.interactions.create(
            input=function_results, previous_interaction_id=interaction.id, **kwargs)

    yield {"type": "final", "predicted_filtered": None, "confidence": 0.0, "primary_signal": "combination",
           "reasoning": f"Exceeded max turns ({MAX_TURNS}) without a judgment.", "turns_used": MAX_TURNS}
