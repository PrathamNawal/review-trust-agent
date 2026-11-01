"""Describes how the active agent architecture works, read from the real config and code.

Nothing here is typed in by hand when the code already knows it. The Jev
description reads the deployed config (decision method, cutoff, instructions,
which evidence sources are switched on) and the Jev client's own constants.
The Agno description reads the tools the Agno runner actually builds, their
stop flag, the safety cap, and the Agent's memory defaults.

What is *not* read from code, and is stated as such on the page:
  - Jev is a fixed straight line (evidence -> one request -> label -> rule ->
    person), so there is no control-flow object to introspect.
  - Agno runs its own built-in tool loop and exports no graph, so the loop's
    arrows are drawn.
  - The plain-language questions for each evidence source, and the routing
    rule's wording, mirror the code they describe.
"""

import inspect

from dashboard.config_store import DEFAULT_JEV_EVIDENCE_SOURCES, DEFAULT_JEV_PROMPTS
from dashboard.jev_agent import MODEL_ID
from dashboard.jev_model import MAX_ATTEMPTS, REVIEW_STATE_FIELDS

KIND_NAMES = {
    "input": "input", "evidence": "evidence", "tool": "tool", "model": "model",
    "output": "output", "terminal": "terminal", "rule": "rule", "human": "person",
}

KIND_COLORS = {
    "input": "#667085", "evidence": "#0d9488", "tool": "#0d9488", "model": "#2563eb",
    "output": "#7c3aed", "terminal": "#7c3aed", "rule": "#d97706", "human": "#059669",
}

PLAY_PAGE = "7. Play — Tweak the Agent"

# (config key, short name, two-line diagram label, question it answers, what it returns)
JEV_SOURCES = (
    ("reviewer_history", "Reviewer activity", "Reviewer\\nactivity",
     "Does this reviewer's posting pattern look normal?",
     "earlier review count, rating spread, timing gaps, and the most reviews in any 7-day window"),
    ("business_trend", "Business trend", "Business\\ntrend",
     "Was there a review-volume spike near this date?",
     "earlier review count, rating in a 14-day window, and the burst ratio against the business's earlier baseline"),
    ("reviewer_text_similarity", "Reviewer text reuse", "Reviewer\\ntext reuse",
     "Is this wording reused from the reviewer's earlier reviews?",
     "closest-match score, average score, number compared, and the top matching snippets"),
    ("business_text_similarity", "Business text reuse", "Business\\ntext reuse",
     "Is this wording reused from earlier reviews of the business?",
     "closest-match score, average score, number compared, and the top matching snippets"),
)

_AGNO_PLAIN = {
    "reviewer_history_lookup": ("Reviewer history", "Does this reviewer's posting pattern look normal?"),
    "business_trend_lookup": ("Business trend", "Was there a review spike at this business around this date?"),
    "text_similarity_check": ("Text similarity", "Does the wording look templated or copy-pasted?"),
}


def _step(step_id, label, kind, title, why, rows, tunables=None, tunable_note="",
          play=False, off=False):
    return {"id": step_id, "label": label, "kind": kind, "kind_name": KIND_NAMES[kind],
            "title": title, "why": why, "rows": rows, "tunables": tunables,
            "tunable_note": tunable_note, "play": play, "off": off}


def describe_mechanism(config):
    """Return the steps, diagram graph and state/memory cards for the active config."""
    architecture = config.get("architecture", "agno")
    if architecture == "jev":
        return _describe_jev(config)
    if architecture == "agno":
        return _describe_agno(config)
    raise ValueError(f"No mechanism description for architecture {architecture!r}.")


# --------------------------------------------------------------------------- Jev

def _describe_jev(config):
    method = config.get("jev_method", "choice")
    method_name = method.title()
    cutoff = None if method == "choice" else float(config.get("jev_threshold", 0.5))
    enabled = set(config.get("jev_evidence_sources", DEFAULT_JEV_EVIDENCE_SOURCES))
    prompts = {**DEFAULT_JEV_PROMPTS, **config.get("jev_prompts", {})}
    instructions = "default" if prompts[method] == DEFAULT_JEV_PROMPTS[method] else "edited"
    n_on = sum(1 for key, *_ in JEV_SOURCES if key in enabled)

    steps = [_step(
        "input", "Review in", "input", "Review in",
        "Only the rating, date and review text go in as the review. Dataset labels are never sent to the model.",
        [("Fields sent", ", ".join(REVIEW_STATE_FIELDS)),
         ("Read from", "REVIEW_STATE_FIELDS in jev_model.py")],
        tunable_note="Fixed by code — not a Play setting.",
    )]

    for key, name, _short, asks, returns in JEV_SOURCES:
        on = key in enabled
        steps.append(_step(
            f"src_{key}", name, "evidence", name,
            "Built by code before Jev is called — not chosen by the model. "
            "Uses only reviews dated strictly before this one.",
            [("Asks", asks), ("Returns", returns),
             ("Included now", "on — sent to Jev" if on else "off — Jev does not receive it")],
            tunables=[("Include this source", "on" if on else "off")],
            play=True, off=not on,
        ))

    steps.append(_step(
        "jev", "Jev (the model)", "model", "Jev · the model",
        "One typed question per review, answered in a single call. Jev doesn't pick tools, "
        "loop, or call anything back — the evidence is already in the request.",
        [("Model", f"{MODEL_ID} — pinned for reproducibility"),
         ("Temperature", "not sent — the client doesn't expose it"),
         ("Question type", method_name),
         ("Retries", f"up to {MAX_ATTEMPTS} attempts when rate limited (HTTP 429 / 529)")],
        tunables=[("Decision method", method_name),
                  ("Instructions", f"{instructions} · {method}"),
                  ("Cutoff", "n/a — Choice has none" if cutoff is None else f"{cutoff:.2f}")],
        play=True,
    ))

    if cutoff is None:
        label_why = ("Jev picks filtered or recommended itself and reports the probability of "
                     "its pick. There is no cutoff.")
        label_rows = [("Output", "label + probability of the chosen label"),
                      ("Calibrated?", "No — the app marks confidence_calibrated = False")]
    else:
        label_why = (f"Jev returns a 0–1 value. A PM-set cutoff ({cutoff:.2f}) turns it into filtered "
                     "(at or above) or recommended — an experimental rule, not a calibrated one.")
        label_rows = [("Output", "0–1 value, then cutoff → label"),
                      ("Calibrated?", "No — the app marks confidence_calibrated = False")]
    steps.append(_step(
        "label", "Advisory label", "output", "Advisory label", label_why, label_rows,
        tunables=None if cutoff is None else [("Cutoff", f"{cutoff:.2f}")],
        tunable_note="Choice has no cutoff.", play=cutoff is not None,
    ))

    steps.append(_step(
        "router", "Routing rule", "rule", "Routing rule",
        "Not a model. Jev gives no confidence number (confidence is None), and the policy treats a "
        "missing confidence as 'queue' — so every Jev result goes to a person, and the model can "
        "never trigger a removal.",
        [("Rule (compute_routing)",
          "confidence missing → human queue. Only when a number exists: at or above the threshold → "
          "remove or ignore, below → queue"),
         ("Play's confidence threshold", "no effect on Jev — it is compared only when a confidence exists")],
        tunable_note="Fixed by code for Jev — not a Play setting.",
    ))

    steps.append(_step(
        "human", "Human reviewer", "human", "Human reviewer",
        "A person makes the final call: accept, override (flip the label), or escalate. Each action "
        "is logged, so disagreement with the model is visible instead of silent.",
        [("Where", "5 · Analyst Queue"),
         ("Logged as", "an analyst action in MLflow, with a required reason for override or escalate")],
        tunable_note="Not a Play setting.",
    ))

    nodes = {"input": {"title": "Review in", "body": ["rating · date · text"], "kind": "input"}}
    for key, name, *_ in JEV_SOURCES:
        nodes[f"src_{key}"] = {"title": name, "body": [], "kind": "evidence", "off": key not in enabled}
    nodes["jev"] = {"title": "Jev (model)", "body": [MODEL_ID, f"question: {method_name}"], "kind": "model"}
    nodes["label"] = {"title": "Advisory label",
                      "body": ["no cutoff"] if cutoff is None else [f"cutoff {cutoff:.2f}"], "kind": "output"}
    nodes["router"] = {"title": "Routing rule", "body": ["always a human"], "kind": "rule"}
    nodes["human"] = {"title": "Human reviewer", "body": ["accept · override"], "kind": "human"}

    first, last = f"src_{JEV_SOURCES[0][0]}", f"src_{JEV_SOURCES[-1][0]}"
    graph = {
        "nodes": nodes,
        "clusters": [{"id": "ev", "label": "Evidence step · built by code, earlier reviews only",
                      "members": [f"src_{key}" for key, *_ in JEV_SOURCES]}],
        "edges": [
            {"src": "input", "dst": first, "lhead": "ev", "short": "build evidence"},
            {"src": last, "dst": "jev", "ltail": "ev", "label": "one request · no tool loop",
             "short": "one request"},
            {"src": "jev", "dst": "label", "label": "Jev answers", "short": "answers"},
            {"src": "label", "dst": "router", "short": "advisory"},
            {"src": "router", "dst": "human", "label": "a rule, not a model", "short": "every result"},
        ],
    }

    return {
        "architecture": "jev",
        "name": f"Jev {method_name}",
        "intro": ("The app builds evidence about the reviewer, the business and any reused wording — "
                  "using only reviews dated before this one — then asks Jev a single question. "
                  "Jev's answer is *advisory*: it produces no confidence number, so every result "
                  "goes to a person."),
        "steps": steps, "default_step": "jev", "graph": graph,
        "state_card": {
            "title": "What goes in, what comes out",
            "body": (f"One request: the review's **{', '.join(REVIEW_STATE_FIELDS)}**, plus the "
                     f"**{n_on} of {len(JEV_SOURCES)} evidence sources that are on**, plus the question and "
                     "the instructions. Jev answers once with a label and a probability or 0–1 "
                     "value. Nothing is fed back — there is no conversation and no tool results."),
            "source": "Read from: the fields sent by `classify_review()`. Dataset labels are not among them.",
        },
        "memory_card": {
            "title": "What persists between reviews",
            "body": ("**Nothing, inside the call.** Each review is judged on its own; there is no memory "
                     "and no history carried over. What *is* kept lives outside: each judgment and "
                     "analyst action in MLflow, and each config version."),
            "source": "On the free public host, saved data resets when the app restarts.",
        },
    }


# -------------------------------------------------------------------------- Agno

def _first_sentence(text):
    return (text or "").strip().split(". ")[0].rstrip(".") + "."


def _describe_agno(config):
    from agno.agent import Agent

    from dashboard.live_agent import MAX_TURNS, _build_tools

    threshold = float(config.get("confidence_threshold", 0.75))
    provider, model = config.get("provider", "?"), config.get("model", "?")
    temperature = float(config.get("temperature", 0.0))

    tools = _build_tools(None, None)  # the exact functions the runner hands the agent
    evidence_tools = [t for t in tools if not t.stop_after_tool_call]
    terminal = next((t for t in tools if t.stop_after_tool_call), None)

    agent_defaults = inspect.signature(Agent.__init__).parameters
    memory = {name: agent_defaults[name].default for name in
              ("db", "add_history_to_context", "enable_agentic_memory", "session_state")}

    steps = [_step(
        "a_input", "Review in", "input", "Review in",
        "The reviewer id, business id, rating, date and text go in. Yelp's own fraud label is never sent.",
        [("Read from", "format_review_prompt()")],
        tunable_note="Fixed by code — not a Play setting.",
    ), _step(
        "a_model", "The agent (model)", "model", "The agent (the model)",
        "The model's own output decides what happens next: call another tool, or submit a judgment. "
        "It may call tools in any order, repeat one, or point one at a different reviewer.",
        [("Read from", "Agent(model=…, instructions=system_prompt, retries=3)"),
         ("Safety cap", f"{MAX_TURNS} model calls (MAX_TURNS in live_agent.py) — code, not a Play setting")],
        tunables=[("System prompt", "default" if config.get("system_prompt") else "—"),
                  ("Provider", provider), ("Model", model), ("Temperature", f"{temperature:.2f}")],
        play=True,
    )]

    nodes = {
        "a_input": {"title": "Review in", "body": ["rating · date · text"], "kind": "input"},
        "a_model": {"title": "The agent", "body": [model], "kind": "model"},
    }
    tool_ids = []
    for tool in evidence_tools:
        name, asks = _AGNO_PLAIN.get(tool.name, (tool.name, _first_sentence(tool.description)))
        params = list((tool.parameters or {}).get("properties", {}))
        tid = f"t_{tool.name}"
        tool_ids.append(tid)
        steps.append(_step(
            tid, name, "tool", name,
            "A real lookup; its result is appended to the model's context. The model chooses "
            "whether, when and for whom to call it.",
            [("Asks", asks), ("Parameters (read from the tool)", " · ".join(params))],
            tunable_note="Not tunable from Play — lives in scripts/tools.py.",
        ))
        nodes[tid] = {"title": name, "body": [], "kind": "tool"}

    if terminal is not None:
        props = list((terminal.parameters or {}).get("properties", {}))
        steps.append(_step(
            "a_term", terminal.name, "terminal", terminal.name,
            "The model ends the run by calling this. It is marked stop_after_tool_call, so the run "
            "stops immediately with no extra model call.",
            [("Read from", f"the tool whose stop_after_tool_call is True"),
             ("Output", " · ".join(props))],
            tunable_note="Fixed by code — not a Play setting.",
        ))
        nodes["a_term"] = {"title": terminal.name, "body": ["ends the run"], "kind": "terminal"}

    steps.append(_step(
        "a_router", "Routing", "rule", "Routing",
        "Not the model. A plain comparison against a number the PM sets; the model never sees it.",
        [("Rule (compute_routing)",
          f"fake and confidence ≥ {threshold:.2f} → auto-remove · genuine and ≥ {threshold:.2f} → "
          f"ignore · below → human queue")],
        tunables=[("Confidence threshold", f"{threshold:.2f}")], play=True,
    ))
    nodes["a_router"] = {"title": "Routing", "body": [f"threshold {threshold:.2f}"], "kind": "rule"}
    nodes["o_remove"] = {"title": "Auto-remove", "body": [f"fake, ≥ {threshold:.2f}"], "kind": "output"}
    nodes["o_queue"] = {"title": "Human queue", "body": [f"below {threshold:.2f}"], "kind": "human"}
    nodes["o_ignore"] = {"title": "Ignore", "body": [f"genuine, ≥ {threshold:.2f}"], "kind": "output"}
    nodes["cap"] = {"title": "Safety cap", "body": [f"{MAX_TURNS} model calls"], "kind": "input", "dashed": True}

    edges = [{"src": "a_input", "dst": "a_model", "short": "review"}]
    if tool_ids:
        edges += [{"src": "a_model", "dst": tool_ids[0], "lhead": "tools", "label": "calls a tool", "short": "calls a tool"},
                  {"src": tool_ids[-1], "dst": "a_model", "ltail": "tools", "label": "result goes back", "short": "result back"}]
    if terminal is not None:
        edges += [{"src": "a_model", "dst": "a_term", "label": "decides it has enough", "short": "has enough"},
                  {"src": "a_term", "dst": "a_router", "short": "judgment"}]
    edges += [{"src": "a_router", "dst": "o_remove", "short": "remove"},
              {"src": "a_router", "dst": "o_queue", "short": "queue"},
              {"src": "a_router", "dst": "o_ignore", "short": "ignore"},
              {"src": "a_model", "dst": "cap", "dashed": True, "short": "no verdict"}]

    return {
        "architecture": "agno",
        "name": "Agno agent",
        "intro": ("The agent gathers evidence the way an analyst would — checking the reviewer, the "
                  "business and the wording — then stops when *it* decides it has enough. What "
                  "happens next is *not* the model's call: a plain threshold the PM sets decides that."),
        "steps": steps, "default_step": "a_model",
        "graph": {"nodes": nodes,
                  "clusters": [{"id": "tools", "label": "Tool step · one box per real tool",
                                "members": tool_ids}] if tool_ids else [],
                  "edges": edges},
        "state_card": {
            "title": "What moves between steps",
            "body": ("One growing conversation. It starts as the system prompt plus the review. Every "
                     "tool call and its result is **appended** — never overwritten — so on each turn "
                     "the model sees everything gathered so far."),
            "source": f"Read from: the run's message list. No separate state object exists "
                      f"(`session_state={memory['session_state']}`).",
        },
        "memory_card": {
            "title": "What persists between reviews",
            "body": ("**Nothing, inside the agent.** No database, no history carried over, no agent "
                     "memory — every investigation starts blank. What *is* kept lives outside it: "
                     "MLflow and the config versions."),
            "source": (f"Read from: `db={memory['db']}` · "
                       f"`add_history_to_context={memory['add_history_to_context']}` · "
                       f"`enable_agentic_memory={memory['enable_agentic_memory']}`."),
        },
    }


# ------------------------------------------------------------------------ stages

STAGE_NAMES = {"evidence": "Evidence", "tool": "Tools"}


def build_stages(mechanism):
    """Collapse the steps into the stages a visitor walks through, in flow order.

    Steps that share a diagram cluster (the four evidence sources, or the Agno tools)
    become one stage, so the rail stays short while each member keeps its own card.
    """
    cluster_of = {m: c["id"] for c in mechanism["graph"]["clusters"] for m in c["members"]}
    stages = []
    for step in mechanism["steps"]:
        cluster = cluster_of.get(step["id"])
        if cluster and stages and stages[-1].get("cluster") == cluster:
            stages[-1]["steps"].append(step)
        elif cluster:
            stages.append({"id": cluster, "cluster": cluster, "kind": step["kind"],
                           "label": STAGE_NAMES.get(step["kind"], step["label"]), "steps": [step]})
        else:
            stages.append({"id": step["id"], "cluster": None, "kind": step["kind"],
                           "label": step["label"], "steps": [step]})
    for stage in stages:
        off = sum(1 for step in stage["steps"] if step["off"])
        if stage["cluster"] and off:
            stage["label"] += f" · {len(stage['steps']) - off}/{len(stage['steps'])}"
    return stages


# ----------------------------------------------------------------------- diagram

ACCENT = "#C65346"  # the app's primaryColor (.streamlit/config.toml): "selected" in the rail too
MUTED = "#667085"

KIND_CAPTIONS = {
    "input": "INPUT", "evidence": "EVIDENCE · BY CODE", "tool": "TOOLS · MODEL PICKS",
    "model": "MODEL", "output": "OUTPUT", "terminal": "ENDS THE RUN",
    "rule": "RULE · NOT A MODEL", "human": "PERSON",
}


def _q(text):
    return str(text).replace('"', '\\"')


def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_label(caption, title, body, color):
    """A node label with three levels: coloured caption, bold title, muted detail lines."""
    rows = [f'<TR><TD><FONT POINT-SIZE="11" COLOR="{color}"><B>{_esc(caption)}</B></FONT></TD></TR>',
            f'<TR><TD><FONT POINT-SIZE="19"><B>{_esc(title)}</B></FONT></TD></TR>']
    rows += [f'<TR><TD><FONT POINT-SIZE="15" COLOR="#475467">{_esc(line)}</FONT></TD></TR>' for line in body]
    return ('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">'
            + "".join(rows) + "</TABLE>>")


def _stage_node(number, stage, graph):
    """(caption, title, body lines) for a stage, read from the graph nodes and steps."""
    caption = f"{number} · {KIND_CAPTIONS[stage['kind']]}"
    if not stage["cluster"]:
        node = graph["nodes"][stage["steps"][0]["id"]]
        return caption, node["title"], node["body"]
    total = len(stage["steps"])
    on = sum(1 for step in stage["steps"] if not step["off"])
    noun = "sources" if stage["kind"] == "evidence" else "tools"
    body = [f"{total} {noun}" if on == total else f"{on} of {total} on"]
    return caption, STAGE_NAMES.get(stage["kind"], stage["label"]), body


def build_stage_dot(mechanism, stages, selected=None):
    """Left-to-right flow with one numbered node per stage; the selected stage is filled in."""
    graph = mechanism["graph"]
    node_of = {step["id"]: stage["id"] for stage in stages for step in stage["steps"]}
    lines = [
        "digraph G {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.4;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#ffffff", '
        'margin="0.2,0.2", width=1.55, height=1.5];',
        f'  edge [color="{MUTED}", fontname="Helvetica", fontsize=13, fontcolor="{MUTED}", arrowsize=0.9];',
    ]

    def node(node_id, label, kind, dashed=False):
        color = KIND_COLORS[kind]
        attrs = [f"label={label}", f'color="{color}"', "penwidth=2"]
        if node_id == selected:
            attrs = [f"label={label}", f'color="{ACCENT}"', "penwidth=5", f'fillcolor="{ACCENT}1a"']
        if dashed:
            attrs.append('style="rounded,filled,dashed"')
        lines.append(f"  {node_id} [{', '.join(attrs)}];")

    for number, stage in enumerate(stages, 1):
        caption, title, body = _stage_node(number, stage, graph)
        color = ACCENT if stage["id"] == selected else KIND_COLORS[stage["kind"]]
        node(stage["id"], _html_label(caption, title, body, color), stage["kind"])
    for node_id, spec in graph["nodes"].items():
        if node_id not in node_of:
            node(node_id, _html_label(KIND_CAPTIONS[spec["kind"]], spec["title"], spec["body"],
                                      KIND_COLORS[spec["kind"]]), spec["kind"], spec.get("dashed", False))
    arrows = []
    for edge in graph["edges"]:
        src, dst = node_of.get(edge["src"], edge["src"]), node_of.get(edge["dst"], edge["dst"])
        if src != dst and (src, dst) not in [(a, b) for a, b, _ in arrows]:
            arrows.append((src, dst, edge))
    # A straight line of stages reads fine without arrow text; label arrows only where the flow branches or loops.
    branching = len({src for src, _, _ in arrows}) < len(arrows)
    for src, dst, edge in arrows:
        attrs = []
        if branching and edge.get("short"):
            attrs.append(f'label="  {_q(edge["short"])}  "')
        if edge.get("dashed"):
            attrs.append("style=dashed")
        lines.append(f"  {src} -> {dst}{' [' + ', '.join(attrs) + ']' if attrs else ''};")
    lines.append("}")
    return "\n".join(lines)
