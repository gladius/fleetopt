"""Tools the sessions read with, and what the loop shares with them.

Everything here is deterministic. The agent decides *what* to look at and *what*
to change; it does not get to decide what the numbers are. Measurement and the
equivalence gate stay in code because their output is a claim handed to another
team, and a claim has to be reproducible.
"""

import datetime
import time

from claude_agent_sdk import create_sdk_mcp_server, tool

from fleetopt.evidence import measure as measure_mod
from fleetopt.evidence import shape as shape_mod
from fleetopt.probe import store

# Set once by session.py before the agent starts.
CTX = {}


def _ok(text):
    return {"content": [{"type": "text", "text": text}]}


def say(line):
    """A line for whoever is watching the run. What the tools established, in words a
    person outside fleetopt reads; the names of tools and what a session thinks aloud
    go to the log."""
    CTX["said_at"] = time.time()
    print(line, flush=True)


WORDS = {"improved": "better", "regressed": "worse", "within noise": "no real change",
         "baseline finished nothing": "the original finished nothing"}
SHOWN = (("cost_usd", "cost"), ("wall_ms", "time"), ("llm_calls", "model calls"))


def compared(name, result):
    """One line for a comparison: what moved, and whether it is past the noise."""
    parts = [f"{word} {v['delta_pct']:+.0f}% ({WORDS.get(v['verdict'], v['verdict'])})"
             for key, word in SHOWN if (v := result.get(key)) and v.get("delta_pct") is not None]
    done = result.get("completed") or {}
    if done.get("before") is not None and done.get("after") is not None and done["before"] != done["after"]:
        parts.append(f"requests finished per run {done['before']:g} to {done['after']:g} "
                     f"({WORDS.get(done['verdict'], done['verdict'])})")
    return f"[fleetopt] {name} against the original: " + (", ".join(parts) or "nothing could be compared")


def over():
    """Why no further run of the agent may start, or None. The session's own spend and
    turns are capped by the SDK; these two are what it could otherwise spend without
    end: the team's money, and time."""
    limit = CTX.get("max_team_usd")
    if limit is not None:
        runs, cost = measure_mod.spent(CTX["out"], CTX["project"], CTX.get("first_session", 0))
        if cost is not None and cost >= limit:
            return f"the limit on the team's key is reached: ${cost:.2f} spent in {runs} runs of the agent, limit ${limit:.2f}"
    if CTX.get("deadline") and time.time() >= CTX["deadline"]:
        return f"the time limit for a run is reached: {CTX.get('max_minutes', 0):g} minutes"
    return None


def _record(event, **data):
    """One line in the run record (session.py writes run.json): what a tool
    established, never the target's prompts or outputs."""
    CTX.setdefault("events", []).append(
        {"t": datetime.datetime.now().isoformat(timespec="seconds"), "event": event, **data})


def _state(ids):
    """The code fingerprint the sessions ran against."""
    if not ids:
        return None
    with _conn() as conn:
        return conn.execute("SELECT code_state FROM sessions WHERE id = ?", (ids[0],)).fetchone()["code_state"]


def _conn():
    return store.connect(CTX["out"] / "fleetopt.db")


def _ids(label):
    """Sessions under a label, for THIS project, that actually completed, at the code
    state of the newest measurement under that label.

    Three filters, each learned the expensive way. exit_code: a crashed run leaves a
    truncated trace that must not reach a median. project: the db is shared, and a
    'baseline' from another repo must never be pooled. code_state: labels get reused
    across days, and a 'baseline' measured on last week's source is not this baseline -
    without this the agent hit the mixed-state guard and re-measured under a fresh
    label, three runs it did not need."""
    # A review reads crashed runs too: the crash is its first finding. Nothing that
    # feeds a median ever sets this.
    ok = "" if CTX.get("include_failed") else " AND exit_code = 0"
    with _conn() as conn:
        return [r["id"] for r in conn.execute(
            f"SELECT id FROM sessions WHERE label = ?{ok} AND project = ?"
            "   AND code_state IS (SELECT code_state FROM sessions"
            f"                       WHERE label = ?{ok} AND project = ?"
            "                       ORDER BY id DESC LIMIT 1)",
            (label, str(CTX["project"]), label, str(CTX["project"])),
        )]


@tool(
    "query_traces",
    "Run read-only SQL against the capture database. Tables: runs(session_id, "
    "run_id, parent_run_id, trace_id, name, run_type, node, step, model, "
    "provider, duration_ms, input_tokens, output_tokens, cache_read_tokens, "
    "cache_write_tokens, prompt_chars, prompt, completion, inputs, outputs, "
    "error), sessions(id, label, run_cmd, exit_code), graphs(session_id, nodes, "
    "edges, mermaid). Prefer aggregates - full prompt text is large.",
    {"sql": str},
)
async def query_traces(args):
    sql = args["sql"].strip()
    if not sql.lower().startswith("select"):
        return _ok("refused: read-only, SELECT statements only")
    with _conn() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        return _ok("(no rows)")
    out = [" | ".join(rows[0].keys())]
    out += [" | ".join(str(r[c])[:200] for c in rows[0].keys()) for r in rows[:100]]
    if len(rows) > 100:
        out.append(f"... {len(rows) - 100} more rows")
    return _ok("\n".join(out))


@tool(
    "graph_topology",
    "The target's LangGraph structure: nodes, edges, which edges are conditional, "
    "and a mermaid diagram. Captured from the compiled graph, not parsed from source.",
    {},
)
async def graph_topology(args):
    with _conn() as conn:
        row = conn.execute(
            "SELECT nodes, edges, mermaid FROM graphs ORDER BY session_id DESC LIMIT 1"
        ).fetchone()
    if not row:
        return _ok("no graph captured - the target may not be a LangGraph project")
    return _ok(
        f"nodes: {row['nodes']}\n\nedges: {row['edges']}\n\n{row['mermaid'] or ''}"
    )


@tool(
    "graph_shape",
    "Structural facts from the traces under a label: branches declared but never taken, "
    "dispatchers whose target order never varies (and whether a model was consulted to "
    "decide it), loops that run the same number of rounds in every trace, repeated model "
    "calls with identical replies. Numbers only - the evidence an architecture review cites.",
    {"label": str},
)
async def graph_shape(args):
    ids = _ids(args["label"])
    if not ids:
        return _ok(f"no completed measurement under label {args['label']!r}")
    with _conn() as conn:
        result = shape_mod.analyze(conn, ids)
    _record("graph_shape", label=args["label"], traces=result["traces"], findings=result["findings"])
    return _ok(shape_mod.render(result))


_TOOLS = [query_traces, graph_topology, graph_shape]


def server(names=None):
    """The in-process tool server; `names` (mcp__fleetopt__* or bare) selects a subset,
    which is how the reviewer gets a read-only one."""
    chosen = _TOOLS if names is None else [
        t for t in _TOOLS if t.name in names or f"mcp__fleetopt__{t.name}" in names]
    return create_sdk_mcp_server(name="fleetopt", tools=chosen)


TOOL_NAMES = [f"mcp__fleetopt__{t.name}" for t in _TOOLS]
