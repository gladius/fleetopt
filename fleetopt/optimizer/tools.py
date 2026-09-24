"""Tools the optimizer agent calls.

Everything here is deterministic. The agent decides *what* to look at and *what*
to change; it does not get to decide what the numbers are. Measurement and the
equivalence gate stay in code because their output is a claim handed to another
team, and a claim has to be reproducible.
"""

import json

from claude_agent_sdk import create_sdk_mcp_server, tool

from fleetopt.evidence import judge as judge_mod
from fleetopt.evidence import measure as measure_mod
from fleetopt.probe import runner, store

# Set once by session.py before the agent starts.
CTX = {}


def _ok(text):
    return {"content": [{"type": "text", "text": text}]}


def _conn():
    return store.connect(CTX["out"] / "fleetopt.db")


def _ids(label):
    """Sessions under a label, for THIS project, that actually completed. A crashed
    run left a truncated trace behind; it must not reach a median. Scoped by project
    because the db is shared - a 'baseline' from another repo must never be pooled."""
    with _conn() as conn:
        return [r["id"] for r in conn.execute(
            "SELECT id FROM sessions WHERE label = ? AND exit_code = 0 AND project = ?",
            (label, str(CTX["project"])),
        )]


@tool(
    "set_run_command",
    "Tell the harness how to invoke the target agent once, end to end (e.g. "
    "'python main.py' or 'pytest tests/test_agent.py'). Required before measuring "
    "if it wasn't supplied on the command line. Find it in the README, pyproject "
    "scripts, tests, or langgraph.json.",
    {"cmd": str},
)
async def set_run_command(args):
    # An operator-supplied command is locked until it has actually failed under
    # measure. The alternative, observed: 13 turns re-deriving a command that was
    # already correct, two of them wrong.
    if CTX.get("run_locked") and not CTX.get("run_failed"):
        return _ok(f"run command was supplied by the operator and is locked: {CTX['run_cmd']!r}. "
                   "Call measure with it. It unlocks only if measure fails with it.")
    CTX["run_cmd"] = args["cmd"]
    return _ok(f"run command set: {args['cmd']}")


@tool(
    "measure",
    "Run the target agent n times under instrumentation and store the results "
    "under a label. Use 'baseline' before changing anything, and another label "
    "after. Returns median tokens, cost and wall time.",
    {"label": str, "n": int},
)
async def measure(args):
    label, n = args["label"], args.get("n", 3)
    if not CTX.get("run_cmd"):
        return _ok("no run command yet - call set_run_command first")
    try:
        measure_mod.collect(CTX["project"], CTX["run_cmd"], CTX["out"], n, label)
    except RuntimeError as e:
        CTX["run_failed"] = True  # unlocks set_run_command
        return _ok(f"measurement failed: {e}")
    with _conn() as conn:
        stats, _ = measure_mod.aggregate(conn, _ids(label))
    return _ok(f"{label} ({n} runs), medians:\n{json.dumps(stats, indent=2)}")


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
    "compare",
    "Before/after two labelled measurements. Reports 'within noise' when a delta "
    "sits inside the baseline's own run-to-run spread - that is not a saving.",
    {"baseline": str, "candidate": str},
)
async def compare(args):
    with _conn() as conn:
        base, cand = _ids(args["baseline"]), _ids(args["candidate"])
        if not base or not cand:
            return _ok(f"missing measurements: {args['baseline']}={len(base)}, "
                       f"{args['candidate']}={len(cand)}")
        return _ok(measure_mod.render(measure_mod.compare(conn, base, cand)))


@tool(
    "judge",
    "Check the optimized agent still answers the same. Runs as an isolated call "
    "that sees only the task and the two outputs - not your patch or reasoning. "
    "A cost saving with a failed judge is a regression.",
    {"task": str, "baseline": str, "candidate": str},
)
async def judge(args):
    base, cand = _ids(args["baseline"]), _ids(args["candidate"])
    if not base or not cand:
        return _ok("need both measurements before judging")
    with _conn() as conn:
        passed, results = await judge_mod.judge_sessions(conn, args["task"], base[0], cand[0])
    lines = [f"[{'PASS' if r['equivalent'] else 'FAIL'}] {r['reason']}" for r in results]
    lines.append(f"\nequivalence: {'PASSED' if passed else 'FAILED'} ({len(results)} cases)")
    return _ok("\n".join(lines))


_TOOLS = [set_run_command, measure, query_traces, graph_topology, compare, judge]


def server():
    return create_sdk_mcp_server(name="fleetopt", tools=_TOOLS)


TOOL_NAMES = [f"mcp__fleetopt__{t.name}" for t in _TOOLS]
