"""Tools the sessions call.

Everything here is deterministic. The agent decides *what* to look at and *what*
to change; it does not get to decide what the numbers are. Measurement and the
equivalence gate stay in code because their output is a claim handed to another
team, and a claim has to be reproducible.
"""

import datetime
import json
import pathlib
import re
import time

from claude_agent_sdk import create_sdk_mcp_server, tool

from fleetopt.evidence import evals as evals_mod
from fleetopt.evidence import judge as judge_mod
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


def announce(label):
    """Say which finding a measurement belongs to, once, by the name the review gave it."""
    found = re.match(r"([CDN]\d+)", label)
    if not found or found.group(1) in CTX.setdefault("announced", set()):
        return
    CTX["announced"].add(found.group(1))
    title = next((f["title"] for f in CTX.get("findings") or [] if f["id"] == found.group(1)),
                 "found while applying the others")
    say(f"\n[fleetopt] {found.group(1)}: {title}")


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


def sides(baseline, candidate, base_state, cand_state):
    """Which code each side ran, in words nobody can misread. Observed: an optimizer
    measured its patched code under a label called 'baseline-retest', then cited that
    comparison as proof the unmodified agent had the same defect."""
    line = f"{baseline} ran code {base_state}; {candidate} ran code {cand_state}."
    if base_state == cand_state:
        line += ("\nNOTE: both sides ran the SAME code. This shows the agent's own run-to-run"
                 " variation. It says nothing about the effect of a change.")
    return line


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
    "measure",
    "Run the target agent n times under instrumentation and store the results "
    "under a label. Use 'baseline' before changing anything, and another label "
    "after. fleetopt starts the agent itself, with the same inputs every time. Returns "
    "median cost, wall time, finished requests and tokens.",
    {"label": str, "n": int},
)
async def measure(args):
    label, n = args["label"], args.get("n", 3)
    stop = over()
    if stop:
        _record("limit", label=label, reason=stop)
        say(f"[fleetopt] stopping here: {stop}")
        return _ok(f"refused: {stop}. No further measurement is possible in this run. Undo any change "
                   "that has not been measured and judged, and write your report now.")
    announce(label)
    try:
        measure_mod.collect(CTX["project"], CTX["run_cmd"], CTX["out"], n, label, say=say)
    except RuntimeError as e:
        _record("measure_failed", label=label, error=str(e)[:300])
        say(f"[fleetopt] {measure_mod.plain(label)}: could not be measured")
        return _ok(f"measurement failed: {e}")
    with _conn() as conn:
        ids = _ids(label)
        stats, _ = measure_mod.aggregate(conn, ids)
    state = _state(ids)
    original = CTX.setdefault("baseline_state", state)  # the first measurement of a run is the unmodified code
    _record("measure", label=label, n=n, sessions=len(ids), stats=stats, code_state=state)
    cost = "no price for its model" if stats.get("cost_usd") is None else f"${stats['cost_usd']:.4f} per run"
    done = "" if stats.get("completed") is None else f", {stats['completed']:g} requests finished per run"
    say(f"[fleetopt] {measure_mod.plain(label)}: {cost}{done}")
    note = f"\ncode state {state}"
    if state != original:
        note += (f" - this is CHANGED code (the run's first measurement was {original})."
                 " Whatever the label says, it is not a baseline.")
    return _ok(f"{label} ({n} runs), medians:\n{json.dumps(stats, indent=2)}{note}")


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
        comparison = measure_mod.compare(conn, base, cand)
    b, c = _state(base), _state(cand)
    _record("compare", baseline=args["baseline"], candidate=args["candidate"], result=comparison,
            baseline_state=b, candidate_state=c)
    if b != c:  # the same code twice is the agent's own variation, of no interest to who is watching
        say(compared(args["candidate"], comparison))
    return _ok(sides(args["baseline"], args["candidate"], b, c) + "\n\n" + measure_mod.render(comparison))


@tool(
    "judge",
    "Check the changed agent still answers as well. Runs as isolated calls that see "
    "only the task and the outputs - not your patch or reasoning. A request passes "
    "when its answer is correct by the team's eval case, or, where no case covers it, "
    "when it is equivalent to the original answer. One failed request fails the change.",
    {"task": str, "baseline": str, "candidate": str},
)
async def judge(args):
    base, cand = _ids(args["baseline"]), _ids(args["candidate"])
    if not base or not cand:
        return _ok("need both measurements before judging")
    cases = CTX.get("eval_cases")
    with _conn() as conn:
        passed, results, correctness = await judge_mod.judge_sessions(
            conn, args["task"], base[0], cand[0], cases
        )
    b, c = _state(base), _state(cand)
    _record("judge", baseline=args["baseline"], candidate=args["candidate"],
            passed=passed, equivalence=results, correctness=correctness, baseline_state=b, candidate_state=c)
    if b != c:
        say(f"[fleetopt] {args['candidate']} judged: {sum(bool(r['kept_on']) for r in results)} of {len(results)} "
            f"requests passed. {'Passed' if passed else 'Failed'}")
    lines = [sides(args["baseline"], args["candidate"], b, c), "", "per request (what it passed on):"]
    lines += [f"  [{'PASS: ' + r['kept_on'] if r['kept_on'] else 'FAIL'}] "
              f"{'unchanged' if r['equivalent'] else 'changed'}: {r['reason']}" for r in results]
    if correctness:
        m = correctness["matched"]
        lines.append(f"\ncorrectness against the team's eval cases ({correctness['cases']} loaded, {m} matched a captured run):")
        lines.append(f"  baseline passes {correctness['baseline_pass']}/{m}, candidate passes {correctness['candidate_pass']}/{m}")
        lines += [f"  [{'PASS' if r['candidate_pass'] else 'FAIL'}] {r['input'][:60]!r}: {r['reason']}" for r in correctness["rows"]]
        if m < correctness["cases"]:
            lines.append(f"  {correctness['cases'] - m} cases were not exercised by the run command and could not be graded.")
    else:
        lines.append("\ncorrectness: not checked - no eval cases loaded. A pass here means unchanged, not correct.")
    lines.append(f"\nverdict: {'PASSED' if passed else 'FAILED'} ({len(results)} invocations)")
    return _ok("\n".join(lines))


@tool(
    "load_eval_cases",
    "Load the project's eval cases (input + expected answer) from a file or folder "
    "before measuring: JSONL/JSON with input/expected keys, or deepeval test files "
    "(LLMTestCase/Golden). See fleetopt:evals for where to look. Once loaded, judge "
    "also grades correctness against the expected answers for every captured run "
    "whose input matches a case, and reports pass rates before and after.",
    {"path": str},
)
async def load_eval_cases(args):
    path = pathlib.Path(args["path"])
    if not path.is_absolute():
        path = CTX["project"] / path
    if not path.exists():
        return _ok(f"no such path: {path}")
    cases, notes = evals_mod.load(path)
    CTX["eval_cases"] = cases
    _record("eval_cases", path=str(path), loaded=len(cases))
    if cases:  # kept in fleetopt's own folder, never in the team's repo
        keep = CTX["out"] / "evals"
        keep.mkdir(parents=True, exist_ok=True)
        (keep / f"{CTX['project'].name}.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n", encoding="utf-8"
        )
    sample = cases[0] if cases else None
    lines = [f"{len(cases)} eval cases loaded from {path}"] + [f"  {n}" for n in notes]
    if sample:
        lines.append(f"  first case: input={sample['input'][:80]!r} expected={sample['expected'][:80]!r}")
    else:
        lines.append("  nothing loadable here - see fleetopt:evals for the formats that work")
    return _ok("\n".join(lines))


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


_TOOLS = [measure, query_traces, graph_topology, graph_shape, compare, judge,
          load_eval_cases]


def server(names=None):
    """The in-process tool server; `names` (mcp__fleetopt__* or bare) selects a subset,
    which is how the reviewer gets a read-only one."""
    chosen = _TOOLS if names is None else [
        t for t in _TOOLS if t.name in names or f"mcp__fleetopt__{t.name}" in names]
    return create_sdk_mcp_server(name="fleetopt", tools=chosen)


TOOL_NAMES = [f"mcp__fleetopt__{t.name}" for t in _TOOLS]
