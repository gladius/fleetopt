"""The tools the sessions work with, and the limits inside them.

The agent decides what to look at, what to change, what to try next and when to stop.
It does not decide what the numbers are or whether a change is kept, and it cannot go
past a limit: those are here, the same for every agent. Measuring, keeping and undoing
are code because what they establish is a claim handed to another team, and a claim has
to be reproducible. Git is theirs alone, so a session cannot fake a baseline (observed:
a session ran `git checkout` to measure "the original" and measured its own change).

The limits, each learned on a real agent:
- a changed agent runs once before it runs three times, and a run that takes far more
  steps than the original is stopped (observed: a fix removed the crash that was the
  only thing ending a loop; the next run went 170 rounds on the team's key);
- the team's key, the clock and fleetopt's own spend each end a run;
- a finding gets two measured attempts (observed: four attempts at one fix);
- the agent as it is is measured once, and edits wait for it (observed: the original
  measured again for no reason);
- a change is kept only if something got better past the noise and the judge passed
  it; a change to the graph's structure only under --design and on the team's cases;
  never one that writes the team's expected answers into the code.
"""

import asyncio
import datetime
import pathlib
import subprocess
import time

from claude_agent_sdk import create_sdk_mcp_server, tool

from fleetopt.evidence import evals as evals_mod
from fleetopt.evidence import judge as judge_mod
from fleetopt.evidence import measure as measure_mod
from fleetopt.evidence import shape as shape_mod
from fleetopt.probe import runner, store

RUNS = 3           # runs of the agent per measurement
ATTEMPTS = 2       # measured versions per finding
STEP_FACTOR = 3    # a changed agent may take this many times the original's steps per run
BROKEN_FACTOR = 6  # ... or this many, when the original finished nothing and so stopped early
MIN_STEPS = 150
TEAM_USD = 2.0     # cap on the team's key; FLEETOPT_TEAM_USD overrides
MAX_MINUTES = 120  # the whole run; FLEETOPT_MAX_MINUTES overrides
GAINS = ("cost_usd", "wall_ms", "completed", "cost_per_completed")

# Set by start() before the apply session, by cli.py before the reviewer.
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
    across days, and a 'baseline' measured on last week's source is not this baseline."""
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


# --- the apply run: git, measurement and the gate, all in code --------------------------

def start(project, out, run_cmd, *, findings, task, cases=None, only=False, design=False, team_usd=TEAM_USD,
          minutes=MAX_MINUTES, first_session=0, run_dir=None):
    """Everything the apply tools share for one run, from the code as it stands."""
    CTX.clear()
    CTX.update(project=pathlib.Path(project).resolve(), out=pathlib.Path(out).resolve(), run_cmd=run_cmd,
               events=[], findings=findings, task=task, eval_cases=cases or None, only=only, design=design,
               max_team_usd=team_usd, max_minutes=minutes, deadline=time.time() + 60 * minutes,
               first_session=first_session, run_dir=run_dir, said_at=time.time(),
               baseline=None, attempts={}, rows={}, compared={}, pending=None, measured=None, editing=set())
    CTX.update(start_sha=_git("rev-parse", "HEAD"), start_state=runner.code_state(CTX["project"]))
    CTX.update(kept_sha=CTX["start_sha"], kept_label="baseline", start_untracked=_untracked())
    CTX["untracked"] = set(CTX["start_untracked"])


def _git(*args):
    done = subprocess.run(["git", "-C", str(CTX["project"]), *args], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.strip()[:200]}")
    return done.stdout.strip()


def _untracked():
    return set(_git("ls-files", "--others", "--exclude-standard").splitlines())


def _dirty():
    """Edits not yet measured: changed tracked files, or files created since the last run."""
    return bool(_git("status", "--porcelain", "--untracked-files=no")) or bool(_untracked() - CTX["untracked"])


def _commit(message):
    """Commit what the session changed: tracked files, and files it created. Not files
    the agent wrote into its own repository while it ran (observed)."""
    new = sorted(_untracked() - CTX["untracked"])
    _git("add", "-u")
    if new:
        _git("add", "--", *new)
    _git("-c", "user.name=fleetopt", "-c", "user.email=fleetopt@localhost", "commit", "-q", "-m", message)


def _reset():
    """Back to the code as last kept, and nothing half-made left behind."""
    _git("reset", "-q", "--hard", CTX["kept_sha"])
    for path in _untracked() - CTX["start_untracked"]:
        target = (CTX["project"] / path).resolve()
        if not target.is_relative_to(CTX["out"]):  # fleetopt's own records, if they live in the project
            target.unlink(missing_ok=True)
    CTX.update(pending=None, measured=None, untracked=_untracked(), editing=set())


def _stats(label):
    ids = _ids(label)
    with _conn() as conn:
        stats, _ = measure_mod.aggregate(conn, ids)
        steps = sorted(conn.execute("SELECT COUNT(*) FROM runs WHERE session_id = ?", (i,)).fetchone()[0] for i in ids)
    _record("measure", label=label, n=len(ids), stats=stats, code_state=_state(ids))
    return {**stats, "steps": steps[len(steps) // 2]}


def _measure(label, max_steps=None, probe=False):
    """(stats, None) or (None, why)."""
    try:
        measure_mod.collect(CTX["project"], CTX["run_cmd"], CTX["out"], RUNS, label,
                            say=say, max_steps=max_steps, probe=probe)
    except RuntimeError as exc:
        _record("measure_failed", label=label, error=str(exc)[:300])
        return None, str(exc).removeprefix(f"{label} ")
    return _stats(label), None


def _shape(label):
    """The agent's structure as its compiled graphs recorded it: names, nodes, edges."""
    ids = _ids(label)
    with _conn() as conn:
        return sorted(tuple(r) for r in conn.execute(
            "SELECT name, nodes, edges FROM graphs WHERE session_id = ?", (ids[0],))) if ids else []


def _compare(before, after):
    with _conn() as conn:
        result = measure_mod.compare(conn, _ids(before), _ids(after))
    _record("compare", baseline=before, candidate=after, result=result,
            baseline_state=_state(_ids(before)), candidate_state=_state(_ids(after)))
    return result


async def _judge(label):
    base, cand = _ids("baseline"), _ids(label)
    with _conn() as conn:
        passed, results, correctness = await judge_mod.judge_sessions(conn, CTX["task"], base[0], cand[0],
                                                                      CTX.get("eval_cases"))
    _record("judge", baseline="baseline", candidate=label, passed=passed, equivalence=results,
            correctness=correctness, baseline_state=_state(base), candidate_state=_state(cand))
    return passed, results, correctness


def _gain(result):
    """The first thing that got better, past the noise, and nothing finished less often."""
    if (result.get("completed") or {}).get("verdict") == "regressed":
        return None
    return next((k for k in GAINS if (result.get(k) or {}).get("verdict") in ("improved", "baseline finished nothing")), None)


def _answers_written():
    """Expected answers from the team's cases that the pending change writes into the code.
    A coding agent was caught hardcoding answers for its test inputs (arXiv 2607.18064)."""
    added = " ".join(line[1:] for line in _git("diff", CTX["kept_sha"], "HEAD").splitlines()
                     if line.startswith("+") and not line.startswith("+++"))
    text = evals_mod._norm(added)
    # ponytail: an answer pasted whole or by its opening; a paraphrase gets through, and a
    # person reads the branch before anything is merged
    needles = (evals_mod._norm(c["expected"])[:80] for c in CTX.get("eval_cases") or [])
    return [n for n in needles if len(n) >= 20 and n in text]


def _brief(stats):
    cost = "not priced" if stats.get("cost_usd") is None else f"${stats['cost_usd']:.4f}"
    done = "?" if stats.get("completed") is None else f"{stats['completed']:g}"
    return (f"cost {cost} per run, {stats.get('wall_ms') or 0:,.0f} ms, {stats.get('llm_calls') or 0:g} model calls, "
            f"{stats.get('input_tokens') or 0:,.0f} input and {stats.get('output_tokens') or 0:,.0f} output tokens, "
            f"{done} requests finished, {stats['steps']} steps")


def _row(finding, outcome, detail):
    CTX["rows"][finding] = [outcome, detail]


async def _baseline():
    if CTX.get("baseline_failed"):
        return _ok(f"Refused: the agent as it is could not be measured ({CTX['baseline_failed']}). Nothing can be "
                   "proven in this run; say why in your report.")
    if _dirty():
        return _ok("Refused: measure the agent as it is before changing it. Undo your changes first.")
    ids = _ids("baseline")
    if len(ids) >= RUNS and _state(ids) == CTX["start_state"]:
        say("[fleetopt] the agent as it is was measured before, on this same code: not run again")
        stats = _stats("baseline")
    else:
        say("[fleetopt] measuring the agent as it is")
        stats, why = await asyncio.to_thread(_measure, "baseline")
        if stats is None:
            CTX["baseline_failed"] = why.splitlines()[0]
            return _ok(f"The agent as it is could not be measured: {why}\n\nNothing can be proven in this run. "
                       "Say why in your report.")
    factor = BROKEN_FACTOR if not stats.get("completed") else STEP_FACTOR
    CTX.update(baseline=stats, max_steps=max(MIN_STEPS, factor * stats["steps"]), untracked=_untracked())
    return _ok(f"The agent as it is, medians of {RUNS} runs: {_brief(stats)}.\nEdits are allowed now.")


@tool(
    "measure",
    "Run the agent 3 times as the code stands now and compare it with the code as last kept. "
    "`finding`: the id of the finding the change is for (C1, D2, or N1 for one you found). Leave it "
    "empty once, before any change, to measure the agent as it is. fleetopt commits the change first. "
    "Changed code runs once before it runs three times, and a run that takes far more steps than the "
    "original is stopped. Two measured attempts per finding.",
    {"finding": str},
)
async def measure(args):
    finding = (args.get("finding") or "").strip().upper()
    stop = over()
    if stop:
        return _ok(f"Refused: {stop}. Undo what is not kept, and write your report.")
    if CTX["baseline"] is None:
        return await _baseline()
    if not finding:
        return _ok("The agent as it is was measured already. Name the finding the change is for.")
    if CTX["only"] and finding not in {f["id"] for f in CTX["findings"]}:
        return _ok(f"Refused: a person chose exactly {', '.join(f['id'] for f in CTX['findings'])}. "
                   f"Leave {finding} for your report.")
    pending = CTX["pending"]
    if pending and pending != finding:
        return _ok(f"Refused: {pending} is neither kept nor undone. Call keep or undo for it first.")
    if not _dirty():
        return _ok("Nothing changed since the last measurement. Make the change first.")
    tried = CTX["attempts"].get(finding, 0)
    if tried >= ATTEMPTS:
        return _ok(f"Refused: {finding} has been measured {ATTEMPTS} times, the most a finding gets. Undo it and go on.")
    CTX["attempts"][finding] = tried + 1
    _commit(f"{finding}: trying")
    CTX.update(pending=finding, measured=None, editing=set())
    label = finding if not tried else f"{finding}-{tried + 1}"
    say(f"[fleetopt] {finding}: measuring the change" + (" (second attempt)" if tried else ""))
    stats, why = await asyncio.to_thread(_measure, label, CTX["max_steps"], True)
    CTX["untracked"] = _untracked()
    left = ATTEMPTS - tried - 1
    if stats is None:
        _row(finding, "failed", why.splitlines()[0])
        say(f"[fleetopt] {finding}: {why.splitlines()[0]}")
        return _ok(f"The changed agent could not be measured: {why}\n\nUndo it, or fix what this points to and "
                   f"measure again ({left} attempt(s) left).")
    result = _compare(CTX["kept_label"], label)
    line = compared(finding, result).replace("against the original", "against the code before it")
    say(line)
    CTX["compared"][label] = result
    CTX["measured"] = label
    _row(finding, "measured", line.removeprefix("[fleetopt] "))
    structural = _shape(label) != _shape("baseline")
    notes = ["It got better past the noise: keep judges its answers and keeps it if they hold." if _gain(result)
             else f"Nothing got better past the noise, so keep will refuse it ({left} attempt(s) left)."]
    if structural:
        notes.append("It changes the agent's structure (its nodes or edges)"
                     + (": it is kept only if the team's cases cover every request and it passes them."
                        if CTX["design"] else ": this run is cost only, so it cannot be kept."))
    return _ok(f"{finding} ({label}), medians of {RUNS} runs: {_brief(stats)}.\n{line.removeprefix('[fleetopt] ')}\n\n"
               f"{measure_mod.render(result)}\n\n" + " ".join(notes))


def _refuse(finding, why):
    _row(finding, "not kept", why)
    _record("refused", finding=finding, why=why)
    say(f"[fleetopt] {finding}: not kept, {why}")
    left = ATTEMPTS - CTX["attempts"].get(finding, 0)
    return _ok(f"Refused: {why}.\n\nUndo it, or fix that and measure again ({left} attempt(s) left).")


@tool(
    "keep",
    "Keep the change for a finding on the branch, if it has earned it: something got better past the "
    "noise, the judge finds the answers still hold, and it stays within what this run allows. Otherwise "
    "it is refused, with the reason. `summary`: what you changed, in under 15 words.",
    {"finding": str, "summary": str},
)
async def keep(args):
    finding = (args.get("finding") or "").strip().upper()
    label = CTX["measured"]
    if CTX["pending"] != finding or label is None:
        return _ok(f"Refused: no measured change for {finding} is waiting. Measure it first.")
    if _dirty():
        return _ok("Refused: the code changed after it was measured. Measure it again, or undo.")
    if not _gain(CTX["compared"][label]):
        return _refuse(finding, "nothing got better past the noise")
    structural = _shape(label) != _shape("baseline")
    if structural and not CTX["design"]:
        return _refuse(finding, "it changes the agent's structure (its nodes or edges), and this run is cost only")
    written = _answers_written()
    if written:
        return _refuse(finding, f"it writes an answer from the team's cases into the code ({written[0][:40]!r}...); "
                                "a change has to earn its answers")
    stop = over()
    if stop:
        return _ok(f"Refused: {stop}. Undo what is not kept, and write your report.")
    say(f"[fleetopt] {finding}: judging the answers")
    passed, results, correctness = await _judge(label)
    ok = sum(bool(r["kept_on"]) for r in results)
    if not passed:
        reasons = "; ".join(r["reason"][:150] for r in results if not r["kept_on"])[:600]
        return _refuse(finding, f"the judge failed {len(results) - ok} of {len(results)} requests: {reasons}")
    covered = correctness["matched"] if correctness else 0
    if structural and covered < len(results):
        return _refuse(finding, f"a change to the structure is kept only on the team's cases, and they cover "
                                f"{covered} of {len(results)} requests")
    # one commit per kept finding, whatever it took to get there
    _git("reset", "-q", "--soft", CTX["kept_sha"])
    summary = " ".join((args.get("summary") or "").split())[:100] or "a change"
    _git("-c", "user.name=fleetopt", "-c", "user.email=fleetopt@localhost", "commit", "-q", "-m", f"{finding}: {summary}")
    CTX.update(kept_sha=_git("rev-parse", "HEAD"), kept_label=label, pending=None, measured=None)
    detail = CTX["rows"][finding][1].split(": ", 1)[-1] + f"; judge {ok} of {len(results)} passed"
    _row(finding, "kept", detail)
    _record("keep", finding=finding, label=label)
    say(f"[fleetopt] {finding}: kept. {detail}")
    return _ok(f"Kept: {detail}. The next change is compared with the code as it is now.")


@tool(
    "undo",
    "Put the code back as it was last kept, dropping the change for a finding. `why`: in under 25 words.",
    {"finding": str, "why": str},
)
async def undo(args):
    finding = (args.get("finding") or "").strip().upper() or CTX["pending"] or "?"
    if not CTX["pending"] and not _dirty():
        return _ok("Nothing to undo: the code is as it was last kept.")
    _reset()
    outcome, detail = CTX["rows"].get(finding, [None, ""])
    why = " ".join((args.get("why") or "").split())[:200]
    _row(finding, "undone", detail if outcome in ("failed", "not kept") else (why or detail))
    _record("undo", finding=finding, why=why)
    say(f"[fleetopt] {finding}: undone. {CTX['rows'][finding][1]}")
    return _ok("Undone. The code is as it was last kept.")


def finish():
    """Nothing unproven is left on the branch: a change neither kept nor undone is undone."""
    if not CTX.get("pending") and not _dirty():
        return
    finding = CTX.get("pending") or "(an edit never measured)"
    detail = CTX["rows"].get(finding, [None, ""])[1]
    _reset()
    _row(finding, "undone", (detail + "; " if detail else "") + "left unproven when the session ended")
    _record("undo", finding=finding, why="left unproven when the session ended")


# --- reading: what the sessions look at ------------------------------------------------

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


_TOOLS = [measure, keep, undo, query_traces, graph_topology, graph_shape]


def server(names=None):
    """The in-process tool server; `names` (mcp__fleetopt__* or bare) selects a subset,
    which is how the reviewer gets a read-only one."""
    chosen = _TOOLS if names is None else [
        t for t in _TOOLS if t.name in names or f"mcp__fleetopt__{t.name}" in names]
    return create_sdk_mcp_server(name="fleetopt", tools=chosen)


TOOL_NAMES = [f"mcp__fleetopt__{t.name}" for t in _TOOLS]
