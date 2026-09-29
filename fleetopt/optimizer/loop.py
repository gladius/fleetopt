"""The loop behind `fleetopt apply`, driven by code.

Before this a model drove it: one long session decided what to try, how often to try
again, and when to measure. Observed on 2026-09-29: four attempts at one fix, the
original measured again for no reason, 25 minutes of output nobody could read, and a
changed agent that looped 170 rounds on the team's key with nothing watching it.

Now the procedure is code, the same for every finding:

    edit    a session makes one change and stops (session.py; it has no git, no measure)
    commit  code commits it, one commit per finding
    try     one run of the changed agent, stopped if it takes far more steps than the original
    measure the rest of the runs
    compare with the code before it: did anything get better?
    judge   the answers, against the team's cases where they cover a request
    keep    or undo, and at most one more attempt, told why the first failed

Time and cost follow from the number of findings, and are said before it starts.
"""

import datetime
import os
import pathlib
import re
import subprocess
import time

from fleetopt.evidence import evals as evals_mod
from fleetopt.evidence import judge as judge_mod
from fleetopt.evidence import measure as measure_mod
from fleetopt.optimizer import session, tools
from fleetopt.probe import runner

RUNS = 3          # runs of the agent per measurement; never more, never again for the same code
ATTEMPTS = 2      # per finding: the change, and one more told why the first failed
STEP_FACTOR = 3   # a changed agent may take this many times the original's steps per run
BROKEN_FACTOR = 6  # ... or this many, when the original finished nothing and so stopped early
MIN_STEPS = 150
TEAM_USD = 2.0    # cap on the team's key; FLEETOPT_TEAM_USD overrides
MAX_MINUTES = 120  # the whole run; FLEETOPT_MAX_MINUTES overrides
GAINS = ("cost_usd", "wall_ms", "completed", "cost_per_completed")


def _git(project, *args):
    done = subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.strip()[:200]}")
    return done.stdout.strip()


def _untracked(project):
    return set(_git(project, "ls-files", "--others", "--exclude-standard").splitlines())


def _commit(project, message, before):
    """Commit what the edit session changed: tracked files, and files it created. Not
    files the agent wrote into its own repository while it ran (observed)."""
    new = sorted(_untracked(project) - before)
    _git(project, "add", "-u")
    if new:
        _git(project, "add", "--", *new)
    if not _git(project, "diff", "--cached", "--name-only"):
        return False
    _git(project, "-c", "user.name=fleetopt", "-c", "user.email=fleetopt@localhost", "commit", "-q", "-m", message)
    return True


def _undo(project):
    _git(project, "reset", "-q", "--hard", "HEAD~1")


def _measure(label, max_steps=None, probe=False):
    """(stats, None) or (None, why). Records the measurement for the verdict."""
    stop = tools.over()
    if stop:
        return None, stop
    try:
        measure_mod.collect(tools.CTX["project"], tools.CTX["run_cmd"], tools.CTX["out"], RUNS, label,
                            say=tools.say, max_steps=max_steps, probe=probe)
    except RuntimeError as exc:
        tools._record("measure_failed", label=label, error=str(exc)[:300])
        return None, str(exc).splitlines()[0].removeprefix(f"{label} ")
    ids = tools._ids(label)
    with tools._conn() as conn:
        stats, _ = measure_mod.aggregate(conn, ids)
        steps = sorted(conn.execute("SELECT COUNT(*) FROM runs WHERE session_id = ?", (i,)).fetchone()[0] for i in ids)
    state = tools._state(ids)
    tools.CTX.setdefault("baseline_state", state)
    tools._record("measure", label=label, n=RUNS, sessions=len(ids), stats=stats, code_state=state)
    return {**stats, "steps": steps[len(steps) // 2]}, None


def _shape(label):
    """The agent's structure as its compiled graphs recorded it: names, nodes, edges."""
    ids = tools._ids(label)
    with tools._conn() as conn:
        return sorted(tuple(r) for r in conn.execute(
            "SELECT name, nodes, edges FROM graphs WHERE session_id = ?", (ids[0],))) if ids else []


def _compare(before, after):
    with tools._conn() as conn:
        result = measure_mod.compare(conn, tools._ids(before), tools._ids(after))
    b, c = tools._state(tools._ids(before)), tools._state(tools._ids(after))
    tools._record("compare", baseline=before, candidate=after, result=result, baseline_state=b, candidate_state=c)
    return result


async def _judge(task, label):
    base, cand = tools._ids("baseline"), tools._ids(label)
    with tools._conn() as conn:
        passed, results, correctness = await judge_mod.judge_sessions(conn, task, base[0], cand[0],
                                                                      tools.CTX.get("eval_cases"))
    tools._record("judge", baseline="baseline", candidate=label, passed=passed, equivalence=results,
                  correctness=correctness, baseline_state=tools._state(base), candidate_state=tools._state(cand))
    return passed, results


async def _edit(project, finding, review, feedback, model, max_usd, start_branch):
    """One change for one finding. Returns (reply line, cost)."""
    from claude_agent_sdk import AssistantMessage, ClaudeSDKError, ResultMessage, TextBlock, query

    options = session.build_options(project, tools.CTX["run_cmd"], model, max_usd=max_usd, start_branch=start_branch)
    prompt = (f"The finding to act on is {finding['id']}: {finding['title']}.\n"
              + (f"\nAn earlier attempt at it was undone, because: {feedback}\n" if feedback else "")
              + ("\nIt changes the design. List to yourself what must survive before you edit, and keep all of it.\n"
                 if finding["kind"] != "cost" else "")
              + f"\n--- the review ---\n{review}")
    texts, cost = [], 0.0
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                texts += [b.text for b in message.content if isinstance(b, TextBlock) and b.text.strip()]
            elif isinstance(message, ResultMessage):
                cost = getattr(message, "total_cost_usd", None) or 0.0
                if message.result:
                    texts.append(message.result)
    except ClaudeSDKError as exc:
        return f"CANNOT: the session ended without an answer ({str(exc).splitlines()[0][:80]})", cost
    lines = [line.strip() for text in texts for line in text.splitlines() if line.strip()]
    reply = next((line for line in reversed(lines) if re.match(r"(DONE|CANNOT):", line)), None)
    return reply or "DONE: " + (lines[-1][:90] if lines else "a change"), cost


def _gain(result):
    """The first thing that got better, past the noise, and nothing finished less often."""
    if (result.get("completed") or {}).get("verdict") == "regressed":
        return None
    return next((k for k in GAINS if (result.get(k) or {}).get("verdict") in ("improved", "baseline finished nothing")), None)


def _report(rows, total_line):
    lines = ["| Finding | Outcome | What was measured |", "|---|---|---|"]
    lines += [f"| {f['id']} {f['title']} | {outcome} | {detail} |" for f, outcome, detail in rows]
    return "\n".join(["## What was tried", "", *lines, "", total_line])


async def run(project, out_dir, run_cmd, review, findings, *, task, model=None, max_usd=5.0, evals=None,
              first_session=0):
    """Try each of `findings` of the `review` text, in order, on a new branch. Returns
    the facts the summary is computed from."""
    project, out = pathlib.Path(project).resolve(), pathlib.Path(out_dir).resolve()
    started = datetime.datetime.now()
    run_dir = out / "runs" / f"{started:%Y%m%d-%H%M%S}-{project.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_branch, start_sha = _git(project, "rev-parse", "--abbrev-ref", "HEAD"), _git(project, "rev-parse", "HEAD")
    start_state = runner.code_state(project)
    minutes = float(os.environ.get("FLEETOPT_MAX_MINUTES") or MAX_MINUTES)
    tools.CTX.clear()
    tools.CTX.update({"project": project, "out": out, "run_cmd": run_cmd, "events": [], "findings": findings,
                      "max_team_usd": float(os.environ.get("FLEETOPT_TEAM_USD") or TEAM_USD),
                      "first_session": first_session, "max_minutes": minutes,
                      "deadline": time.time() + 60 * minutes, "said_at": time.time(), "run_dir": run_dir})
    cases, _ = evals_mod.load(pathlib.Path(evals).resolve() if evals else project)
    tools.CTX["eval_cases"] = cases or None

    branch = f"fleetopt/{started:%Y%m%d-%H%M%S}"
    _git(project, "checkout", "-q", "-b", branch)
    rows, own, log = [], 0.0, []
    per_edit = max(0.5, min(1.5, max_usd / (len(findings) * ATTEMPTS + 1)))

    tools.say("\n[fleetopt] measuring the agent as it is")
    base, why = _measure("baseline")
    if base is None:
        rows = [(f, "not tried", f"the agent as it is could not be measured: {why}") for f in findings]
    else:
        factor = BROKEN_FACTOR if not base.get("completed") else STEP_FACTOR
        max_steps = max(MIN_STEPS, factor * base["steps"])
        previous = "baseline"
        for i, finding in enumerate(findings, 1):
            tools.say(f"\n[fleetopt] finding {i} of {len(findings)}: {finding['id']} {finding['title']}")
            feedback, outcome = None, None
            for attempt in range(1, ATTEMPTS + 1):
                stop = tools.over() or (f"fleetopt's own spend reached ${own:.2f}" if own >= max_usd else None)
                if stop:
                    outcome = ("not tried", stop)
                    tools.say(f"[fleetopt] stopping here: {stop}")
                    break
                tools.say(f"[fleetopt] {finding['id']}: making the change" + (" (second attempt)" if attempt > 1 else ""))
                before = _untracked(project)
                reply, cost = await _edit(project, finding, review, feedback, model, per_edit, start_branch)
                own += cost
                log.append(f"{finding['id']} attempt {attempt}: {reply} (${cost:.2f})")
                label = finding["id"] if attempt == 1 else f"{finding['id']}-{attempt}"
                if reply.startswith("CANNOT") or not _commit(project, f"{finding['id']}: {reply.removeprefix('DONE:').strip()}", before):
                    for path in _untracked(project) - before:  # nothing half-made is left behind
                        (project / path).unlink(missing_ok=True)
                    _git(project, "checkout", "-q", "--", ".")
                    outcome = ("not changed", reply.removeprefix("CANNOT:").strip() or "no change was made")
                    tools.say(f"[fleetopt] {finding['id']}: not changed. {outcome[1]}")
                    break
                stats, why = _measure(label, max_steps=max_steps, probe=True)
                if stats is None:
                    _undo(project)
                    feedback, outcome = why, ("undone", why)
                    tools.say(f"[fleetopt] {finding['id']}: undone. {why}")
                    if tools.over():
                        break
                    continue
                if finding["kind"] == "cost" and _shape(label) != _shape("baseline"):
                    # what a cost finding is, enforced: the graph keeps its nodes and edges
                    _undo(project)
                    why = "it changed the agent's structure (its steps or how they connect); a cost change must not"
                    feedback, outcome = why, ("undone", why)
                    tools.say(f"[fleetopt] {finding['id']}: undone, {why}")
                    continue
                result = _compare(previous, label)
                moved = tools.compared(finding["id"], result).replace("against the original", "against the code before it")
                tools.say(moved)
                gain = _gain(result)
                if not gain:
                    _undo(project)
                    feedback, outcome = "nothing got better past the noise", ("undone", "no real gain")
                    tools.say(f"[fleetopt] {finding['id']}: undone, no real gain")
                    continue
                passed, results = await _judge(task, label)
                ok = sum(bool(r["kept_on"]) for r in results)
                if not passed:
                    _undo(project)
                    reasons = "; ".join(r["reason"][:120] for r in results if not r["kept_on"])[:400]
                    feedback, outcome = f"the judge failed {len(results) - ok} of {len(results)} requests: {reasons}", \
                        ("undone", f"judge failed, {ok} of {len(results)} requests passed")
                    tools.say(f"[fleetopt] {finding['id']}: undone, {ok} of {len(results)} requests passed the judge")
                    continue
                previous, outcome = label, ("kept", moved.split(": ", 1)[-1] + f"; judge {ok} of {len(results)} passed")
                tools.say(f"[fleetopt] {finding['id']}: kept. {outcome[1]}")
                break
            rows.append((finding, *outcome))
            if outcome[0] == "not tried":
                rows += [(f, "not tried", outcome[1]) for f in findings[i:]]
                break

    kept = int(_git(project, "rev-list", "--count", f"{start_sha}..HEAD") or 0)
    total = "Nothing was kept. The branch holds the code it started from."
    if kept:
        whole = _compare("baseline", previous)
        total = tools.compared("All kept changes", whole)
    final_state = runner.code_state(project)
    events = tools.CTX["events"]
    computed = session.verdict(events, final_state, start_state)
    tried, gained = session.numbers(events, computed, start_state, final_state)
    report = _report(rows, total.removeprefix("[fleetopt] "))
    print("\n--- report ---\n" + report)
    print(f"\n--- fleetopt verdict (computed from the measurements) ---\n{computed}")
    facts = {"verdict": computed, "tried": tried, "kept": kept, "gained": gained, "branch": branch,
             "own_cost_usd": own, "run_dir": str(run_dir)}
    meta = {"model": model, "evals_path": evals, "max_usd": max_usd, "findings": findings,
            "code_state_after": final_state, "start_branch": start_branch, **facts,
            "rows": [{"id": f["id"], "outcome": o, "detail": d} for f, o, d in rows]}
    try:
        session._write_record(run_dir, project, start_sha, started, meta, [report], [], [], {}, log)
    except OSError as exc:
        print(f"[fleetopt] could not write the run record: {exc}")
    return facts
