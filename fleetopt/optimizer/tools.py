"""The agent's tools, and the limits inside them.

The agent decides how to start the project's agent, what to change, what to bundle, what
to try next and when to stop. It does not decide what the numbers are or whether a change
is kept, and it cannot go past a limit: those are here, the same for every agent. Git is
fleetopt's alone, so a session cannot fake a baseline.

Limits, each learned on a real agent:
- start: 4 tries, one input each on the team's key; "started" only when a try shows
  requests ran and a model answered, never on the session's word;
- measure: changed code runs once before three times, and a run that takes 3x the
  original's steps is stopped (a fix once removed the only exit from a loop: 170 rounds);
- the team's key, the clock and fleetopt's own spend each end the run;
- keep: better past the noise, the judge passes every request, the graph keeps its nodes
  and edges, and no answer from the team's eval cases is written into the code.
"""

import asyncio
import datetime
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import time

from claude_agent_sdk import create_sdk_mcp_server, tool

from fleetopt.evidence import evals as evals_mod
from fleetopt.evidence import judge as judge_mod
from fleetopt.evidence import measure as measure_mod
from fleetopt.probe import runner, store

RUNS = 3           # runs of the agent per measurement
TRIES = 4          # tries to start the agent, one input each
MAX_CASES = 20     # the team's eval cases run on every measurement, up to this many; FLEETOPT_MAX_CASES overrides
MIN_INPUTS = 3     # without cases: fewer, and the variation in a model's answers hides a real saving (observed: 1)
STEP_FACTOR = 3    # a changed agent may take this many times the original's steps per run
BROKEN_FACTOR = 6  # ... or this many, when the original finished nothing and so stopped early
MIN_STEPS = 150
TEAM_USD = 2.0     # cap on the team's key; FLEETOPT_TEAM_USD overrides
MAX_MINUTES = 120  # the whole run; FLEETOPT_MAX_MINUTES overrides
DRIVER = pathlib.Path(runner.__file__).with_name("driver.py")
SECRET = re.compile(r"key|token|secret|password|credential", re.I)
KEYISH = re.compile(r"^[A-Z][A-Z0-9_]*(?:API_KEY|AUTH_TOKEN)$")

CTX = {}  # set by begin() for one run


def _ok(text):
    return {"content": [{"type": "text", "text": text}]}


def say(line):
    """A line for whoever is watching: what the tools established, in plain words."""
    CTX["said_at"] = time.time()
    print(line, flush=True)


# --- numbers in words ---------------------------------------------------------------------

WORDS = {"improved": "better", "regressed": "worse", "within noise": "no real change",
         "baseline finished nothing": "the original finished nothing"}
SHOWN = (("cost_usd", "cost"), ("input_tokens", "tokens in"), ("output_tokens", "tokens out"),
         ("llm_calls", "model calls"), ("wall_ms", "time"))


def moved(result):
    """What changed past the noise, in words: 'cost -24% · tokens in -28%', or 'no real change'."""
    parts = [f"{word} {v['delta_pct']:+.0f}%" for key, word in SHOWN
             if (v := result.get(key)) and v.get("delta_pct") is not None and v["verdict"] in ("improved", "regressed")]
    done = result.get("completed") or {}
    if done.get("before") is not None and done.get("after") is not None and done["before"] != done["after"]:
        parts.append(f"requests finished {done['before']:g} to {done['after']:g}")
    return " · ".join(parts) or "no real change"


def brief(stats):
    cost = "cost not priced" if stats.get("cost_usd") is None else f"${stats['cost_usd']:.4f} a run"
    return (f"{cost} · {stats.get('input_tokens') or 0:,.0f} tokens in · {stats.get('output_tokens') or 0:,.0f} out · "
            f"{stats.get('llm_calls') or 0:g} model calls · {(stats.get('wall_ms') or 0) / 1000:.1f} s")


GAINS = ("cost_usd", "wall_ms", "completed", "cost_per_completed")


def gain(result):
    """What got better past the noise, or None. Nothing may finish less often and cost may
    not rise. A clean cut in tokens counts too, with neither count worse: total cost can sit
    inside the noise because the length of a model's answers varies from run to run, which
    the change does not control (observed: input tokens -20% on every run, cost -11% called
    noise)."""
    verdict = lambda key: (result.get(key) or {}).get("verdict")
    if "regressed" in (verdict("completed"), verdict("cost_usd")):
        return None
    first = next((k for k in GAINS if verdict(k) in ("improved", "baseline finished nothing")), None)
    tokens = (verdict("input_tokens"), verdict("output_tokens"))
    if first or "regressed" in tokens:
        return first
    return next((k for k, v in zip(("input_tokens", "output_tokens"), tokens) if v == "improved"), None)


# --- limits and records -------------------------------------------------------------------

def over():
    """Why no further run of the agent may start, or None."""
    limit = CTX.get("max_team_usd")
    if limit is not None:
        runs, cost = measure_mod.spent(CTX["out"], CTX["project"], CTX.get("first_session", 0))
        if cost is not None and cost >= limit:
            return f"the limit on the team's key is reached: ${cost:.2f} spent in {runs} runs of the agent, limit ${limit:.2f}"
    if CTX.get("deadline") and time.time() >= CTX["deadline"]:
        return f"the time limit for a run is reached: {CTX.get('max_minutes', 0):g} minutes"
    return None


def _record(event, **data):
    """One line in run.json: what a tool established, never the target's prompts or outputs."""
    CTX.setdefault("events", []).append(
        {"t": datetime.datetime.now().isoformat(timespec="seconds"), "event": event, **data})


def _conn():
    return store.connect(CTX["out"] / "fleetopt.db")


def _state(ids):
    if not ids:
        return None
    with _conn() as conn:
        return conn.execute("SELECT code_state FROM sessions WHERE id = ?", (ids[0],)).fetchone()["code_state"]


def _ids(label):
    """Sessions under a label, for THIS project, that completed, at the code state of the
    newest measurement under that label: a crashed run must not reach a median, the db is
    shared between projects, and labels are reused across days."""
    with _conn() as conn:
        return [r["id"] for r in conn.execute(
            "SELECT id FROM sessions WHERE label = ? AND exit_code = 0 AND project = ?"
            "   AND code_state IS (SELECT code_state FROM sessions WHERE label = ? AND exit_code = 0 AND project = ?"
            "                       ORDER BY id DESC LIMIT 1)",
            (label, str(CTX["project"]), label, str(CTX["project"])))]


# --- how the project's agent is started ---------------------------------------------------

def interpreter(project):
    """The project's own environment; fleetopt's only when it has none."""
    for rel in (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe", "venv/Scripts/python.exe"):
        if (project / rel).exists():
            return str(project / rel)
    return sys.executable


def entry_path(out, project, name="agent"):
    key = hashlib.sha1(str(project).encode()).hexdigest()[:8]
    return pathlib.Path(out) / "entries" / f"{project.name}-{key}-{re.sub(r'[^A-Za-z0-9_.-]+', '-', name)}.json"


def command(path, entry, limit=None):
    parts = [entry["interpreter"], str(DRIVER), str(path)] + (["--limit", str(limit)] if limit else [])
    return subprocess.list2cmdline(parts) if sys.platform == "win32" else shlex.join(parts)


def spread(texts, n):
    """n of them, taken evenly across the list, so they differ in kind."""
    texts = list(dict.fromkeys(texts))
    return texts if len(texts) <= n else [texts[i * len(texts) // n] for i in range(n)]


def team_inputs(project, supplied=None):
    """(cases, inputs, where from): the team's eval cases, supplied or found in the project.
    All their inputs are what the agent is run on, up to MAX_CASES, so every request is
    judged on its expected answer."""
    cases, _ = evals_mod.load(pathlib.Path(supplied).resolve() if supplied else project)
    if not cases:
        if supplied:
            raise ValueError(f"no eval cases could be read from {supplied}: a case is an input and its expected answer")
        return [], [], None
    by_source = {}
    for case in cases:
        by_source.setdefault(case["source"], []).append(case["input"])
    source, texts = max(by_source.items(), key=lambda kv: len(kv[1]))
    texts = list(dict.fromkeys(texts))
    cap = int(os.environ.get("FLEETOPT_MAX_CASES") or MAX_CASES)
    where = pathlib.Path(source).name + (f" ({cap} of its {len(texts)} cases, spread across it)" if len(texts) > cap else
                                         f" (all {len(texts)} cases)")
    return cases, spread(texts, cap), where


def key_names(project):
    """Names of the provider keys set in the environment or the project's .env. Names
    only: a value is never read into fleetopt."""
    names = {k for k in os.environ if KEYISH.match(k)}
    env = project / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            name, _, value = line.strip().removeprefix("export ").partition("=")
            if KEYISH.match(name.strip()) and value.strip():
                names.add(name.strip())
    return sorted(names)


def trial(path, entry, timeout=600):
    """One input, for real, under the probe: what the agent did, and its last lines."""
    raw, traces, _, code = runner.execute(entry["project"], command(path, entry, limit=1), CTX["out"], timeout=timeout)
    runs = []
    if traces.exists():
        for line in traces.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                runs.append(json.loads(line))
            except ValueError:
                pass
    log = raw / "target.log"
    tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]) if log.exists() else ""
    shutil.rmtree(raw, ignore_errors=True)
    roots = [r for r in runs if not r.get("parent_run_id")]
    failed = [r["error"] for r in roots if r.get("error") and not r["error"].startswith("GraphInterrupt")]
    return {"exit": code, "requests": len(roots), "finished": len(roots) - len(failed),
            "model_calls": sum(r.get("run_type") == "llm" for r in runs),
            "answered": sum(r.get("run_type") == "llm" and not r.get("error") for r in runs),
            "error": (failed[0].splitlines() or [""])[0][:300] if failed else None, "tail": tail}


def _entry(plan):
    """The entry the session proposed, checked where code can check it."""
    project = CTX["project"]
    if not isinstance(plan, dict):
        raise ValueError("the entry must be a JSON object")
    graph = plan.get("graph")
    if not isinstance(graph, str) or ":" not in graph:
        raise ValueError(f"no graph named as file.py:name or module:name (got {graph!r})")
    graph = graph.removeprefix("./")
    if graph.split(":")[0].endswith(".py") and not (project / graph.split(":")[0]).exists():
        raise ValueError(f"the graph names {graph.split(':')[0]}, which is not in the project")
    python = interpreter(project)
    if plan.get("interpreter"):
        named = (project / plan["interpreter"]).resolve()
        if not named.is_relative_to(project) or not named.exists():
            raise ValueError(f"the interpreter must be one inside the project; {plan['interpreter']} is not")
        python = str(named)
    template = plan.get("input_template")
    if isinstance(template, str):  # observed: the template sent as JSON text, and the graph got a string
        try:
            template = json.loads(template)
        except ValueError:
            pass
    inputs = CTX["inputs"] or list(dict.fromkeys(t for t in plan.get("inputs") or [] if isinstance(t, str) and t.strip()))
    if len(inputs) < MIN_INPUTS:
        raise ValueError(f"{len(inputs)} input(s) given; give 4 that differ in kind (at least {MIN_INPUTS}), as the "
                         "agent's users would send them")
    env = {k: str(v) for k, v in (plan.get("env") or {}).items() if not SECRET.search(k)}
    return {"project": str(project), "name": plan.get("agent") or "agent", "job": str(plan.get("job") or "")[:300],
            "graph": graph, "paths": [str(p) for p in plan.get("paths") or ["."]], "interpreter": python,
            "env_file": plan.get("env_file"), "env": env, "config": plan.get("config") or {},
            "context": plan.get("context") or {}, "store": plan.get("store"), "input_template": template,
            "inputs": inputs if CTX["inputs"] else spread(inputs, 6), "inputs_from": CTX["inputs_from"] or plan.get("inputs_from") or "written by fleetopt"}


def started(entry):
    """Use this entry from now on: the run command for every measurement."""
    CTX.update(entry=entry, run_cmd=command(CTX["entry_path"], entry), job=entry.get("job") or "answer the user's request")


@tool("start", "Start the agent on one input with this entry, under fleetopt's probe, and say what happened: "
      "requests seen and finished, model calls seen, the first error, the last lines it printed. `entry`: "
      "the entry as a JSON object (see the guide). At most 4 tries, each on the team's key.", {"entry": str})
async def start(args):
    if CTX.get("run_cmd"):
        return _ok("The agent is started already. Measure it.")
    if CTX["tries"] >= TRIES:
        return _ok(f"Refused: {TRIES} tries, the most a start gets. Report what stops it.")
    try:
        entry = _entry(json.loads(args.get("entry") or ""))
    except ValueError as exc:  # JSONDecodeError included
        return _ok(f"Not tried: {exc}.")
    CTX["tries"] += 1
    path = CTX["entry_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=1), encoding="utf-8")
    result = await asyncio.to_thread(trial, path, entry)
    ok = bool(result["exit"] == 0 and result["requests"] and result["answered"])
    with path.with_suffix(".log").open("a", encoding="utf-8") as log:
        log.write(f"--- try {CTX['tries']}, {datetime.datetime.now():%H:%M:%S}\n{result['tail']}\n")
    facts = (f"exit {result['exit']}; {result['requests']} request(s) seen, {result['finished']} finished; "
             f"{result['model_calls']} model call(s) seen; first error: {result['error'] or 'none'}")
    if ok:
        entry["proven"] = datetime.datetime.now().isoformat(timespec="seconds")
        entry["broken"] = None if result["finished"] else result["error"]
        path.write_text(json.dumps(entry, indent=1), encoding="utf-8")
        started(entry)
        say(f"  started: {entry['name']} ({entry['graph'].rsplit('/', 1)[-1]}), {len(entry['inputs'])} test inputs "
            f"from {entry['inputs_from']}, each run" + (f" (try {CTX['tries']})" if CTX["tries"] > 1 else ""))
        verdict = "It started. Now measure it." + ("" if result["finished"] else
                                                   " No request finished: it is measured and reviewed as broken.")
    elif result["requests"] and not result["model_calls"]:
        say(f"  try {CTX['tries']}: it ran, and no model call was seen")
        verdict = "It ran, and fleetopt saw no model call: find out how it calls its model."
    else:
        say(f"  try {CTX['tries']}: did not start ({(result['error'] or 'see its output')[:100]})")
        verdict = "It did not start."
    return _ok(f"{verdict}\n{facts}\n\nLast lines it printed:\n{result['tail']}\n\n{TRIES - CTX['tries']} try(s) left.")


# --- the run: git, measurement and the gate -----------------------------------------------

def begin(project, out, *, entry_file, entry=None, cases=(), inputs=(), inputs_from=None, look_only=False,
          team_usd=TEAM_USD, minutes=MAX_MINUTES, first_session=0, reuse_baseline=False):
    """Everything the tools share for one run, from the code as it stands."""
    CTX.clear()
    CTX.update(project=pathlib.Path(project).resolve(), out=pathlib.Path(out).resolve(), events=[], entry_path=entry_file,
               eval_cases=list(cases) or None, inputs=list(inputs), inputs_from=inputs_from, look_only=look_only,
               max_team_usd=team_usd, max_minutes=minutes, deadline=time.time() + 60 * minutes,
               first_session=first_session, said_at=time.time(), tries=0, run_cmd=None, job="answer the user's request",
               baseline=None, reuse_baseline=reuse_baseline, saved=[], measured=None, results={}, changes={}, n=0)
    CTX.update(start_sha=_git("rev-parse", "HEAD"), start_state=runner.code_state(CTX["project"]))
    CTX.update(kept_sha=CTX["start_sha"], kept_label="baseline", start_untracked=_untracked())
    CTX["untracked"] = set(CTX["start_untracked"])
    if entry:
        started(entry)


def _git(*args):
    done = subprocess.run(["git", "-C", str(CTX["project"]), *args], capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.strip()[:200]}")
    return done.stdout.strip()


def _untracked():
    return set(_git("ls-files", "--others", "--exclude-standard").splitlines())


def _dirty():
    """Edits not yet saved: changed tracked files, or files created since the last run."""
    return bool(_git("status", "--porcelain", "--untracked-files=no")) or bool(_untracked() - CTX["untracked"])


def _reset():
    """Back to the code as last kept, and nothing half-made left behind."""
    _git("reset", "-q", "--hard", CTX["kept_sha"])
    for path in _untracked() - CTX["start_untracked"]:
        target = (CTX["project"] / path).resolve()
        if not target.is_relative_to(CTX["out"]):  # fleetopt's own records, if they live in the project
            target.unlink(missing_ok=True)
    CTX.update(saved=[], measured=None, untracked=_untracked())


def _measure(label, max_steps=None, probe=False):
    """(stats, None) or (None, why)."""
    try:
        measure_mod.collect(CTX["project"], CTX["run_cmd"], CTX["out"], RUNS, label,
                            say=say, max_steps=max_steps, probe=probe)
    except RuntimeError as exc:
        _record("measure_failed", label=label, error=str(exc)[:300])
        return None, str(exc).removeprefix(f"{label} ")
    return _stats(label), None


def _stats(label):
    ids = _ids(label)
    with _conn() as conn:
        stats, _ = measure_mod.aggregate(conn, ids)
        steps = sorted(conn.execute("SELECT COUNT(*) FROM runs WHERE session_id = ?", (i,)).fetchone()[0] for i in ids)
    _record("measure", label=label, n=len(ids), stats=stats, code_state=_state(ids))
    return {**stats, "steps": steps[len(steps) // 2]}


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
        passed, results, correctness = await judge_mod.judge_sessions(conn, CTX["job"], base[0], cand[0],
                                                                      CTX.get("eval_cases"))
    _record("judge", baseline="baseline", candidate=label, passed=passed, equivalence=results,
            correctness=correctness, baseline_state=_state(base), candidate_state=_state(cand))
    return passed, results


def _answers_written():
    """Expected answers from the team's cases written into the code since it was last kept.
    A coding agent was caught hardcoding answers for its test inputs (arXiv 2607.18064)."""
    added = " ".join(line[1:] for line in _git("diff", CTX["kept_sha"], "HEAD").splitlines()
                     if line.startswith("+") and not line.startswith("+++"))
    text = evals_mod._norm(added)
    # ponytail: an answer pasted whole or by its opening; a paraphrase gets through, and a
    # person reads the branch before anything is merged
    needles = (evals_mod._norm(c["expected"])[:80] for c in CTX.get("eval_cases") or [])
    return [n for n in needles if len(n) >= 20 and n in text]


def _requests():
    n = len(CTX["entry"]["inputs"])
    return f"{n} request{'s' * (n != 1)} each, the same {n} input{'s' * (n != 1)} every run"


def _mark(outcome, detail):
    for name in CTX["saved"]:
        CTX["changes"][name] = [outcome, detail]


async def _baseline():
    ids = _ids("baseline")
    if CTX["reuse_baseline"] and len(ids) >= RUNS and _state(ids) == CTX["start_state"]:
        say("  measuring it as it is: measured before on this same code, not run again")
        stats = _stats("baseline")
    else:
        say(f"  measuring it as it is ({RUNS} runs)")
        stats, why = await asyncio.to_thread(_measure, "baseline")
        if stats is None:
            CTX["baseline_failed"] = why.splitlines()[0]
            say(f"  could not measure it: {CTX['baseline_failed'][:160]}")
            return _ok(f"The agent as it is could not be measured: {why}\n\nNothing can be proven. Say why and stop.")
    if not stats.get("llm_calls"):
        CTX["baseline_failed"] = "no model call was seen"
        say("  it ran, and no model call was seen: fleetopt cannot see what it spends")
        return _ok("It ran and no model call was recorded: it calls its model without LangChain. Find where "
                   "(file and line), say so, and stop.")
    factor = BROKEN_FACTOR if not stats.get("completed") else STEP_FACTOR
    CTX.update(baseline=stats, max_steps=max(MIN_STEPS, factor * stats["steps"]), untracked=_untracked())
    say(f"  as it is: {brief(stats)}")
    return _ok(f"The agent as it is, medians of {RUNS} runs of {_requests()}: {brief(stats)}; {stats.get('completed')} requests "
               f"finished, {stats['steps']} steps a run." + ("" if CTX["look_only"] else " Edits are allowed now."))


@tool("measure", "Run the agent 3 times as the code stands and compare it with the code as last kept. The first "
      "time, before any edit, it measures the agent as it is. After that it measures what you saved with "
      "save_change; changed code runs once before three times, and a run that takes far more steps than the "
      "original is stopped.", {})
async def measure(args):
    if not CTX.get("run_cmd"):
        return _ok("Refused: start the agent first.")
    if CTX.get("baseline_failed"):
        return _ok(f"Refused: the agent as it is could not be measured ({CTX['baseline_failed']}). Say why and stop.")
    stop = over()
    if stop:
        return _ok(f"Refused: {stop}. Undo what is not kept, and report.")
    if CTX["baseline"] is None:
        return await _baseline()
    if _dirty():
        return _ok("Refused: there are unsaved edits. save_change them with a plain name, or undo.")
    if not CTX["saved"]:
        return _ok("Nothing saved since the code was last kept, so nothing new to measure.")
    if CTX["measured"]:
        return _ok("This code was measured already: keep or undo it, or save another change and measure again.")
    CTX["n"] += 1
    label, names = f"change-{CTX['n']}", " + ".join(CTX["saved"])
    say(f"  measuring: {names}")
    stats, why = await asyncio.to_thread(_measure, label, CTX["max_steps"], True)
    CTX["untracked"] = _untracked()
    if stats is None:
        _mark("failed", why.splitlines()[0][:160])
        say(f"  could not measure it: {why.splitlines()[0][:160]}")
        return _ok(f"The changed agent could not be measured: {why}\n\nUndo it, or fix what this points to, save the "
                   "fix and measure again.")
    result = _compare(CTX["kept_label"], label)
    CTX["results"][label], CTX["measured"] = result, label
    line = moved(result)
    _mark("measured", line)
    say(f"  measured: {line}")
    notes = ["Something got better past the noise: keep judges the answers and keeps it if they hold." if gain(result)
             else "Nothing got better past the noise, so keep will refuse it."]
    if _shape(label) != _shape("baseline"):
        notes.append("It changes the graph's nodes or edges, so keep will refuse it.")
    return _ok(f"{names}, medians of {RUNS} runs of {_requests()}: {brief(stats)}.\nAgainst the code as last kept: {line}\n\n"
               f"{measure_mod.render(result)}\n\n" + " ".join(notes))


@tool("save_change", "Save your edits as one change, its own commit, named in plain words for the team "
      "(e.g. 'cache the system prompt'). Save several changes you are sure of, then measure them together.",
      {"name": str})
async def save_change(args):
    name = " ".join((args.get("name") or "").split())[:80]
    if not name:
        return _ok("Refused: name the change in plain words.")
    if CTX["baseline"] is None:
        return _ok("Refused: measure the agent as it is first.")
    if not _dirty():
        return _ok("Nothing changed since the last save.")
    new = sorted(_untracked() - CTX["untracked"])
    _git("add", "-u")
    if new:
        _git("add", "--", *new)
    _git("-c", "user.name=fleetopt", "-c", "user.email=fleetopt@localhost", "commit", "-q", "-m", name)
    CTX["saved"].append(name)
    CTX.update(measured=None, untracked=_untracked())
    CTX["changes"][name] = ["saved", ""]
    say(f"  changed: {name}")
    return _ok(f"Saved '{name}' as its own commit. Save more, or measure.")


def _refuse(why):
    _mark("not kept", why)
    _record("refused", changes=list(CTX["saved"]), why=why)
    say(f"  not kept: {why}")
    return _ok(f"Refused: {why}.\n\nUndo it, or change what this points to, save the fix and measure again.")


@tool("keep", "Keep what was saved and measured since the code was last kept, if it has earned it: something got "
      "better past the noise, the judge finds the answers still hold, the graph keeps its nodes and edges. "
      "Otherwise it is refused, with the reason.", {})
async def keep(args):
    label = CTX["measured"]
    if not CTX["saved"] or not label:
        return _ok("Refused: nothing measured is waiting. Save and measure first.")
    if _dirty():
        return _ok("Refused: there are edits since the measurement. Save and measure them, or undo.")
    result = CTX["results"][label]
    if not gain(result):
        return _refuse("nothing got better past the noise")
    if _shape(label) != _shape("baseline"):
        return _refuse("it changes the graph's nodes or edges, which is a design change")
    written = _answers_written()
    if written:
        return _refuse(f"it writes an answer from the team's eval cases into the code ({written[0][:40]!r}...)")
    stop = over()
    if stop:
        return _ok(f"Refused: {stop}. Undo what is not kept, and report.")
    passed, results = await _judge(label)
    ok = sum(bool(r["kept_on"]) for r in results)
    if not passed:
        reasons = "; ".join(r["reason"][:150] for r in results if not r["kept_on"])[:600]
        return _refuse(f"answers changed on {len(results) - ok} of {len(results)} requests: {reasons}")
    detail = f"{moved(result)} · answers hold {ok}/{len(results)}"
    names = list(CTX["saved"])
    _mark("kept", detail)
    CTX.update(kept_sha=_git("rev-parse", "HEAD"), kept_label=label, saved=[], measured=None)
    _record("keep", changes=names, label=label)
    say(f"  kept: {' + '.join(names)} ({detail})")
    return _ok(f"Kept: {detail}. What comes next is compared with the code as it is now.")


@tool("undo", "Put the code back as it was last kept, dropping every change saved since. `why`: a few words.",
      {"why": str})
async def undo(args):
    if not CTX["saved"] and not _dirty():
        return _ok("Nothing to undo: the code is as it was last kept.")
    why = " ".join((args.get("why") or "").split())[:160]
    names = list(CTX["saved"])
    for name in names:
        outcome, detail = CTX["changes"].get(name, ["", ""])
        CTX["changes"][name] = ["undone", detail if outcome in ("failed", "not kept") else (why or detail)]
    _reset()
    _record("undo", changes=names, why=why)
    say(f"  undone: {' + '.join(names) or 'unsaved edits'}" + (f" ({why})" if why else ""))
    return _ok("Undone. The code is as it was last kept.")


def finish():
    """Nothing unproven is left on the branch."""
    if not CTX.get("saved") and not _dirty():
        return
    for name in CTX["saved"]:
        detail = CTX["changes"].get(name, ["", ""])[1]
        CTX["changes"][name] = ["undone", (detail + "; " if detail else "") + "not proven when the session ended"]
    _reset()
    _record("undo", why="not proven when the session ended")


@tool("query", "Read-only SQL on what fleetopt recorded. Tables: runs(session_id, run_id, parent_run_id, trace_id, "
      "name, run_type, node, step, model, provider, duration_ms, input_tokens, output_tokens, cache_read_tokens, "
      "cache_write_tokens, prompt_chars, prompt, completion, inputs, outputs, error), sessions(id, label, "
      "code_state, exit_code), graphs(session_id, name, nodes, edges, mermaid). Prefer aggregates.", {"sql": str})
async def query(args):
    sql = args["sql"].strip()
    if not sql.lower().startswith(("select", "with")):
        return _ok("refused: read-only, SELECT statements only")
    with _conn() as conn:
        try:
            rows = conn.execute(sql).fetchall()
        except Exception as exc:  # noqa: BLE001 - a bad query is the session's to fix
            return _ok(f"error: {exc}")
    if not rows:
        return _ok("(no rows)")
    out = [" | ".join(rows[0].keys())] + [" | ".join(str(r[c])[:200] for c in rows[0].keys()) for r in rows[:100]]
    if len(rows) > 100:
        out.append(f"... {len(rows) - 100} more rows")
    return _ok("\n".join(out))


LOOK = [start, measure, query]
CHANGE = LOOK + [save_change, keep, undo]


def server(look_only=False):
    return create_sdk_mcp_server(name="fleetopt", tools=LOOK if look_only else CHANGE)


def names(look_only=False):
    return [f"mcp__fleetopt__{t.name}" for t in (LOOK if look_only else CHANGE)]
