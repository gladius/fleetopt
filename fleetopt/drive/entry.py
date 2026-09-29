"""How to start a project's agent: worked out once, proven once, remembered.

A read-only session reads the project the way a developer joining the team would, and
answers with an entry (setup.py, guided by SETUP.md): which graph, how it is called,
which settings, which inputs, or what only the team can provide. Code checks that
answer the only way that counts, by loading the agent and starting it with one input,
and gives the session one more try with what happened. What it settles on is a small
JSON file in fleetopt's own folder, never in the team's repo. The next run reads it.

What stays code is what keeps a developer's machine safe and the answer honest: the
project's own interpreter, nothing installed, no secret read, the trial run.
"""

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

from fleetopt import config
from fleetopt.evidence import evals

DRIVER = pathlib.Path(__file__).with_name("driver.py")
MAX_INPUTS = 4
ATTEMPTS = 2
KEYISH = re.compile(r"^[A-Z][A-Z0-9_]*(?:API_KEY|AUTH_TOKEN)$")


class Unstartable(RuntimeError):
    """fleetopt could not work out how to start this agent. The message says what it
    found, what it tried and what failed."""


class NotReady(Unstartable):
    """The project is not in a state to run, and what is missing is the team's to
    provide: an environment, a key, a service. fleetopt adds nothing to a developer's
    machine; it says what is missing, all of it at once, and stops."""

    def __init__(self, problems):
        self.problems = problems
        n = len(problems)
        listed = "\n".join(f"  {i}. {p}" for i, p in enumerate(problems, 1))
        super().__init__(f"fleetopt cannot start this agent yet. {n} thing{'s' if n > 1 else ''} to set up, "
                         f"then run the same command again:\n\n{listed}")


REFUSED = re.compile(r"authentication|api[_ -]?key|\b401\b|unauthorized|invalid x-api-key|credit balance|quota|billing", re.I)
UNREACHABLE = re.compile(r"connection refused|could not connect|connecterror|name or service not known|"
                         r"temporary failure in name resolution|max retries exceeded", re.I)


def interpreter(project):
    """The project's own environment; the one fleetopt runs in only when it has none."""
    for rel in (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe", "venv/Scripts/python.exe"):
        if (project / rel).exists():
            return str(project / rel), "the project's own environment"
    return sys.executable, "fleetopt's interpreter, because the project has no .venv of its own"


def team_inputs(project, supplied=None):
    """(inputs, where they came from) from eval cases: those supplied, else the team's
    own. Cases are the inputs, not only the answers: a request that matches no case
    cannot be judged on one. ([], None) when there are none."""
    cases, _ = evals.load(pathlib.Path(supplied).resolve() if supplied else project)
    if not cases:
        if supplied:
            raise Unstartable(f"no eval cases could be read from {supplied}. A case is an input and the answer "
                              "expected for it")
        return [], None
    by_source = {}
    for case in cases:
        by_source.setdefault(case["source"], []).append(case["input"])
    source, texts = max(by_source.items(), key=lambda kv: len(kv[1]))
    where = pathlib.Path(source).name if supplied else pathlib.Path(source).relative_to(project)
    return _spread(texts), (f"the eval cases you supplied, {where}" if supplied else f"the team's eval cases, {where}")


def _spread(texts, n=MAX_INPUTS):
    """n of them, taken evenly across the list, so they differ in kind."""
    texts = list(dict.fromkeys(texts))
    return texts if len(texts) <= n else [texts[i * len(texts) // n] for i in range(n)]


def path_for(out, project, name):
    key = hashlib.sha1(str(project).encode()).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return pathlib.Path(out) / "entries" / f"{project.name}-{key}-{safe}.json"


def command(entry_path, entry, limit=None):
    parts = [entry["interpreter"], str(DRIVER), str(entry_path)] + (["--limit", str(limit)] if limit else [])
    return subprocess.list2cmdline(parts) if sys.platform == "win32" else shlex.join(parts)


DONE = re.compile(r"^\[driver\] (\d+) finished, (\d+) paused", re.M)
ANSWERED = "[fleetopt] the model answered {n} call(s) before the request failed"


def prove(entry_path, entry, timeout=600):
    """One input, for real, under the probe. Returns (worked, what the agent printed
    last). When the request failed after the model had answered, the last line says so:
    that agent starts, and what it does next is its own."""
    from fleetopt.probe import runner

    out = pathlib.Path(entry_path).parent.parent
    raw, traces, _, code = runner.execute(entry["project"], command(entry_path, entry, limit=1), out,
                                          timeout=timeout)
    if code == runner.TIMED_OUT:
        shutil.rmtree(raw, ignore_errors=True)
        return False, f"no answer within {timeout} seconds"
    text = (raw / "target.log").read_text(encoding="utf-8", errors="replace")
    answered = 0
    if traces.exists():
        for line in traces.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                run = json.loads(line)
            except ValueError:
                continue
            answered += run.get("run_type") == "llm" and not run.get("error")
    shutil.rmtree(raw, ignore_errors=True)
    done = DONE.search(text)
    worked = code == 0 and bool(done) and int(done.group(1)) + int(done.group(2)) > 0
    tail = "\n".join(text.splitlines()[-30:])
    return worked, tail + ("\n" + ANSWERED.format(n=answered) if answered and not worked else "")


def failure_line(tail):
    """The one line that says why a trial did not answer."""
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if line.startswith("[driver] FAILED") and i + 1 < len(lines):
            return lines[i + 1][:200]
        if line.startswith("[driver] could not load"):
            return line.removeprefix("[driver] ")[:200]
    return (lines[-1] if lines else "it printed nothing")[:200]


def starts_but_fails(tail):
    """The agent's own failure, when the model had answered before it: one line."""
    if ANSWERED.split("{n}")[0] not in tail:
        return None
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    failed = next((i for i, l in enumerate(lines) if l.startswith("[driver] FAILED")), None)
    return lines[failed + 1][:300] if failed is not None and failed + 1 < len(lines) else "it raised"


def install_hint(project):
    """The command that gives this project its environment, in its own terms."""
    for file, how in (("uv.lock", "uv sync"), ("poetry.lock", "poetry install"),
                      ("requirements.txt", "python -m venv .venv && .venv/bin/pip install -r requirements.txt"),
                      ("pyproject.toml", "python -m venv .venv && .venv/bin/pip install -e .")):
        if (project / file).exists():
            return f"It has a {file}, so: {how}"
    return "Its README should say how to install it"


def preflight(entry_path, entry):
    """What stands between this project and a first run, found without calling a
    model and in a few seconds. Only what is the team's to provide; what fleetopt can
    work out itself (which graph, the shape of the input) is left to the trial."""
    project = pathlib.Path(entry["project"])
    own = pathlib.Path(entry["interpreter"]).is_relative_to(project)
    problems = []
    env_file = entry.get("env_file")
    if env_file and not (project / env_file).exists():
        copy = ". Copy .env.example to it and fill it in" if (project / ".env.example").exists() else ""
        problems.append(f"{env_file} does not exist. The project reads its keys and settings from it{copy}")

    parts = [entry["interpreter"], str(DRIVER), str(entry_path), "--check"]
    try:
        done = subprocess.run(subprocess.list2cmdline(parts) if sys.platform == "win32" else shlex.join(parts),
                              shell=True, cwd=project, env=config.child_env(), capture_output=True, timeout=180)
        text = (done.stdout + done.stderr).decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return problems + ["Loading the agent took more than three minutes and was stopped"], None
    line = next((l for l in reversed(text.splitlines()) if l.startswith("[driver-check] ")), None)
    if line is None:
        return problems + [f"The project's interpreter could not run ({entry['interpreter']}): "
                           + (text.strip().splitlines() or ["no output"])[-1][:200]], None
    report = json.loads(line.removeprefix("[driver-check] "))

    if report["missing_module"]:
        where = ("The project's environment has no module named" if own else
                 "No environment. The project has no .venv, and fleetopt's own interpreter has no module named")
        problems.append(f"{where} `{report['missing_module']}`. {install_hint(project)}")
    elif report["error"] and report["error_type"] not in ("AttributeError", "ImportError"):
        problems.append(f"The agent failed while loading: {report['error']}")
    if report["providers"] and not report["keys_present"] and not report["needs_no_key"]:
        names = " or ".join(dict.fromkeys(report["keys_accepted"]))
        where = f"in {env_file}" if env_file else "in the environment (the project names no env file)"
        problems.append(f"No key for a model provider. The agent uses {', '.join(report['providers'])}; set {names} {where}")
    return problems, report


def refused(tail):
    """A failed trial that no change to the entry can fix, as a line for the team."""
    lines = [l.strip() for l in tail.splitlines() if l.strip()]
    for pattern, what in ((REFUSED, "The model provider refused the call"),
                          (UNREACHABLE, "A service the agent depends on could not be reached")):
        hit = next((l for l in reversed(lines) if pattern.search(l)), None)
        if hit:
            return f"{what}: {hit[:220]}"
    return None


def save(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=1), encoding="utf-8")


def _entry(plan, project, python, inputs, source):
    return {"adapter": "langgraph", "project": str(project), "name": plan.get("agent") or "agent",
            "graph": plan["graph"], "paths": plan["paths"], "interpreter": python, "env_file": plan.get("env_file"),
            "env": plan["env"], "config": plan.get("config") or {}, "context": plan.get("context") or {},
            "store": plan.get("store"), "input_template": plan.get("input_template"),
            "inputs": inputs or _spread(plan["inputs"]),
            "inputs_source": source or plan.get("inputs_from") or "fleetopt, as the agent's users would write them",
            "proven": None}


def _mine(problems, report):
    """Problems a better answer could fix, as opposed to what the team must set up: the
    graph did not load for a reason in how it was named, or the input did not fit."""
    fixable = [p for p in problems if p.startswith("The agent failed while loading")]
    if report and not report.get("loaded") and not report.get("missing_module") and not fixable:
        fixable.append(f"The graph could not be loaded as named: {report.get('error')}")
    if report and report.get("loaded") and report.get("input") not in (None, "ok"):
        fixable.append(f"The input does not fit the graph: {report['input']}")
    return fixable


def ensure(project, out, wanted=None, say=print, supplied=None):
    """The proven entry for this project, settled now if it was not already.
    Returns (path to the entry, the entry). Raises Unstartable, or NotReady for what the
    team must set up. `supplied` is a file or folder of eval cases: its inputs are then
    the ones the agent is run on."""
    from fleetopt.drive import setup  # the step that needs a model; imported late so tests can replace it

    project = pathlib.Path(project).resolve()
    out = pathlib.Path(out).resolve()  # the driver runs from inside the project: a relative path would point nowhere
    path = path_for(out, project, wanted or "agent")
    inputs, source = team_inputs(project, supplied)

    if path.exists():
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("proven") and pathlib.Path(entry["interpreter"]).exists():
            asked = list(entry["inputs"])
            if supplied:  # how it is started does not change; what it is asked does
                entry["inputs"], entry["inputs_source"] = inputs, source
                save(path, entry)
            say(f"[fleetopt] agent: {entry['name']} ({entry['graph']}), {len(entry['inputs'])} inputs from "
                f"{entry['inputs_source']}")
            if entry["inputs"] != asked:
                # a review of other requests is not a review of these. Not saved: true of this call only
                entry = {**entry, "asked_anew": True}
            if entry.get("broken"):
                say(f"[fleetopt] when it was last started no request finished. Its own failure: {entry['broken']}")
            return path, entry

    python, where = interpreter(project)
    keys = sorted(k for k in os.environ if KEYISH.match(k))
    details = path.with_suffix(".log")
    details.parent.mkdir(parents=True, exist_ok=True)
    say("[fleetopt] reading the project to work out how to run its agent (a few minutes, once per project)")
    plan, failure, tail, entry, unloaded = None, None, "", None, []
    for attempt in range(1, ATTEMPTS + 1):
        try:
            plan = setup.settle(project, keys=keys, have_inputs=bool(inputs), wanted=wanted, earlier=plan,
                                failure=failure)
        except ValueError as exc:
            failure = f"Your answer could not be used: {exc}"
            say(f"[fleetopt] {failure}")
            continue
        if plan["missing"]:
            raise NotReady(plan["missing"])
        if plan.get("agent_fault") and starts_but_fails(tail):
            break  # the session agrees the failure is the agent's own
        entry = _entry(plan, project, python, inputs, source)
        if not entry["inputs"]:
            failure = "Your answer gave no inputs, and fleetopt has none."
            continue
        say(f"[fleetopt] agent: {entry['name']} ({entry['graph']})" + (f", because {plan['why']}" if plan.get("why") else ""))
        if plan.get("others"):
            say(f"[fleetopt] the others: {', '.join(map(str, plan['others']))[:200]} (--graph picks another)")
        say(f"[fleetopt] runs on {where}")
        say(f"[fleetopt] {len(entry['inputs'])} test inputs, from {entry['inputs_source']}:")
        for text in entry["inputs"]:
            say("    - " + " ".join(text.split())[:90] + ("..." if len(text) > 90 else ""))
        save(path, entry)

        problems, report = preflight(path, entry)
        fixable = _mine(problems, report)
        team = [p for p in problems if p not in fixable]
        if team:
            raise NotReady(team)
        keys = sorted(set(keys) | set((report or {}).get("keys_present") or []))
        unloaded = fixable
        if fixable:
            failure = "\n".join(fixable)
            say(f"[fleetopt] it did not load: {fixable[0][:200]}")
            continue

        say("[fleetopt] trying the agent on one input")
        ok, tail = prove(path, entry)
        with details.open("a", encoding="utf-8") as log:
            log.write(f"--- trial {attempt}, {datetime.datetime.now():%H:%M:%S}, "
                      f"{'answered' if ok else 'did not answer'}\n{tail}\n")
        if ok:
            entry["proven"] = datetime.datetime.now().isoformat(timespec="seconds")
            save(path, entry)
            say("[fleetopt] it answered. How to start it is saved, and not worked out again next time")
            return path, entry
        say(f"[fleetopt] it did not answer: {failure_line(tail)}")
        failure = tail

    broken = starts_but_fails(tail)
    if entry and broken:
        # It starts and the model answers; the request fails on the agent's own code.
        # The agent that does not finish is the one that most needs a review.
        entry.update(proven=datetime.datetime.now().isoformat(timespec="seconds"), broken=broken)
        save(path, entry)
        say(f"[fleetopt] it starts, and no request finishes. Its own failure: {broken}")
        say("[fleetopt] carrying on: this is reviewed as a broken agent")
        return path, entry
    if unloaded:  # two answers, and it still does not load: the cause is in the project
        raise NotReady(unloaded)
    if refused(tail):
        raise NotReady([refused(tail)])
    raise Unstartable(f"fleetopt could not start the agent after {ATTEMPTS} tries. The last thing it printed:\n"
                      f"{tail or failure}\nEvery attempt is in {details}")
