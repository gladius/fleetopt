"""How to start a project's agent: worked out once, proven once, remembered.

A session reads the project and tries its own answer until the agent starts (start.py,
guided by START.md). What it settles on is a small JSON file in fleetopt's own folder,
never in the team's repo, and only an entry a trial proved is kept. The next run reads it.
"""

import datetime
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys

from fleetopt.evidence import evals

DRIVER = pathlib.Path(__file__).with_name("driver.py")
MAX_INPUTS = 4


class Unstartable(RuntimeError):
    """fleetopt could not start this agent. The message says what was found."""


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


def save(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=1), encoding="utf-8")


def ensure(project, out, wanted=None, say=print, supplied=None):
    """The proven entry for this project, settled now if it was not already.
    Returns (path to the entry, the entry). Raises Unstartable, or NotReady for what the
    team must set up. `supplied` is a file or folder of eval cases: its inputs are then
    the ones the agent is run on."""
    from fleetopt.drive import start  # the step that needs a model; imported late so tests can replace it

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
    say("[fleetopt] reading the project and trying to start its agent (a few minutes, once per project)")
    answer, proven = start.settle(project, path, python=python, inputs=inputs, source=source, wanted=wanted)
    details = path.with_suffix(".log")
    if proven:
        entry = {**proven, "name": answer.get("agent") or proven["name"],
                 "proven": datetime.datetime.now().isoformat(timespec="seconds")}
        save(path, entry)
        say(f"[fleetopt] agent: {entry['name']} ({entry['graph']})" + (f", because {answer['why']}" if answer.get("why") else ""))
        if answer.get("others"):
            say(f"[fleetopt] the others: {', '.join(map(str, answer['others']))[:200]} (--graph picks another)")
        say(f"[fleetopt] runs on {where if entry['interpreter'] == python else entry['interpreter']}")
        say(f"[fleetopt] {len(entry['inputs'])} test inputs, from {entry['inputs_source']}:")
        for text in entry["inputs"]:
            say("    - " + " ".join(text.split())[:90] + ("..." if len(text) > 90 else ""))
        if entry.get("broken"):
            say(f"[fleetopt] it starts, and no request finishes. Its own failure: {entry['broken']}")
            say("[fleetopt] carrying on: this is reviewed as a broken agent")
        return path, entry
    missing = [str(m) for m in answer.get("missing") or [] if str(m).strip()]
    if missing:
        raise NotReady(missing)
    explanation = answer.get("explanation") or "the session gave no reason"
    if answer.get("status") == "cannot_see":
        raise Unstartable(f"fleetopt can start this agent but cannot see its model calls, so there is no cost "
                          f"to review: {explanation}")
    raise Unstartable(f"fleetopt could not start the agent: {explanation}" +
                      (f"\nEvery trial is in {details}" if details.exists() else ""))
