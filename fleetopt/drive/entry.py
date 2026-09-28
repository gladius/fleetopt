"""How to start a project's agent: found once, proven once, remembered.

Starting an agent is a fact about the project, not something a user should have to
type. fleetopt reads what the project declares (langgraph.json), or finds the
compiled graph in its source, picks the project's own interpreter and env file,
takes inputs from the team's eval cases, and proves the result with one input. What
it settles on is an entry: a small JSON file in fleetopt's own folder, never in the
team's repo. The next run reads it and asks nothing.

fleetopt then runs its own driver against that entry (see driver.py) and nothing else.
Another framework is another way of filling in the same entry.
"""

import ast
import datetime
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys

from fleetopt import config
from fleetopt.evidence import evals

DRIVER = pathlib.Path(__file__).with_name("driver.py")
SKIP_DIRS = evals.SKIP_DIRS | {"tests", "test", "docs", "notebooks", "build", "dist"}
CONSTRUCTORS = {"create_agent", "create_react_agent", "create_supervisor", "create_swarm"}
FACTORY = re.compile(r"^_?(build|create|make|get|compile)\w*(graph|agent|workflow)\w*$")
LIKELY_FILES = ("graph.py", "agent.py", "main.py", "app.py", "workflow.py")
MAX_INPUTS = 4


class Unstartable(RuntimeError):
    """fleetopt could not work out how to start this agent. The message says what it
    found, what it tried and what failed."""


def declared(project):
    """What langgraph.json says: graphs by name, and the env file."""
    try:
        data = json.loads((project / "langgraph.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, None
    graphs = {}
    for name, spec in (data.get("graphs") or {}).items():
        spec = spec.get("path") if isinstance(spec, dict) else spec
        if isinstance(spec, str) and ":" in spec:
            graphs[name] = spec.removeprefix("./")
    env = data.get("env")
    return graphs, env.removeprefix("./") if isinstance(env, str) else None


def _calls(node):
    """Names called along a chain like builder.compile().with_config(...). A regular
    expression being compiled is not a graph."""
    while isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            if not (isinstance(func.value, ast.Name) and func.value.id in ("re", "regex")):
                yield func.attr
            node = func.value
        else:
            if isinstance(func, ast.Name):
                yield func.id
            return


def scanned(project, limit=600):
    """Compiled graphs and zero-argument graph factories found in the source, best first."""
    found = []
    files = sorted(f for f in project.rglob("*.py") if not set(f.relative_to(project).parts[:-1]) & SKIP_DIRS
                   and not any(p.startswith(".") for p in f.relative_to(project).parts[:-1]))[:limit]
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            if "langgraph" not in text and "langchain" not in text:
                continue
            tree = ast.parse(text, filename=str(f))
        except (OSError, SyntaxError, ValueError):
            continue
        rel = f.relative_to(project).as_posix()
        graphs, factories = [], []
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    names = set(_calls(node.value))
                    if "compile" in names or names & CONSTRUCTORS:
                        graphs.append(targets[0].id)
            elif isinstance(node, ast.FunctionDef) and FACTORY.match(node.name):
                required = len(node.args.args) - len(node.args.defaults) + sum(d is None for d in node.args.kw_defaults)
                if required == 0:
                    factories.append(f"{node.name}()")
        for name in graphs or factories:
            rank = (not graphs, f.name not in LIKELY_FILES, len(f.relative_to(project).parts), rel)
            found.append((rank, f"{rel}:{name}"))
    return [spec for _, spec in sorted(found)]


def candidates(project):
    """[(name, spec)]: what the project declares first, then what the source shows."""
    graphs, _ = declared(project)
    out = list(graphs.items())
    known = {spec for _, spec in out}
    out += [(spec, spec) for spec in scanned(project) if spec not in known]
    return out


def choose(found, wanted=None):
    if not found:
        raise Unstartable("no graph found: the project has no langgraph.json, and no compiled graph or "
                          "zero-argument graph factory at module level in its source")
    if wanted is None:
        return found[0]
    for name, spec in found:
        if wanted in (name, spec) or spec.endswith(wanted):
            return name, spec
    raise Unstartable(f"no graph called {wanted!r} here. Found: " + ", ".join(name for name, _ in found))


def interpreter(project):
    """The project's own environment; the one fleetopt runs in only when it has none."""
    for rel in (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe", "venv/Scripts/python.exe"):
        if (project / rel).exists():
            return str(project / rel), "the project's own environment"
    return sys.executable, "fleetopt's interpreter, because the project has no .venv of its own"


def inputs_for(project):
    """(inputs, where they came from). The team's eval cases first; then any file of
    inputs the project keeps; nothing if it has neither."""
    cases, _ = evals.load(project)
    if cases:
        by_source = {}
        for case in cases:
            by_source.setdefault(case["source"], []).append(case["input"])
        source, texts = max(by_source.items(), key=lambda kv: len(kv[1]))
        return _spread(texts), f"the team's eval cases, {pathlib.Path(source).relative_to(project)}"
    for f in sorted(project.rglob("*")):
        rel = f.relative_to(project)
        if f.suffix in (".jsonl", ".json") and re.search(r"input|question|prompt|quer", f.stem, re.I) \
                and not set(rel.parts[:-1]) & SKIP_DIRS and len(rel.parts) <= 4:
            texts = _texts(f)
            if texts:
                return _spread(texts), f"the project's {rel}"
    return [], None


def _texts(f):
    try:
        raw = f.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()] if f.suffix == ".jsonl" else json.loads(raw)
    except (OSError, ValueError):
        return []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, str):
            out.append(row)
        elif isinstance(row, dict):
            lower = {str(k).lower(): v for k, v in row.items()}
            value = next((lower[k] for k in evals.INPUT_KEYS if k in lower), None)
            if isinstance(value, str):
                out.append(value)
    return out


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


def prove(entry_path, entry, timeout=600):
    """One input, for real. Returns (worked, what the agent printed last)."""
    try:
        done = subprocess.run(command(entry_path, entry, limit=1), shell=True, cwd=entry["project"],
                              env=config.child_env(), capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"no answer within {timeout} seconds"
    text = (done.stdout + done.stderr).decode("utf-8", errors="replace")
    return done.returncode == 0, "\n".join(text.splitlines()[-30:])


def save(path, entry):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=1), encoding="utf-8")


def ensure(project, out, wanted=None, say=print):
    """The proven entry for this project, settled now if it was not already.
    Returns (path to the entry, the entry). Raises Unstartable."""
    from fleetopt.drive import setup  # the two steps that need a model; imported late so tests can replace them

    project = pathlib.Path(project).resolve()
    found = candidates(project)
    name, spec = choose(found, wanted)
    path = path_for(out, project, name)
    others = [n for n, _ in found if n != name]

    if path.exists():
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("proven") and pathlib.Path(entry["interpreter"]).exists():
            say(f"[fleetopt] agent: {name} ({entry['graph']}), {len(entry['inputs'])} inputs from {entry['inputs_source']}")
            return path, entry

    python, why = interpreter(project)
    _, env_file = declared(project)
    inputs, source = inputs_for(project)
    if not inputs:
        inputs, source = setup.propose_inputs(project, spec), "fleetopt, written from the README and the graph's source"
    entry = {
        "adapter": "langgraph", "project": str(project), "name": name, "graph": spec,
        "paths": [".", "src"] if (project / "src").is_dir() else ["."],
        "interpreter": python, "env_file": env_file or (".env" if (project / ".env").exists() else None),
        "env": {}, "config": {}, "input_template": None,
        "inputs": inputs, "inputs_source": source, "proven": None,
    }
    say(f"[fleetopt] agent: {name} ({spec})" + (f"; also here: {', '.join(others[:8])} (pick one with --graph)" if others else ""))
    say(f"[fleetopt] runs on {why}; {len(inputs)} inputs from {source}")

    tail = ""
    for attempt in range(3):
        save(path, entry)
        ok, tail = prove(path, entry)
        if ok:
            entry["proven"] = datetime.datetime.now().isoformat(timespec="seconds")
            save(path, entry)
            say("[fleetopt] started it once with one input: it runs")
            return path, entry
        if attempt == 2:
            break
        say(f"[fleetopt] it did not start (attempt {attempt + 1}); working out why")
        fix = setup.repair(project, entry, tail)
        if not fix:
            break
        entry.update(fix)
    raise Unstartable(f"fleetopt could not start {name} ({entry['graph']}). The last thing it printed:\n{tail}\n"
                      f"What was tried is in {path}")
