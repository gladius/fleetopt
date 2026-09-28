"""fleetopt's driver: starts an agent the way its entry says and feeds it inputs.

It runs INSIDE the target's own interpreter as a plain script, so it imports nothing
from fleetopt and nothing the project has not installed. This is the only thing
fleetopt ever executes in a target: a named graph, called with given inputs. Never a
command somebody guessed.

    <the project's python> driver.py <entry.json> [--limit N]

Exit 0 when at least one input finished or paused for a human, 1 when none did,
2 when the graph could not be loaded at all.
"""

import asyncio
import importlib
import importlib.util
import json
import os
import pathlib
import sys
import traceback
import uuid

BLANK = {"string": "", "array": [], "integer": 0, "number": 0, "boolean": False, "object": {}}


def load_env(path):
    """KEY=VALUE lines into the environment, the way the project's own tooling would.
    What is already set wins."""
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def resolve(spec, root, paths):
    """'pkg/module.py:name', 'pkg.module:name', or either ending in '()' for a factory."""
    target, _, attr = spec.partition(":")
    call = attr.endswith("()")
    attr = attr.removesuffix("()")
    if target.endswith(".py"):
        file = (root / target).resolve()
        base = next((pathlib.Path(p) for p in paths if file.is_relative_to(p)), root)
        name = ".".join(file.relative_to(base).with_suffix("").parts)
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] != name.split(".")[0]:
                raise  # something the project needs is not installed: say so, do not mask it
            module_spec = importlib.util.spec_from_file_location(file.stem, file)
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    found = getattr(module, attr)
    return found() if call else found


def fill(template, text):
    if isinstance(template, str):
        return template.replace("{input}", text)
    if isinstance(template, list):
        return [fill(v, text) for v in template]
    if isinstance(template, dict):
        return {k: fill(v, text) for k, v in template.items()}
    return template


def build_input(graph, text, template=None):
    """The graph's input for one piece of text. A `messages` field gets a user message;
    otherwise the first text field gets the text and the other required fields start
    empty. Anything more particular belongs in the entry's input_template."""
    if template is not None:
        return fill(template, text)
    schema = graph.get_input_jsonschema()
    fields = schema.get("properties", {})
    if "messages" in fields:
        return {"messages": [{"role": "user", "content": text}]}
    payload, placed = {}, False
    for name, spec in fields.items():
        kind = spec.get("type")
        if kind == "string" and not placed:
            payload[name], placed = text, True
        elif "default" not in spec and kind in BLANK:
            payload[name] = type(BLANK[kind])(BLANK[kind])
    if not placed:
        raise ValueError(f"the graph's input has no text field to put the input in (fields: {', '.join(fields) or 'none'})")
    return payload


def main(argv):
    entry = json.loads(pathlib.Path(argv[0]).read_text(encoding="utf-8"))
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    root = pathlib.Path(entry["project"])
    os.chdir(root)
    paths = [str((root / p).resolve()) for p in entry.get("paths", ["."])]
    sys.path[:0] = [p for p in paths if p not in sys.path]
    if entry.get("env_file") and (root / entry["env_file"]).exists():
        load_env(root / entry["env_file"])
    os.environ.update(entry.get("env") or {})

    try:
        graph = resolve(entry["graph"], root, paths)
    except Exception as exc:  # noqa: BLE001 - whatever it is, the caller needs to read it
        traceback.print_exc()
        print(f"[driver] could not load {entry['graph']}: {type(exc).__name__}: {exc}")
        return 2

    inputs = entry["inputs"][:limit]
    finished = paused = 0
    for text in inputs:
        config = {"configurable": {"thread_id": str(uuid.uuid4()), **(entry.get("config") or {})}}
        try:
            result = asyncio.run(graph.ainvoke(build_input(graph, text, entry.get("input_template")), config=config))
        except Exception as exc:  # noqa: BLE001 - one bad input must not hide the others
            print(f"[driver] FAILED {text[:70]!r}\n         {type(exc).__name__}: {str(exc)[:400]}")
            continue
        if isinstance(result, dict) and result.get("__interrupt__"):
            paused += 1
            print(f"[driver] PAUSED for a human {text[:70]!r}")
        else:
            finished += 1
            print(f"[driver] OK {text[:70]!r}")
    print(f"[driver] {finished} finished, {paused} paused, {len(inputs) - finished - paused} failed, of {len(inputs)} inputs")
    return 0 if finished or paused else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
