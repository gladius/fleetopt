"""fleetopt's driver: starts an agent the way its entry says and feeds it inputs.

It runs INSIDE the target's own interpreter as a plain script, so it imports nothing
from fleetopt and nothing the project has not installed. This is the only thing
fleetopt ever executes in a target: a named graph, called with given inputs. Never a
command somebody guessed.

    <the project's python> driver.py <entry.json> [--limit N]
    <the project's python> driver.py <entry.json> --check

--check loads the agent and calls nothing: it reports whether the agent loaded, which
module was missing if it did not, which model providers it pulls in and whether a key
for any of them is set (names only, never a value).

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

# Which environment variable lets each provider package through. An empty tuple: none needed.
PROVIDER_KEYS = {
    "langchain_anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "langchain_openai": ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
    "langchain_google_genai": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "langchain_google_vertexai": ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"),
    "langchain_aws": ("AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK"),
    "langchain_groq": ("GROQ_API_KEY",),
    "langchain_mistralai": ("MISTRAL_API_KEY",),
    "langchain_cohere": ("COHERE_API_KEY",),
    "langchain_deepseek": ("DEEPSEEK_API_KEY",),
    "langchain_xai": ("XAI_API_KEY",),
    "langchain_ollama": (),
}


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


def user_message(text):
    """A user message as the project's own framework makes one. Observed: a graph whose
    `messages` field has no reducer passes the input through untouched, and its nodes
    read `.content`; a dict ends that run on the first node. The dict is for a project
    without langchain, where nothing else is possible."""
    try:
        from langchain_core.messages import HumanMessage
    except ImportError:
        return {"role": "user", "content": text}
    return HumanMessage(content=text)


def build_input(graph, text, template=None):
    """The graph's input for one piece of text. A `messages` field gets a user message;
    otherwise the first text field gets the text and the other required fields start
    empty. Anything more particular belongs in the entry's input_template."""
    if template is not None:
        return fill(template, text)
    schema = graph.get_input_jsonschema()
    fields = schema.get("properties", {})
    if "messages" in fields:
        return {"messages": [user_message(text)]}
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


def check(entry, root, paths):
    """Load the agent, call nothing, say what was found. One line of JSON for fleetopt."""
    report = {"loaded": False, "error": None, "error_type": None, "missing_module": None, "input": None}
    try:
        graph = resolve(entry["graph"], root, paths)
    except ModuleNotFoundError as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", error_type="ModuleNotFoundError", missing_module=exc.name)
    except Exception as exc:  # noqa: BLE001
        report.update(error=f"{type(exc).__name__}: {str(exc)[:300]}", error_type=type(exc).__name__)
    else:
        report["loaded"] = True
        try:
            build_input(graph, "check", entry.get("input_template"))
            report["input"] = "ok"
        except Exception as exc:  # noqa: BLE001
            report["input"] = str(exc)[:300]
    providers = sorted(m for m in PROVIDER_KEYS if m in sys.modules)
    report["providers"] = providers
    report["keys_present"] = sorted(k for m in providers for k in PROVIDER_KEYS[m] if os.environ.get(k))
    report["keys_accepted"] = sorted(k for m in providers for k in PROVIDER_KEYS[m])
    report["needs_no_key"] = any(not PROVIDER_KEYS[m] for m in providers)
    print("[driver-check] " + json.dumps(report))
    return 0 if report["loaded"] else 2


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
    if "--check" in argv:
        return check(entry, root, paths)

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
