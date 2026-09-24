"""Eval cases: where a team's expected answers come from.

A case is {"input": str, "expected": str, "source": str}. Loaded from JSONL or JSON
with flexible keys, or from deepeval-style Python test files (LLMTestCase(...) /
Golden(...) with literal input and expected_output). We never invent expected
answers and never run the team's eval framework: it bills their own graders.

Cases are matched to captured runs by their input text appearing in the run's root
inputs, so only cases the run command actually exercised get graded. The rest are
reported as unmatched, never silently dropped.
"""

import ast
import json
import pathlib
import re

INPUT_KEYS = ("input", "query", "question", "prompt", "user_input", "actual_input", "messages")
EXPECTED_KEYS = ("expected", "expected_output", "reference", "answer", "ideal",
                 "ground_truth", "reference_output", "expected_answer")
SKIP_DIRS = {".venv", "venv", "node_modules", "site-packages", "__pycache__", ".git", ".fleetopt"}
CASE_CALLS = {"LLMTestCase", "Golden", "ConversationalTestCase"}


def load(path):
    """Returns (cases, notes). A directory is walked for .jsonl/.json/.py files."""
    root = pathlib.Path(path)
    if root.is_dir():
        files = sorted(
            f for f in root.rglob("*")
            if f.suffix in (".jsonl", ".json", ".py")
            and not (set(f.parts) & SKIP_DIRS)
        )
    else:
        files = [root]

    cases, notes = [], []
    for f in files:
        try:
            found = _load_file(f)
        except (OSError, ValueError, SyntaxError) as exc:
            notes.append(f"{f}: skipped ({exc})")
            continue
        if found:
            cases += found
            notes.append(f"{f}: {len(found)} cases")
    return cases, notes


def _load_file(f):
    src = str(f)
    if f.suffix == ".jsonl":
        objs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif f.suffix == ".json":
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((data[k] for k in ("cases", "tests", "data", "examples", "goldens") if isinstance(data.get(k), list)), [data])
        objs = data if isinstance(data, list) else []
    elif f.suffix == ".py":
        return _from_python(f)
    else:
        return []
    return [c for c in (_from_obj(o, src) for o in objs if isinstance(o, dict)) if c]


def _from_obj(obj, source):
    lower = {str(k).lower(): v for k, v in obj.items()}
    inp = next((lower[k] for k in INPUT_KEYS if k in lower), None)
    exp = next((lower[k] for k in EXPECTED_KEYS if k in lower), None)
    if inp is None or exp is None:
        return None
    return {"input": _text(inp), "expected": _text(exp), "source": source}


def _text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value)
    if isinstance(value, dict) and "content" in value:
        return _text(value["content"])
    return json.dumps(value, ensure_ascii=False)


def _from_python(f):
    """deepeval-style: literal input=/expected_output= keywords on known calls."""
    tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
    cases = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in CASE_CALLS:
            continue
        kw = {k.arg: k.value for k in node.keywords}
        inp, exp = kw.get("input"), kw.get("expected_output")
        if isinstance(inp, ast.Constant) and isinstance(inp.value, str) \
                and isinstance(exp, ast.Constant) and isinstance(exp.value, str):
            cases.append({"input": inp.value, "expected": exp.value, "source": str(f)})
    return cases


def _norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def match(cases, run_inputs):
    """The first case whose input appears in a captured run's root inputs."""
    haystack = _norm(run_inputs)
    if not haystack:
        return None
    for case in cases:
        needle = _norm(case["input"])
        if needle and needle in haystack:
            return case
    return None
