"""Capture hook: registers a global LangChain tracer and snapshots LangGraph graphs.

Two independent pieces, both installed at interpreter startup:

1. A `BaseTracer` subclass registered through `register_configure_hook`, which
   LangChain *appends* to every callback manager. It sits alongside any
   LangChainTracer the project already configured - their LangSmith traces keep
   flowing untouched, we just also write JSONL locally.

2. A patch on `StateGraph.compile` so every compiled graph snapshots its own
   topology. No need to locate the graph object in the target's source.
"""

import json
import os
import sys
import threading

_LOCK = threading.Lock()
_installed = False


def _warn(msg):
    print(f"[fleetopt] {msg}", file=sys.stderr)


def _append(path_env, record):
    path = os.environ.get(path_env)
    if not path:
        return
    line = json.dumps(record, default=str)
    with _LOCK:
        with open(path, "a") as fh:
            fh.write(line + "\n")


def _find_usage(obj, depth=0):
    """Dig token usage out of a serialized LLMResult.

    The exact nesting moves between langchain versions, so search for the key
    rather than hardcoding a path.
    """
    if depth > 6:
        return None
    if isinstance(obj, dict):
        for key in ("usage_metadata", "token_usage", "usage"):
            value = obj.get(key)
            if isinstance(value, dict) and value:
                return value
        for value in obj.values():
            found = _find_usage(value, depth + 1)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_usage(value, depth + 1)
            if found:
                return found
    return None


def _text_of(obj, depth=0):
    """Best-effort text out of a prompt or generation structure."""
    if depth > 5:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("text", "content", "prompts", "messages", "generations"):
            if key in obj:
                found = _text_of(obj[key], depth + 1)
                if found:
                    return found
        return None
    if isinstance(obj, list):
        parts = [t for t in (_text_of(v, depth + 1) for v in obj) if t]
        return "\n---\n".join(parts) if parts else None
    return None


def _clip(text):
    """Cap stored payloads. The full length is kept separately - size is the
    signal that detects a growing re-sent context, the bytes are just evidence."""
    if text is None:
        return None
    limit = int(os.environ.get("FLEETOPT_MAX_CHARS", "4000"))
    return text if len(text) <= limit else text[:limit] + f"...[+{len(text) - limit} chars]"


def _to_record(run):
    metadata = (run.extra or {}).get("metadata") or {}
    duration_ms = None
    if run.end_time and run.start_time:
        duration_ms = round((run.end_time - run.start_time).total_seconds() * 1000, 2)

    record = {
        "run_id": str(run.id),
        "parent_run_id": str(run.parent_run_id) if run.parent_run_id else None,
        "trace_id": str(run.trace_id) if run.trace_id else None,
        "name": run.name,
        "run_type": run.run_type,
        "node": metadata.get("langgraph_node"),
        "step": metadata.get("langgraph_step"),
        "model": metadata.get("ls_model_name"),
        "provider": metadata.get("ls_provider"),
        "start_time": run.start_time,
        "end_time": run.end_time,
        "duration_ms": duration_ms,
        "error": run.error,
        "tags": run.tags or [],
    }

    if run.run_type == "llm":
        record["usage"] = _find_usage(run.outputs)
        prompt = _text_of(run.inputs)
        # Always record the size: it costs 8 bytes and it's what exposes a node
        # re-sending a context that grows every iteration.
        record["prompt_chars"] = len(prompt) if prompt else None
        if os.environ.get("FLEETOPT_CAPTURE_IO"):
            record["prompt"] = _clip(prompt)
            record["completion"] = _clip(_text_of(run.outputs))

    # Root-run inputs/outputs are the baseline for comparing an optimized run
    # against the original, but they carry real payloads - opt in explicitly.
    if run.parent_run_id is None and os.environ.get("FLEETOPT_CAPTURE_IO"):
        record["inputs"] = run.inputs
        record["outputs"] = run.outputs

    return record


def _register_tracer():
    from contextvars import ContextVar

    from langchain_core.tracers.base import BaseTracer
    from langchain_core.tracers.context import register_configure_hook

    class FleetOptTracer(BaseTracer):
        name = "fleetopt_tracer"

        def _persist_run(self, run):
            """Required by BaseTracer; we stream per-run instead."""

        def _on_run_update(self, run):
            # Called as each run finishes, root or not - so a crash mid-graph
            # still leaves everything completed so far on disk.
            try:
                _append("FLEETOPT_TRACE_FILE", _to_record(run))
            except Exception as exc:
                _warn(f"dropped a run: {exc}")

    # LangChain instantiates the class with no args whenever the env var is set,
    # and skips it if an instance of the same class is already attached.
    register_configure_hook(
        ContextVar("fleetopt_tracer", default=None),
        True,
        FleetOptTracer,
        "FLEETOPT_CAPTURE",
    )


def _snapshot_graph(compiled):
    graph = compiled.get_graph(xray=True)
    try:
        mermaid = graph.draw_mermaid()
    except Exception:
        mermaid = None
    _append(
        "FLEETOPT_GRAPH_FILE",
        {
            "name": getattr(compiled, "name", None),
            "nodes": sorted(graph.nodes),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "conditional": getattr(edge, "conditional", False),
                    "label": getattr(edge, "data", None),
                }
                for edge in graph.edges
            ],
            "mermaid": mermaid,
        },
    )


def _patch_langgraph():
    from langgraph.graph.state import StateGraph

    original = StateGraph.compile
    if getattr(original, "_fleetopt", False):
        return

    def compile(self, *args, **kwargs):  # noqa: A001 - matching the patched name
        compiled = original(self, *args, **kwargs)
        try:
            _snapshot_graph(compiled)
        except Exception as exc:
            _warn(f"graph snapshot failed: {exc}")
        return compiled

    compile._fleetopt = True
    StateGraph.compile = compile


def install():
    """Import langchain/langgraph eagerly and instrument both.

    Eager import rather than a lazy post-import hook: we already know this is a
    LangGraph project, and it guarantees the compile patch lands before any
    module-level `builder.compile()` in the target runs.
    """
    global _installed
    if _installed:
        return
    _installed = True

    for label, fn in (("tracer", _register_tracer), ("langgraph", _patch_langgraph)):
        try:
            fn()
        except ImportError as exc:
            _warn(f"{label} not instrumented ({exc})")
        except Exception as exc:
            _warn(f"{label} instrumentation failed: {exc}")
