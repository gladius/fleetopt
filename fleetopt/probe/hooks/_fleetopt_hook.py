"""Capture hook: records every model call, agent step and tool call of the agent run, and
snapshots its structure, in one format whatever framework it is built on.

LangGraph, installed at interpreter startup:

1. A `BaseTracer` subclass registered through `register_configure_hook`, which
   LangChain *appends* to every callback manager. It sits alongside any
   LangChainTracer the project already configured - their LangSmith traces keep
   flowing untouched, we just also write JSONL locally.

2. A patch on `StateGraph.compile` so every compiled graph snapshots its own
   topology. No need to locate the graph object in the target's source.

ADK (Google's Agent Development Kit): a plugin the driver gives the Runner it builds, with
the same records: an agent is a node, a workflow agent (sequential, parallel, loop) a nested
graph, a hand-off between agents a conditional edge. Nothing in the team's code is touched.
"""

import datetime
import json
import os
import sys
import threading
import uuid

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


def _path(metadata):
    """Where in the nested graphs a run happened, as the graph names its nodes: 'team:model'.
    The node's name alone is not enough: two sub-graphs may each have a 'model'."""
    ns = metadata.get("langgraph_checkpoint_ns")
    return ":".join(part.rsplit(":", 1)[0] for part in ns.split("|")) if ns else None


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
        "path": _path(metadata),
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


def _sources(compiled):
    """{node, as the graph names it: [file, first line, last line]} for each node that is a
    function, sub-graphs included. It ties a change in the code to the node it changes."""
    import inspect

    found = {}
    try:
        graphs = [("", compiled)] + [(ns.replace("|", ":") + ":", sub)
                                     for ns, sub in compiled.get_subgraphs(recurse=True)]
        for prefix, graph in graphs:
            for name, spec in getattr(getattr(graph, "builder", None), "nodes", {}).items():
                fn = getattr(spec.runnable, "func", None) or getattr(spec.runnable, "afunc", None)
                try:
                    lines, first = inspect.getsourcelines(fn)
                    found[prefix + name] = [inspect.getsourcefile(fn), first, first + len(lines) - 1]
                except (TypeError, OSError):  # a sub-graph, a class, a partial: no lines of its own
                    pass
    except Exception as exc:
        _warn(f"node sources not recorded: {exc}")
    return found


def is_adk(obj):
    """An ADK agent of any kind (it runs and has sub-agents; a LangGraph graph has neither), or an
    ADK App around one (what `adk run` starts first when a file has one)."""
    return (hasattr(obj, "run_async") and hasattr(obj, "sub_agents") or hasattr(obj, "root_agent")) and not hasattr(obj, "ainvoke")


def adk_root(obj):
    """The agent itself, or an App's root agent."""
    return obj.root_agent if is_adk(obj) and not hasattr(obj, "run_async") else obj


def _agent_sources(root, paths):
    """{node: [file, first line, last line]} of the assignment that made each agent, found by
    the agent's identity among the loaded modules' globals: `lookup = LlmAgent(...)` is the
    code of node `lookup`, as a node function's lines are a graph node's. An agent made in a
    loop or a factory has no assignment of its own, and no lines."""
    import ast
    import inspect

    found, agents = {}, {}

    def walk(agent):
        agents[id(agent)] = paths[agent.name]
        for sub in agent.sub_agents or []:
            walk(sub)

    walk(root)
    for module in list(sys.modules.values()):
        try:
            names = {k: agents[id(v)] for k, v in vars(module).items() if id(v) in agents}
            if not names:
                continue
            file = inspect.getsourcefile(module)
            tree = ast.parse(inspect.getsource(module))
        except Exception:  # noqa: BLE001 - a module without source, or one that will not be inspected
            continue
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for t in targets:
                if isinstance(t, ast.Name) and t.id in names and names[t.id] not in found:
                    found[names[t.id]] = [file, node.lineno, node.end_lineno]
    return found


def adk_tree(root):
    """The agent tree as a graph. {name: path}, {name: the agent whose run its own run belongs
    under, or None for the root's}, nodes, edges. A workflow agent (sequential, parallel, loop)
    is a nested graph: its sub-agents' paths start with its own. An LlmAgent's sub-agents are
    the agents it can hand the request to: they are its siblings, and each hand-off a
    conditional edge, so a hand-off never made is a branch never taken."""
    paths, under, nodes, edges = {}, {}, [], []

    def walk(agent, prefix, owner):
        path = prefix + agent.name
        paths[agent.name], under[agent.name] = path, owner
        nodes.append(path)
        subs = list(agent.sub_agents or [])
        if hasattr(agent, "tools") or not subs:  # an LlmAgent: hand-offs to siblings
            for sub in subs:
                edges.append({"source": path, "target": prefix + sub.name, "conditional": True})
                if not getattr(sub, "disallow_transfer_to_parent", True):
                    edges.append({"source": prefix + sub.name, "target": path, "conditional": True})
                walk(sub, prefix, owner)
            return
        kind = type(agent).__name__
        for i, sub in enumerate(subs):  # a workflow agent: a graph of its own
            if kind != "ParallelAgent" and i + 1 < len(subs):
                edges.append({"source": path + ":" + sub.name, "target": path + ":" + subs[i + 1].name, "conditional": False})
            walk(sub, path + ":", agent.name)
        if kind == "LoopAgent" and len(subs) > 1:
            edges.append({"source": path + ":" + subs[-1].name, "target": path + ":" + subs[0].name, "conditional": True})
        elif kind == "LoopAgent" and subs:
            edges.append({"source": path + ":" + subs[0].name, "target": path + ":" + subs[0].name, "conditional": True})

    walk(root, "", None)
    return paths, under, nodes, edges


def _parts_text(content):
    out = []
    for p in getattr(content, "parts", None) or []:
        if getattr(p, "text", None):
            out.append(p.text)
        elif getattr(p, "function_call", None):
            out.append(json.dumps({"call": p.function_call.name, "args": p.function_call.args}, default=str))
        elif getattr(p, "function_response", None):
            out.append(json.dumps({"result": p.function_response.response}, default=str))
    return "\n".join(out)


def adk_plugin(root):
    """The probe as an ADK plugin, for the Runner fleetopt's driver builds around `root`."""
    from google.adk.plugins.base_plugin import BasePlugin

    paths, under, _, _ = adk_tree(adk_root(root))
    now = lambda: datetime.datetime.now(datetime.timezone.utc)
    io = bool(os.environ.get("FLEETOPT_CAPTURE_IO"))

    class FleetOptPlugin(BasePlugin):
        def __init__(self):
            super().__init__(name="fleetopt_probe")
            self.inv = {}  # invocation id: {"step", "runs": {agent: run id}, "began": {agent: time}, "llm", "tool", "final"}

        def _inv(self, invocation_id):
            return self.inv.setdefault(invocation_id, {"step": 0, "runs": {}, "began": {}, "llm": {}, "tool": {}, "final": None})

        def _write(self, inv_id, record, began, error=None):
            ended = now()
            record.update({"trace_id": inv_id, "start_time": began, "end_time": ended, "error": error, "tags": [],
                           "duration_ms": round((ended - began).total_seconds() * 1000, 2) if began else None})
            _append("FLEETOPT_TRACE_FILE", record)

        def _parent(self, inv, agent_name):
            owner = under.get(agent_name)
            return inv["runs"].get(owner) if owner else inv["root"]

        async def before_run_callback(self, *, invocation_context):
            inv = self._inv(invocation_context.invocation_id)
            inv["root"], inv["root_began"] = invocation_context.invocation_id, now()
            inv["input"] = _parts_text(invocation_context.user_content)

        async def on_event_callback(self, *, invocation_context, event):
            if event.is_final_response() and event.content and not event.get_function_calls():
                self._inv(invocation_context.invocation_id)["final"] = _parts_text(event.content)

        def _close_run(self, invocation_context, error=None):
            inv_id = invocation_context.invocation_id
            inv = self.inv.get(inv_id)
            if not inv:
                return
            for agent_name in list(inv["began"]):  # an agent that handed the request on gets no after-agent call
                self._agent_record(agent_name, invocation_context)
            self.inv.pop(inv_id)
            record = {"run_id": inv["root"], "parent_run_id": None, "name": invocation_context.agent.name
                      if invocation_context.agent else "adk", "run_type": "chain", "node": None, "path": None,
                      "step": None, "model": None, "provider": None}
            if io:
                record["inputs"] = inv.get("input")
                record["outputs"] = inv["final"] if error is None else None
            self._write(invocation_context.invocation_id, record, inv.get("root_began"), error)

        async def after_run_callback(self, *, invocation_context):
            self._close_run(invocation_context)

        async def on_run_error_callback(self, *, invocation_context, error):
            self._close_run(invocation_context, repr(error))

        async def before_agent_callback(self, *, agent, callback_context):
            inv = self._inv(callback_context.invocation_id)
            inv["step"] += 1
            inv["runs"][agent.name] = str(uuid.uuid4())
            inv["began"][agent.name] = (now(), inv["step"])

        def _agent_record(self, agent_name, callback_context, error=None):
            inv = self._inv(callback_context.invocation_id)
            began, step = inv["began"].pop(agent_name, (None, None))
            self._write(callback_context.invocation_id, {
                "run_id": inv["runs"].get(agent_name), "parent_run_id": self._parent(inv, agent_name), "name": agent_name,
                "run_type": "chain", "node": agent_name, "path": paths.get(agent_name, agent_name), "step": step,
                "model": None, "provider": None}, began, error)

        async def after_agent_callback(self, *, agent, callback_context):
            self._agent_record(agent.name, callback_context)

        async def on_agent_error_callback(self, *, agent, callback_context, error):
            self._agent_record(agent.name, callback_context, repr(error))

        async def before_model_callback(self, *, callback_context, llm_request):
            inv = self._inv(callback_context.invocation_id)
            system = llm_request.config.system_instruction if llm_request.config else None
            prompt = "\n---\n".join(filter(None, [system if isinstance(system, str) else _parts_text(system)]
                                          + [_parts_text(c) for c in llm_request.contents or []]))
            inv["llm"][callback_context.agent_name] = (now(), prompt, llm_request.model or None)

        def _model_record(self, callback_context, response=None, model=None, error=None):
            inv = self._inv(callback_context.invocation_id)
            agent = callback_context.agent_name
            began, prompt, asked = inv["llm"].pop(agent, (None, None, None))
            usage = getattr(response, "usage_metadata", None)
            record = {"run_id": str(uuid.uuid4()), "parent_run_id": inv["runs"].get(agent), "name": "model",
                      "run_type": "llm", "node": agent, "path": paths.get(agent, agent),
                      "step": inv["began"].get(agent, (None, None))[1], "model": asked or model, "provider": None,
                      "prompt_chars": len(prompt) if prompt else None,
                      "usage": {"input_tokens": usage.prompt_token_count or 0, "output_tokens": usage.candidates_token_count or 0,
                                "input_token_details": {"cache_read": usage.cached_content_token_count or 0}} if usage else None}
            if io:
                record["prompt"], record["completion"] = _clip(prompt), _clip(_parts_text(response.content) if response else None)
            self._write(callback_context.invocation_id, record, began, error)

        async def after_model_callback(self, *, callback_context, llm_response):
            if llm_response.partial:
                return None
            self._model_record(callback_context, llm_response,
                               error=f"{llm_response.error_code}: {llm_response.error_message}" if llm_response.error_code else None)

        async def on_model_error_callback(self, *, callback_context, llm_request, error):
            self._model_record(callback_context, model=llm_request.model or None, error=repr(error))

        async def before_tool_callback(self, *, tool, tool_args, tool_context):
            inv = self._inv(tool_context.invocation_id)
            inv["step"] += 1
            inv["tool"][(tool_context.agent_name, tool.name)] = (now(), inv["step"])

        def _tool_record(self, tool, tool_args, tool_context, result=None, error=None):
            inv = self._inv(tool_context.invocation_id)
            agent = tool_context.agent_name
            began, step = inv["tool"].pop((agent, tool.name), (None, None))
            record = {"run_id": str(uuid.uuid4()), "parent_run_id": inv["runs"].get(agent), "name": tool.name,
                      "run_type": "tool", "node": agent, "path": paths.get(agent, agent), "step": step, "model": None,
                      "provider": None}
            if io:
                record["inputs"], record["outputs"] = tool_args, result
            self._write(tool_context.invocation_id, record, began, error)

        async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
            self._tool_record(tool, tool_args, tool_context, result)

        async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
            self._tool_record(tool, tool_args, tool_context, error=repr(error))

    return FleetOptPlugin()


def snapshot_agent(root, driven=False):
    """An ADK agent tree's structure, as snapshot_graph records a graph's."""
    root = adk_root(root)
    paths, _, nodes, edges = adk_tree(root)
    try:
        sources = _agent_sources(root, paths)
    except Exception as exc:  # noqa: BLE001
        _warn(f"agent sources not recorded: {exc}")
        sources = {}
    _append("FLEETOPT_GRAPH_FILE", {"driven": driven, "sources": sources, "name": root.name, "nodes": sorted(nodes),
                                    "edges": edges, "mermaid": None})


def snapshot_graph(compiled, driven=False):
    """One compiled graph's structure. `driven`: fleetopt's driver marks the graph it runs, of
    the several a project may compile."""
    if is_adk(compiled):
        return snapshot_agent(compiled, driven)
    graph = compiled.get_graph(xray=True)
    try:
        mermaid = graph.draw_mermaid()
    except Exception:
        mermaid = None
    _append(
        "FLEETOPT_GRAPH_FILE",
        {
            "driven": driven,
            "sources": _sources(compiled),
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
            snapshot_graph(compiled)
        except Exception as exc:
            _warn(f"graph snapshot failed: {exc}")
        return compiled

    compile._fleetopt = True
    StateGraph.compile = compile


def _watch_files():
    """Which of the project's files this process loaded: imported modules and files it opened
    (prompts in .txt or .yaml, data). Written once, at exit, to the graph file. A change to a
    file the measured run never loaded is proven by nothing, whatever graph it belongs to."""
    import atexit

    root = os.environ.get("FLEETOPT_PROJECT")
    if not root:
        return
    root = os.path.abspath(root) + os.sep
    seen, skip = set(), (".git", ".fleetopt", "__pycache__", ".venv", "venv", "node_modules")

    def keep(path):
        path = os.path.abspath(path)
        if path.startswith(root):
            rel = path[len(root):].replace(os.sep, "/")
            if not rel.split("/")[0] in skip and not rel.endswith((".pyc", ".jsonl")):
                seen.add(rel)

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], str):
            keep(args[0])

    def write():
        for module in list(sys.modules.values()):
            file = getattr(module, "__file__", None)
            if isinstance(file, str):
                keep(file)
        _append("FLEETOPT_GRAPH_FILE", {"loaded": sorted(seen)})

    sys.addaudithook(audit)
    atexit.register(write)


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

    import importlib.util

    for label, fn in (("tracer", _register_tracer), ("langgraph", _patch_langgraph), ("files", _watch_files)):
        try:
            fn()
        except ImportError as exc:
            if importlib.util.find_spec("google.adk") is None:  # an ADK project has no langchain: the driver's plugin records it
                _warn(f"{label} not instrumented ({exc})")
        except Exception as exc:
            _warn(f"{label} instrumentation failed: {exc}")
