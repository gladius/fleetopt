"""Working out how to start a project's agent: one session that reads the project and tries.

The session reads the project the way a developer joining the team would (START.md),
writes an entry for fleetopt's driver, and tries it with `try_start`, which starts the
agent on one input under the probe and says what happened. It reads the failure itself
and changes what the failure points to, until the agent starts or it knows what only
the team can provide. Code holds what keeps a developer's machine safe and the answer
honest: the project's own interpreter, no shell (so nothing installed), no secret read,
at most four trials on the team's key, and "started" only for an entry a trial proved.

This replaced a set of rules that each came from one agent (scan for compiled graphs,
guess the input shape, a provider-key table, switch providers, sort failures by pattern);
the next agent always broke a different one.
"""

import asyncio
import datetime
import json
import os
import pathlib
import re
import shutil

from claude_agent_sdk import create_sdk_mcp_server, tool

from fleetopt import config

SKILL = (pathlib.Path(__file__).parent / "START.md").read_text(encoding="utf-8")
TRIALS = 4  # runs of the agent on one input, on the team's key
SECRET = re.compile(r"key|token|secret|password|credential", re.I)
KEYISH = re.compile(r"^[A-Z][A-Z0-9_]*(?:API_KEY|AUTH_TOKEN)$")

# Set by settle() for the one session it runs.
RUN = {}


def key_names(project):
    """Names of the provider keys set in the environment or the project's .env. Names
    only: a value is never read into fleetopt."""
    names = {k for k in os.environ if KEYISH.match(k)}
    env = project / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            name = line.strip().removeprefix("export ").split("=", 1)[0].strip()
            if KEYISH.match(name) and "=" in line and line.split("=", 1)[1].strip():
                names.add(name)
    return sorted(names)


def trial(path, entry, timeout=600):
    """One input, for real, under the probe: what the agent did, in numbers, and the last
    lines it printed. Leaves nothing behind."""
    from fleetopt.drive import entry as entry_mod
    from fleetopt.probe import runner

    raw, traces, _, code = runner.execute(entry["project"], entry_mod.command(path, entry, limit=1),
                                          pathlib.Path(path).parent.parent, timeout=timeout)
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
    return {"exit": code, "timed_out": code == runner.TIMED_OUT, "requests": len(roots),
            "finished": len(roots) - len(failed), "model_calls": sum(r.get("run_type") == "llm" for r in runs),
            "answered": sum(r.get("run_type") == "llm" and not r.get("error") for r in runs),
            "error": (failed[0].splitlines() or [""])[0][:300] if failed else None, "tail": tail}


def _checked(plan):
    """The entry the session proposed, with what code can check checked. Raises
    ValueError with the reason when it cannot be tried."""
    from fleetopt.drive import entry as entry_mod

    project = RUN["project"]
    if not isinstance(plan, dict):
        raise ValueError("the entry must be a JSON object")
    graph = plan.get("graph")
    if not isinstance(graph, str) or ":" not in graph:
        raise ValueError(f"no graph named as file.py:name or module:name (got {graph!r})")
    graph = graph.removeprefix("./")
    if graph.split(":")[0].endswith(".py") and not (project / graph.split(":")[0]).exists():
        raise ValueError(f"the graph names {graph.split(':')[0]}, which is not in the project")
    python = RUN["python"]
    if plan.get("interpreter"):
        named = (project / plan["interpreter"]).resolve()
        if not named.is_relative_to(project) or not named.exists():
            raise ValueError(f"the interpreter must be one inside the project; {plan['interpreter']} is not")
        python = str(named)
    dropped = sorted(k for k in (plan.get("env") or {}) if SECRET.search(k))
    inputs = RUN["inputs"] or [t for t in plan.get("inputs") or [] if isinstance(t, str) and t.strip()]
    if not inputs:
        raise ValueError("no inputs: give the ones the agent's users would send")
    entry = {"adapter": "langgraph", "project": str(project), "name": plan.get("agent") or "agent", "graph": graph,
             "paths": [str(p) for p in plan.get("paths") or ["."]], "interpreter": python,
             "env_file": plan.get("env_file"),
             "env": {k: str(v) for k, v in (plan.get("env") or {}).items() if k not in dropped},
             "config": plan.get("config") or {}, "context": plan.get("context") or {}, "store": plan.get("store"),
             "input_template": plan.get("input_template"), "inputs": entry_mod._spread(inputs),
             "inputs_source": RUN["source"] or plan.get("inputs_from") or "written by fleetopt", "proven": None}
    return entry, dropped


@tool(
    "try_start",
    "Start the agent on one input with this entry, under fleetopt's probe, and say what happened: "
    "requests seen and finished, model calls seen, the first error, the last lines it printed. "
    "`entry`: the entry as a JSON object (see your instructions). At most 4 trials: each one runs the "
    "agent on the team's key.",
    {"entry": str},
)
async def try_start(args):
    from fleetopt.drive import entry as entry_mod

    if RUN["trials"] >= TRIALS:
        return _ok(f"Refused: {TRIALS} trials, the most a start gets. Give your answer with what you know.")
    try:
        entry, dropped = _checked(json.loads(args.get("entry") or ""))
    except ValueError as exc:  # JSONDecodeError included
        return _ok(f"Not tried: {exc}.")
    RUN["trials"] += 1
    entry_mod.save(RUN["path"], entry)
    print(f"[fleetopt] trying the agent on one input ({entry['graph']})", flush=True)
    result = await asyncio.to_thread(trial, RUN["path"], entry)
    with RUN["path"].with_suffix(".log").open("a", encoding="utf-8") as log:
        log.write(f"--- trial {RUN['trials']}, {datetime.datetime.now():%H:%M:%S}\n"
                  f"{json.dumps({k: v for k, v in entry.items() if k != 'env'})}\n{result['tail']}\n")
    started = bool(result["exit"] == 0 and result["requests"] and result["answered"])
    if started:
        RUN["proven"] = {**entry, "broken": result["error"] if not result["finished"] else None}
    print(f"[fleetopt] {'it started' if started else 'it did not start'}: {result['requests']} request(s), "
          f"{result['finished']} finished, {result['model_calls']} model call(s)", flush=True)
    facts = (f"exit {result['exit']}{' (stopped: no answer within 10 minutes)' if result['timed_out'] else ''}; "
             f"{result['requests']} request(s) seen, {result['finished']} finished; {result['model_calls']} model "
             f"call(s) seen, {result['answered']} answered; first error: {result['error'] or 'none'}")
    if started and result["finished"]:
        verdict = "It started: fleetopt will use this entry."
    elif started:
        verdict = "It started, and no request finished: it is carried on with as broken, which a review is for."
    elif result["requests"] and not result["model_calls"]:
        verdict = "It ran, and fleetopt saw no model call: find out how it calls its model."
    else:
        verdict = "It did not start."
    note = f"\nLeft out of env, as they look like credentials: {', '.join(dropped)}." if dropped else ""
    return _ok(f"{verdict}\n{facts}{note}\n\nLast lines it printed:\n{result['tail']}\n\n"
               f"{TRIALS - RUN['trials']} trial(s) left.")


def _ok(text):
    return {"content": [{"type": "text", "text": text}]}


def _json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(match.group(0)) if match else {}
    except ValueError:
        return {}


def options(project):
    """What the start session may be. No shell and no writing: it reads, and runs the
    agent only through try_start."""
    from claude_agent_sdk import ClaudeAgentOptions

    reads = ["Read", "Grep", "Glob"]
    return ClaudeAgentOptions(
        cwd=str(project), model=os.environ.get("FLEETOPT_MODEL") or "claude-sonnet-5", system_prompt=SKILL,
        tools=reads, mcp_servers={"fleetopt": create_sdk_mcp_server(name="fleetopt", tools=[try_start])},
        allowed_tools=[*reads, "mcp__fleetopt__try_start"],
        disallowed_tools=["AskUserQuestion", *config.DENY_READS], skills=[],
        setting_sources=config.SETTING_SOURCES, extra_args=config.sdk_args(), strict_mcp_config=True,
        env=config.SDK_ENV, max_turns=40, max_budget_usd=1.0, permission_mode="default",
    )


async def _session(prompt, project):
    from claude_agent_sdk import AssistantMessage, ClaudeSDKError, ResultMessage, TextBlock, query

    from fleetopt.progress import ticking

    text, final = "", None
    try:
        async with ticking("working out how to start the agent"):
            async for message in query(prompt=prompt, options=options(project)):
                if isinstance(message, AssistantMessage):
                    text = "".join(b.text for b in message.content if isinstance(b, TextBlock)) or text
                elif isinstance(message, ResultMessage) and not message.is_error:
                    final = message.result
    except ClaudeSDKError as exc:
        # Out of turns or budget arrives as an exception; a proven entry still stands.
        print(f"[fleetopt] the start-up session ended early: {str(exc).splitlines()[0][:200]}")
    return final or text


def settle(project, path, *, python, inputs=(), source=None, wanted=None):
    """Work out how to start the agent. Returns (the session's answer, the entry a trial
    proved or None). `inputs`: the team's eval cases, when there are any; the session is
    then asked for none."""
    RUN.clear()
    RUN.update(project=pathlib.Path(project).resolve(), path=pathlib.Path(path), python=python,
               inputs=list(inputs), source=source, trials=0, proven=None)
    ask = [f"Provider keys that are set (names only): {', '.join(key_names(RUN['project'])) or 'none found'}.",
           f"The interpreter fleetopt found: {python}."]
    if wanted:
        ask.append(f"The person running fleetopt asked for this agent: {wanted}.")
    ask.append(f"Inputs: fleetopt runs it on the team's eval cases ({len(inputs)}, the first: {inputs[0][:120]!r}); "
               "leave `inputs` out." if inputs else "Inputs: give 4 that differ in kind.")
    answer = _json(asyncio.run(_session("\n\n".join(ask), RUN["project"])))
    return answer, RUN["proven"]
