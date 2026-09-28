"""The two steps of starting an agent that need a model: writing inputs when the
project has none, and working out why an entry did not start.

Both are read-only sessions with their own context, like the judge and the reviewer.
They never see a secret (config.DENY_READS) and what they return is data that fleetopt
checks by running it, never a command.
"""

import asyncio
import json
import os
import pathlib
import re

from fleetopt import config

ALLOWED = ("graph", "paths", "env", "config", "input_template", "inputs")

REPAIR = """You work out how to start a LangGraph agent so that a measurement tool can run it.

The tool runs a small driver inside the project's own interpreter. The driver reads an
entry (JSON), puts `paths` on sys.path, loads `env_file`, sets `env`, imports `graph`,
and for each text in `inputs` calls graph.ainvoke(input, config). The entry it tried
and what happened are below. Read the project's source and reply with the fields to
change, as one JSON object and nothing else.

Fields you may set:
- "graph": "path/to/file.py:name", or "package.module:name". End with "()" to call a
  function that takes no arguments and returns the compiled graph.
- "paths": directories, relative to the project, to put on sys.path.
- "env": plain settings the agent needs (a provider name, a model, a data folder).
  NEVER a key, token or password: those come from the project's env file.
- "config": extra values for config["configurable"] (a tenant id, a user id).
- "input_template": the graph's input as JSON, with the string "{input}" where the
  user's text goes. Needed when the input is not just messages or one text field.
- "inputs": only if the current ones cannot work for this agent.

Rules: change as little as possible. Prefer what the project's own entry point does
(its CLI, its API handler, its tests). If nothing can make it start - a missing
dependency, a service it needs, a key that is absent - reply {"cannot": "<the reason>"}.
"""

INPUTS = """You write test inputs for an AI agent so that a measurement tool can run it.
From the README and source below, write {n} realistic requests a real user of this
agent would send. Make them differ in kind, so they exercise different paths. Reply
with a JSON list of {n} strings and nothing else."""


async def _ask(system, prompt, cwd=None, tools=()):
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

    options = ClaudeAgentOptions(
        cwd=str(cwd) if cwd else None,
        model=os.environ.get("FLEETOPT_MODEL") or "claude-sonnet-5",
        system_prompt=system, tools=list(tools), allowed_tools=list(tools),
        disallowed_tools=["AskUserQuestion", *config.DENY_READS], skills=[],
        setting_sources=config.SETTING_SOURCES, extra_args=config.sdk_args(), strict_mcp_config=True,
        env=config.SDK_ENV, max_turns=20 if tools else 1, max_budget_usd=0.5, permission_mode="default",
    )
    text, final = "", None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            text = "".join(b.text for b in message.content if isinstance(b, TextBlock)) or text
        elif isinstance(message, ResultMessage) and not message.is_error:
            final = message.result
    return (final or text or "").strip()


def _json(text):
    match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    try:
        return json.loads(match.group(1)) if match else None
    except ValueError:
        return None


def repair(project, entry, failure):
    """Fields to change in the entry, or None when it cannot be started."""
    shown = {k: v for k, v in entry.items() if k != "env"}  # settings only ever flow one way
    prompt = f"The entry that was tried:\n{json.dumps(shown, indent=1)}\n\nWhat happened:\n{failure[-3000:]}"
    fix = _json(asyncio.run(_ask(REPAIR, prompt, cwd=project, tools=("Read", "Grep", "Glob"))))
    if not isinstance(fix, dict) or "cannot" in fix:
        if isinstance(fix, dict):
            print(f"[fleetopt] cannot be started: {fix['cannot']}")
        return None
    fix = {k: v for k, v in fix.items() if k in ALLOWED}
    if any(re.search(r"key|token|secret|password", k, re.I) for k in (fix.get("env") or {})):
        fix.pop("env")  # a credential never travels through an entry
    return fix or None


def propose_inputs(project, spec, n=4):
    project = pathlib.Path(project)
    readme = next((p for p in (project / "README.md", project / "README.rst", project / "readme.md") if p.exists()), None)
    source = project / spec.split(":")[0]
    material = (f"README:\n{readme.read_text(encoding='utf-8', errors='replace')[:6000]}\n\n" if readme else "") + \
               (f"SOURCE ({source.name}):\n{source.read_text(encoding='utf-8', errors='replace')[:6000]}" if source.exists() else "")
    found = _json(asyncio.run(_ask(INPUTS.format(n=n), material or f"An agent at {spec}.")))
    texts = [t for t in found if isinstance(t, str) and t.strip()] if isinstance(found, list) else []
    if not texts:
        raise RuntimeError("could not write inputs for this agent: the project has no eval cases, no input file, "
                           "and nothing to derive them from")
    return texts[:n]
