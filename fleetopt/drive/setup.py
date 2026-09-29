"""The two steps of starting an agent that need a model: writing inputs when the
project has none, and working out why an entry did not start.

Both are read-only sessions with their own context, like the judge and the reviewer.
They never see a secret (config.DENY_READS) and what they return is data that fleetopt
checks by running it, never a command.
"""

import asyncio
import json
import os
import re

from fleetopt import config

ALLOWED = ("graph", "paths", "env", "config", "context", "store", "input_template", "inputs")

REPAIR = """You work out how to start a LangGraph agent so that a measurement tool can run it.

The tool runs a small driver inside the project's own interpreter. The driver reads an
entry (JSON), puts `paths` on sys.path, loads `env_file`, sets `env`, imports `graph`,
and for each text in `inputs` calls graph.ainvoke(input, config, context=...). The entry it tried
and what happened are below. Read the project's source and reply with the fields to
change, as one JSON object and nothing else.

Fields you may set:
- "graph": "path/to/file.py:name", or "package.module:name". End with "()" to call a
  function that takes no arguments and returns the compiled graph.
- "paths": directories, relative to the project, to put on sys.path.
- "env": plain settings the agent needs (a provider name, a model, a data folder).
  NEVER a key, token or password: those come from the project's env file.
- "config": extra values for config["configurable"] (a tenant id, a user id).
- "context": the run context, for a graph built with a context_schema whose nodes read
  runtime.context (a user id, a model name). Passed as graph.ainvoke(..., context=...).
- "store": "memory", for a graph whose nodes read runtime.store or take a store
  argument and which is compiled without one, because a hosting platform supplies it.
  The driver then attaches an empty in-memory store, as `langgraph dev` does.
- "input_template": the graph's input as JSON, with the string "{input}" where the
  user's text goes. Needed when the input is not just messages or one text field.
- "inputs": only if the current ones cannot work for this agent.

Rules: change as little as possible. A user id, tenant id or name that you have to
make up is a neutral placeholder such as "fleetopt-user", unless the project's own data
or tests name one; never anything about whoever is running this tool. Prefer what the
project's own entry point does
(its CLI, its API handler, its tests). If nothing can make it start - a missing
dependency, a service it needs, a key that is absent - reply {"cannot": "<the reason>"}.
"""

WHICH = """A project contains several LangGraph agents. Decide which ONE the team ships and
tests: the top-level agent. Not a building block that another agent is made from, and
not an earlier version kept for teaching or comparison.

Read what a person would read: the README, the code that builds each agent, and above
all the team's own tests and eval runner, which name the agent they care about.

Reply with one JSON object and nothing else:
{"name": "<exactly one of the names given>", "why": "<the reason in at most 20 words, starting with a lower-case word>"}"""

INPUTS = """You write test inputs for an AI agent so that a measurement tool can run it.

Write what the agent's END USER types to it: the customer, the employee, the person
the agent was built to serve. NOT what a developer working on this repository would
ask. "How do I add a tenant" or "why does langgraph dev fail" are questions about the
repo; a sales agent's user asks about roses and delivery.

Work it out first by reading: the agent's system prompt, its tools, and the data it
works over (a catalogue, a policy folder, a database). Then write {n} messages in that
person's voice that differ in kind, so that they exercise different tools and paths.
Use names, products and ids that exist in the project's own data. Where the person
writing has to have a name, a city or an employer, invent them. Never use the name,
email or anything else about whoever is running this tool: these texts are sent to the
agent's model provider.

Reply with a JSON list of {n} strings and nothing else."""


async def _ask(system, prompt, cwd=None, tools=(), what="reading the agent's code"):
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKError, ResultMessage, TextBlock,
                                  query)

    options = ClaudeAgentOptions(
        cwd=str(cwd) if cwd else None,
        model=os.environ.get("FLEETOPT_MODEL") or "claude-sonnet-5",
        system_prompt=system, tools=list(tools), allowed_tools=list(tools),
        disallowed_tools=["AskUserQuestion", *config.DENY_READS], skills=[],
        setting_sources=config.SETTING_SOURCES, extra_args=config.sdk_args(), strict_mcp_config=True,
        env=config.SDK_ENV, max_turns=40 if tools else 1, max_budget_usd=0.5, permission_mode="default",
    )
    from fleetopt.progress import ticking

    text, final = "", None
    try:
        async with ticking(what):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    text = "".join(b.text for b in message.content if isinstance(b, TextBlock)) or text
                elif isinstance(message, ResultMessage) and not message.is_error:
                    final = message.result
    except ClaudeSDKError as exc:
        # Observed: a session that ran out of turns ended fleetopt in a traceback. A helper
        # that did not answer is no answer; every caller already knows what to do with that.
        print(f"[fleetopt] a helper session ended without an answer: {str(exc).splitlines()[0][:200]}")
        return ""
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


def choose_agent(project, found):
    """(name, why) for the agent the team ships, or None when it cannot be told."""
    listed = "\n".join(f"- {name}: {spec}" for name, spec in found)
    answer = _json(asyncio.run(_ask(WHICH, f"The agents in this project:\n{listed}", cwd=project,
                                    tools=("Read", "Grep", "Glob"), what="reading the project to pick the agent")))
    if isinstance(answer, dict) and answer.get("name") in {name for name, _ in found}:
        why = " ".join(str(answer.get("why", "")).split()).rstrip(".")
        words = why.split()
        why = " ".join(words[:30]) + (" ..." if len(words) > 30 else "")
        return answer["name"], (why[:1].lower() + why[1:]) or "it is the agent the team ships"
    return None


def propose_inputs(project, spec, n=4):
    """Inputs for an agent whose project keeps none. A read-only session that reads the
    agent's prompt, tools and data before writing anything."""
    prompt = f"The agent is {spec}, in the current directory. Write {n} inputs for it."
    found = _json(asyncio.run(_ask(INPUTS.format(n=n), prompt, cwd=project, tools=("Read", "Grep", "Glob"),
                                   what="writing test inputs")))
    texts = [t for t in found if isinstance(t, str) and t.strip()] if isinstance(found, list) else []
    if not texts:
        raise RuntimeError("could not write inputs for this agent: the project has no eval cases, no input file, "
                           "and nothing to derive them from")
    return texts[:n]
