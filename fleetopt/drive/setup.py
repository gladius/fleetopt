"""Working out how to run a project's agent: one read-only session, guided by SETUP.md.

The session reads the project the way a developer joining the team would and answers
with an entry: which graph, how it is called, which settings, which inputs, or what is
missing that only the team can provide. It never runs anything, never sees a secret
(config.DENY_READS), and its answer is data that fleetopt checks by running it.

This replaced a set of rules (scan the source for compiled graphs, guess the input
shape, switch providers, give hosted agents a store). Each rule came from one agent,
and the next agent broke a different one.
"""

import asyncio
import json
import os
import pathlib
import re

from fleetopt import config

SKILL = (pathlib.Path(__file__).parent / "SETUP.md").read_text(encoding="utf-8")
FIELDS = ("graph", "agent", "why", "others", "paths", "env_file", "env", "config", "context", "store",
          "input_template", "inputs", "inputs_from", "missing", "agent_fault")
SECRET = re.compile(r"key|token|secret|password|credential", re.I)


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


def checked(answer, project):
    """The session's answer, with what code can check checked: a graph that names a file
    names one that exists, and no setting that looks like a credential gets through.
    Raises ValueError with the reason when the answer cannot be used."""
    if not isinstance(answer, dict):
        raise ValueError("the answer was not the JSON object asked for")
    plan = {k: answer.get(k) for k in FIELDS}
    plan["missing"] = [str(m) for m in plan["missing"] or [] if str(m).strip()]
    if plan["missing"]:
        return plan
    graph = plan["graph"]
    if not isinstance(graph, str) or ":" not in graph:
        raise ValueError(f"no graph named as file.py:name or module:name (got {graph!r})")
    target = graph.split(":")[0]
    if target.endswith(".py") and not (project / target.removeprefix("./")).exists():
        raise ValueError(f"the graph names {target}, which is not in the project")
    plan["graph"] = graph.removeprefix("./")
    plan["env"] = {k: str(v) for k, v in (plan["env"] or {}).items() if not SECRET.search(k)}
    plan["paths"] = [str(p) for p in plan["paths"] or ["."]]
    plan["inputs"] = [t for t in plan["inputs"] or [] if isinstance(t, str) and t.strip()]
    return plan


def settle(project, *, keys, have_inputs, wanted=None, earlier=None, failure=None):
    """An entry for this project's agent, as the session worked it out and code checked
    it. `keys`: names of the provider keys that are set (names only). `have_inputs`:
    whether fleetopt already has the inputs (the team's eval cases), so none are asked
    for. `earlier`, `failure`: the answer that was tried and what happened."""
    ask = [f"Provider keys that are set (names only): {', '.join(keys) or 'none found'}."]
    if wanted:
        ask.append(f"The person running fleetopt asked for this agent: {wanted}.")
    ask.append("Inputs: fleetopt has them already; leave `inputs` empty." if have_inputs else
               "Inputs: give 4 that differ in kind.")
    if earlier:
        ask.append(f"This answer was tried:\n{json.dumps({k: v for k, v in earlier.items() if k != 'env'}, indent=1)}"
                   f"\n\nWhat happened:\n{failure[-3000:]}")
    what = "reading the project to work out how to run its agent" if not earlier else "working out why it did not answer"
    text = asyncio.run(_ask(SKILL, "\n\n".join(ask), cwd=project, tools=("Read", "Grep", "Glob"), what=what))
    return checked(_json(text), project)
