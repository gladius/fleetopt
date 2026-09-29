"""Reads the team's own eval results, before and after a change: did anything that passed break?

A separate model call with a clean context. It sees the eval command and the two outputs, and
nothing else: not the change, not the reasoning, not the saving. The session that made the
change does not get to grade it. It reads whatever the team's eval system prints (pytest,
deepeval, a LangSmith or Galileo script, their own), so no framework is written into fleetopt.
It fails closed: no answer, an unreadable one, or none in time is a failure.
"""

import asyncio
import json
import os

PROMPT = """Two runs of a team's own eval suite for their AI agent, with the same command: BEFORE a
change to the agent's code, and AFTER it. Decide whether the change broke anything.

It broke something if any test, case, metric or score that passed or met its threshold BEFORE
fails or falls below it AFTER, if AFTER ran fewer evals, or if AFTER did not finish. Something
that already failed BEFORE and still fails is not the change's fault. A score that moved but
still meets its threshold is fine.

COMMAND
{command}

BEFORE (the end of its output)
{before}

AFTER (the end of its output)
{after}

Reply with JSON only: {{"broke": true|false, "what": ["each eval that passed before and fails now"], "reason": "<one sentence>"}}"""

SYSTEM = "You read eval results for a central AI team. Reply with JSON only."
TIMEOUT_S = 300  # observed: a judge call that waited 38 minutes, silently, on the login's usage limit


async def _ask(prompt, model=None):
    """One call through the same Claude Code binary as the agent, so it authenticates the same
    way, in its own process with no tools and one turn. Returns the parsed JSON, or
    {"_unparseable": text}; raises if the call fails."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

    from fleetopt import config

    options = ClaudeAgentOptions(
        model=model or os.environ.get("FLEETOPT_JUDGE_MODEL", "claude-haiku-4-5-20251001"), max_turns=1, tools=[],
        allowed_tools=[], system_prompt=SYSTEM, setting_sources=config.SETTING_SOURCES, extra_args=config.sdk_args(),
        env=config.SDK_ENV, permission_mode="default")
    text, failure = "", None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            text += "".join(b.text for b in message.content if isinstance(b, TextBlock))
        elif isinstance(message, ResultMessage) and message.is_error:
            failure = message.result or message.subtype
    if failure or not text.strip():
        raise RuntimeError(f"the eval reader failed: {failure or 'no answer'}")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_unparseable": text[:200]}


async def compare(command, before, after, model=None, timeout=TIMEOUT_S):
    """(held, reason): whether everything that passed BEFORE still passes AFTER."""
    try:
        answer = await asyncio.wait_for(_ask(PROMPT.format(command=command, before=before[-12000:],
                                                           after=after[-12000:]), model), timeout)
    except asyncio.TimeoutError:
        return False, f"the eval reader gave no answer in {timeout // 60} minutes"
    except RuntimeError as exc:
        return False, str(exc)
    if "_unparseable" in answer or not isinstance(answer.get("broke"), bool):
        return False, f"the eval reader's answer could not be read: {str(answer)[:120]}"
    if answer["broke"]:
        what = ", ".join(map(str, answer.get("what") or []))[:300]
        return False, (f"{what}: " if what else "") + str(answer.get("reason", ""))[:300]
    return True, str(answer.get("reason", ""))[:300]
