"""Did a change break anything? Read by a separate model call, before and after.

Two questions, for what a project has: the output of the team's own eval suite (`compare`),
or the agent's answers to examples of what it is sent, with expected answers when the team
has a golden dataset (`compare_answers`, one call a request so each answer is read whole),
beside a second run of the original so the reader sees how much its answers vary by
themselves. A clean context that sees only that: not the change, not the reasoning, not the
saving. The session that made the change does not get to grade it. It
reads whatever the team's system prints, so no framework is written into fleetopt. It fails
closed: no answer, an unreadable one, or none in time is a failure.
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

If BEFORE did not run any eval at all (the command was not found, nothing was collected, it
stopped before the first case), nothing is proven by these runs: reply broke: true, with the
reason "the evals did not run before the change". Two runs that failed the same way are not a
pass.

COMMAND
{command}

BEFORE (the end of its output)
{before}

AFTER (the end of its output)
{after}

Reply with JSON only: {{"broke": true|false, "what": ["each eval that passed before and fails now"], "reason": "<one sentence>"}}"""

ANSWERS = """A request was sent to a team's AI agent BEFORE a change to its code and AFTER it. Decide
whether the change broke its answer.

BEFORE and AFTER are what the agent returned for the request: its whole final state, so the
answer is usually at the end of it. BEFORE AGAIN, where shown, is a second run of the unchanged
agent on the same request: where it differs from BEFORE, that is how much this agent's answers
vary by themselves, and no fault of the change.

It broke the answer if AFTER is wrong by the EXPECTED answer, where one is given, while the
unchanged agent was right, or, where none is given, is a worse answer to the request than the
unchanged agent's: it misses or gets wrong something every run of the unchanged agent got
right, or did not finish. Different wording, order or length is fine; so is an answer that was
already wrong and still is, and so is a difference no larger than the one between BEFORE and
BEFORE AGAIN.

WHAT THE AGENT IS FOR
{job}

{request}

Reply with JSON only: {{"broke": true|false, "reason": "<one sentence>"}}"""

SYSTEM = "You check whether a change to an AI agent broke anything. Reply with JSON only."
MODEL, FALLBACK = "haiku", "sonnet"  # aliases: whatever this Claude Code setup provides
USED = {}  # the model that last answered, as the session reported it
TIMEOUT_S = 300  # observed: a judge call that waited 38 minutes, silently, on the login's usage limit
ANSWER_CHARS = 100_000  # of one answer the reader is shown: three of them fit its window with room to spare
READERS = 4  # reader calls at once


async def _ask(prompt, model=None):
    """One call through the same Claude Code binary as the agent, so it authenticates the same
    way, in its own process with no tools and one turn. Returns the parsed JSON, or
    {"_unparseable": text}; raises if the call fails."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, SystemMessage, TextBlock, query

    from fleetopt import config

    options = ClaudeAgentOptions(
        model=model or os.environ.get("FLEETOPT_JUDGE_MODEL") or MODEL, fallback_model=FALLBACK, max_turns=1, tools=[],
        allowed_tools=[], system_prompt=SYSTEM, setting_sources=config.SETTING_SOURCES, extra_args=config.sdk_args(),
        env=config.SDK_ENV, permission_mode="default")
    text, failure = "", None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            USED["model"] = (message.data or {}).get("model")
        elif isinstance(message, AssistantMessage):
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


async def _verdict(prompt, model, timeout):
    """(held, reason) from one reader call."""
    try:
        answer = await asyncio.wait_for(_ask(prompt, model), timeout)
    except asyncio.TimeoutError:
        return False, f"the reader gave no answer in {timeout // 60} minutes"
    except RuntimeError as exc:
        return False, str(exc)
    if "_unparseable" in answer or not isinstance(answer.get("broke"), bool):
        return False, f"the reader's answer could not be read: {str(answer)[:120]}"
    if answer["broke"]:
        what = ", ".join(map(str, answer.get("what") or []))[:300]
        return False, (f"{what}: " if what else "") + str(answer.get("reason", ""))[:300]
    return True, str(answer.get("reason", ""))[:300]


async def compare(command, before, after, model=None, timeout=TIMEOUT_S):
    """(held, reason): whether everything the team's evals passed BEFORE still passes AFTER."""
    return await _verdict(PROMPT.format(command=command, before=before[-12000:], after=after[-12000:]), model, timeout)


def _ends(text, limit=ANSWER_CHARS):
    """A long answer keeps both ends. What is recorded is the graph's whole final state, and the
    answer is at its end: observed, cut to its first 3,000 characters, the reader never saw it."""
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit // 6
    return f"{text[:head]}\n[... {len(text) - limit:,} characters left out ...]\n{text[head - limit:]}"


def clipped(pairs):
    """How many of the answers were too long to be read whole."""
    return sum(len(str(p[k])) > ANSWER_CHARS for p in pairs for k in ("before", "again", "after") if p.get(k))


async def compare_answers(job, pairs, model=None, timeout=TIMEOUT_S):
    """(held, reason): whether any request's answer got worse. One reader call per request, so
    each answer is read whole and the request that broke is named by code, not by the reader.
    `pairs`: [{input, expected, before, again, after}], expected None when the team gave none,
    again a second run of the original."""
    gate = asyncio.Semaphore(READERS)

    async def read(p):
        expected = f"EXPECTED\n{str(p['expected'])[:3000]}\n" if p.get("expected") else ""
        again = f"BEFORE AGAIN\n{_ends(p['again'])}\n" if p.get("again") else ""
        request = f"INPUT\n{_ends(p['input'], 3000)}\n{expected}BEFORE\n{_ends(p['before'])}\n{again}AFTER\n{_ends(p['after'])}"
        async with gate:
            return await _verdict(ANSWERS.format(job=job or "not stated", request=request), model, timeout)

    verdicts = await asyncio.gather(*map(read, pairs))
    broke = [f"request {i} ({' '.join(str(p['input']).split())[:60]}): {why}"
             for i, (p, (held, why)) in enumerate(zip(pairs, verdicts), 1) if not held]
    return (False, "; ".join(broke)[:600]) if broke else (True, f"{len(pairs)} read, none worse")
