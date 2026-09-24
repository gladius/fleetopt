"""Output-equivalence gate.

Deliberately a separate API call with a clean context, not something the
optimizer decides about its own work. The optimizer proposed the patch and wants
it to pass; give it the verdict too and you get motivated reasoning wearing a
verdict's clothes. The judge sees the task, the two outputs, and nothing else -
not the patch, not the reasoning, not the savings figure.
"""

import json
import os

PROMPT = """You are checking whether an optimization broke an AI agent.

Two outputs for the same input: one from the original agent, one after a change
intended to make it cheaper. Decide whether the second is still an acceptable
answer to the same task.

Judge substance, not wording. Different phrasing, ordering, or length is fine.
Missing information, wrong facts, a narrower answer, or a dropped step is not.

TASK
{task}

INPUT
{input}

ORIGINAL OUTPUT
{baseline}

NEW OUTPUT
{candidate}

Reply with JSON only: {{"equivalent": true|false, "reason": "<one sentence>"}}"""


SYSTEM = "You are an output-equivalence judge for AI agents. Reply with JSON only."


async def judge(task, agent_input, baseline, candidate, model=None):
    """Returns {"equivalent": bool, "reason": str}.

    Runs through the same Claude Code binary as the optimizer, so it authenticates
    the same way (login, settings.json, gateway, cloud) - but in its own process
    with its own clean context, no tools, one turn. With `tools=[]` the call is
    ~400 input tokens, the same as a direct API call.

    Raises if the call fails - a gate that fails open is not a gate.
    """
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

    from fleetopt import config

    model = model or os.environ.get("FLEETOPT_JUDGE_MODEL", "claude-haiku-4-5-20251001")
    options = ClaudeAgentOptions(
        model=model,
        max_turns=1,
        tools=[],
        allowed_tools=[],
        system_prompt=SYSTEM,
        setting_sources=config.SETTING_SOURCES,
        env=config.SDK_ENV,
        permission_mode="default",
    )
    prompt = PROMPT.format(
        task=task,
        input=str(agent_input)[:2000],
        baseline=str(baseline)[:4000],
        candidate=str(candidate)[:4000],
    )

    text, failure = "", None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            text += "".join(b.text for b in message.content if isinstance(b, TextBlock))
        elif isinstance(message, ResultMessage) and message.is_error:
            failure = message.result or message.subtype
    if failure or not text.strip():
        raise RuntimeError(f"judge call failed: {failure or 'empty response'}")

    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()

    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        # An unparseable verdict is a failed check, not a pass.
        return {"equivalent": False, "reason": f"judge returned unparseable output: {text[:200]}"}

    return {
        "equivalent": bool(verdict.get("equivalent")),
        "reason": str(verdict.get("reason", "")),
    }


async def judge_sessions(conn, task, baseline_session, candidate_session):
    """Pair root runs across two sessions and judge each pair.

    Pairs by position, not by matching the input text. Real agents thread
    generated ids through their state - message uuids, thread ids, timestamps -
    so two runs of the same command produce inputs that never compare equal. Both
    sides ran the same command in the same order, so the k-th invocation on one
    side answers the k-th on the other.
    """

    def roots(session_id):
        return [
            (r["inputs"], r["outputs"])
            for r in conn.execute(
                "SELECT inputs, outputs FROM runs"
                " WHERE session_id = ? AND parent_run_id IS NULL AND outputs IS NOT NULL"
                " ORDER BY start_time",
                (session_id,),
            )
        ]

    before, after = roots(baseline_session), roots(candidate_session)
    if not before or not after:
        raise RuntimeError(
            "no comparable invocations - both sides must be captured with IO "
            f"(baseline has {len(before)}, candidate has {len(after)})"
        )
    if len(before) != len(after):
        raise RuntimeError(
            f"the two sides ran a different number of invocations "
            f"({len(before)} vs {len(after)}) - they are not comparable"
        )

    results = []
    for (agent_input, baseline_out), (_, candidate_out) in zip(before, after):
        verdict = await judge(task, agent_input, baseline_out, candidate_out)
        verdict["input"] = (agent_input or "")[:120]
        results.append(verdict)

    return all(r["equivalent"] for r in results), results
