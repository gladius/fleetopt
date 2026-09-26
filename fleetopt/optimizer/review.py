"""The architecture reviewer: a separate, read-only session with its own context.

Same separation as the judge. The optimizer is about to patch and does not get to
grade the design in the same breath; the reviewer sees the traces and the source,
never the optimizer's reasoning, and returns a report the optimizer includes
verbatim. It runs as its own `query()` from inside the review_architecture tool,
so the call returns the report synchronously. (A Claude Code subagent was tried
first: it ran in the background, the optimizer "paused to wait for the
notification", ended its turn, and the review was lost.)
"""

import os
import pathlib

from fleetopt import config

_HERE = pathlib.Path(__file__).parent
GUIDE = (_HERE / "plugin" / "skills" / "patterns" / "SKILL.md").read_text(encoding="utf-8")

SYSTEM = (
    "You review the architecture of the LangGraph agent in the current directory for a central "
    "AI team. You may read source and query the capture database with the fleetopt tools; you never "
    "edit files, never run anything, never patch. Every finding cites a number from graph_shape or "
    "query_traces and a source location. Follow the guide below exactly and make your final message "
    "the report in its format, nothing else.\n\n" + GUIDE
)

READ_ONLY = ["Read", "Grep", "Glob", "mcp__fleetopt__graph_shape",
             "mcp__fleetopt__graph_topology", "mcp__fleetopt__query_traces"]


async def run(project, label, purpose, model=None, max_usd=1.0):
    """Returns (report text, cost in USD). Raises if the session fails."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

    from fleetopt.optimizer import tools

    options = ClaudeAgentOptions(
        cwd=str(project),
        model=model or os.environ.get("FLEETOPT_REVIEW_MODEL") or "claude-sonnet-5",
        system_prompt=SYSTEM,
        tools=["Read", "Grep", "Glob"],
        # A read-only server: no measure, judge, set_run_command, load_eval_cases,
        # compare, and not review_architecture itself (the first reviewer tried to call it).
        mcp_servers={"fleetopt": tools.server(READ_ONLY)},
        allowed_tools=READ_ONLY,
        disallowed_tools=["AskUserQuestion", *config.DENY_READS],
        skills=[],
        setting_sources=config.SETTING_SOURCES,
        extra_args=config.sdk_args(),
        strict_mcp_config=True,
        env=config.SDK_ENV,
        max_turns=25,
        max_budget_usd=max_usd,
        permission_mode="default",
    )
    prompt = (f"Review the agent in this directory. Baseline traces are under the label {label!r}. "
              f"What it is for: {purpose}")
    texts, final, failure, cost = [], None, None, None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            texts += [b.text for b in message.content if isinstance(b, TextBlock) and b.text.strip()]
        elif isinstance(message, ResultMessage):
            if message.is_error:
                failure = message.result or message.subtype
            final = message.result
            cost = getattr(message, "total_cost_usd", None)
    if failure:
        raise RuntimeError(f"reviewer failed: {failure}")
    report = (final or (texts[-1] if texts else "")).strip()
    if not report:
        raise RuntimeError("reviewer returned nothing")
    return report, cost
