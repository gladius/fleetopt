"""The optimizer agent.

The agent drives. There is no step list here - it reads the project, decides what
to look at, and proves whatever it claims using the tools in tools.py. What this
module owns is the boundary: where the agent runs, what it may touch, and the two
things it has to ask about.

Permission is gated by class of side effect, not per tool. Reading, querying,
measuring and judging are free. Running the target's own command is asked once.
Touching the repository is asked once, and after that git is the undo.
"""

import asyncio
import os
import pathlib
import re

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from fleetopt import config
from fleetopt.optimizer import tools

_HERE = pathlib.Path(__file__).parent
SKILL = (_HERE / "SKILL.md").read_text(encoding="utf-8")
PLUGIN = _HERE / "plugin"  # decision skills, loaded by the harness, triggered by description

MISSION = """Optimize the LangGraph agent in this project so it costs less to run,
without changing what it produces.

Work in this order, but use your judgement - the project decides the details:

1. Understand it. Read the source and the graph topology. What is this agent for?
2. Establish a baseline. Find how to run it once end to end, set that as the run
   command, then measure it. Without a baseline nothing you do afterwards is
   provable.
3. Find the cost. Query the traces. Go where the tokens are.
4. Change one thing. Create a git branch first, then apply a single optimization.
5. Prove it. Measure again under a new label, compare, and judge equivalence.

Report at the end: what you changed, the measured difference, and the equivalence
verdict. If the saving was within noise, or equivalence failed, say so plainly and
leave the branch for review. A cost reduction that broke the agent is a
regression, not a result."""

STOP = ("The user declined to let you {kind}. This is final for the session: do not retry, "
        "do not look for another way to do it. Write your final report now with what you "
        "established from read-only evidence, and state plainly what you could not do.")

RUNS_TARGET = re.compile(r"\bpytest\b|\blanggraph\s+dev\b|\bpython[\d.]*\s+(?!-c\b|-m\s+(?:pip|venv|py_compile)\b)(?:-m\s+)?[\w./-]+")

# Installs and downloads. Observed under --auto: handed an interpreter without
# langgraph, the agent ran `uv run --with langgraph ...` and pulled the packages
# from the internet. Right for a sandbox, wrong on someone else's machine.
ENV_MUTATION = re.compile(
    r"\b(?:pip3?\s+(?:install|uninstall)|python[\d.]*\s+-m\s+pip\s+(?:install|uninstall)"
    r"|uv\s+(?:pip|add|sync|tool|run\s+--with)|pipx\s+(?:install|run)"
    r"|poetry\s+(?:add|install|update)|pdm\s+(?:add|install)|(?:conda|mamba)\s+(?:install|create)"
    r"|(?:npm|pnpm|yarn)\s+(?:install|add|i)\b|npx\s|apt(?:-get)?\s+install|(?:dnf|yum)\s+install"
    r"|brew\s+install|curl\s|wget\s)"
)


def _deny(reason):
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny", "permissionDecisionReason": reason}}


def guard_bash(run_cmd):
    """PreToolUse hook on Bash. Two rules, both enforced rather than asked for.

    The target runs only through `measure`: running it by hand spends the team's
    tokens twice, captures nothing, and fed 12K tokens of pytest tracebacks into
    the optimizer's context last time.

    The target's environment is not ours to change: no installs, no downloads. A
    missing dependency is a finding about the run command, not something to fix."""
    async def hook(input_data, tool_use_id, context):
        cmd = (input_data.get("tool_input") or {}).get("command", "")
        if ENV_MUTATION.search(cmd):
            return _deny("fleetopt never installs packages or downloads anything into the "
                         "target's environment. Find the interpreter that already has the "
                         "project's dependencies (its .venv, `uv run`/`poetry run` if the project "
                         "uses them, a Makefile target) and set that as the run command. If none "
                         "exists, report it - that is the team's finding.")
        hits_run_cmd = bool(run_cmd) and run_cmd.split()[-1] in cmd
        if hits_run_cmd or RUNS_TARGET.search(cmd):
            return _deny("The target runs only through the measure tool. Use measure; if it "
                         "fails, read its error instead of reproducing it.")
        return {}
    return hook


# Read-only git is not a repository mutation; don't spend the user's attention on it.
SAFE_GIT = ("status", "diff", "log", "branch --list", "rev-parse", "show", "ls-files")


class Gate:
    """Asks once per class of side effect, then stays out of the way."""

    def __init__(self, auto=False):
        self.auto = auto
        self.granted = set()
        self.denied = set()

    @staticmethod
    def _classify(tool_name, data):
        if tool_name in ("mcp__fleetopt__measure", "mcp__fleetopt__set_run_command"):
            return "execute"
        if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
            return "mutate"
        if tool_name == "Bash":
            cmd = (data.get("command") or "").strip()
            if cmd.startswith("git ") and any(cmd.startswith(f"git {s}") for s in SAFE_GIT):
                return None
            return "mutate"
        return None

    async def __call__(self, tool_name, data, context):
        kind = self._classify(tool_name, data)
        if kind is None or self.auto or kind in self.granted:
            return PermissionResultAllow()
        if kind in self.denied:  # asked once, answered once - do not nag, do not let it retry
            return PermissionResultDeny(message=STOP.format(kind=kind))

        question = {
            "execute": "The optimizer wants to RUN this project's own command.\n"
                       f"  {data.get('cmd') or data.get('label', '')}\n"
                       "  This executes their code and may cost API tokens.",
            "mutate": "The optimizer wants to MODIFY files in this repository.\n"
                      "  It will work on a git branch, so this is reversible.",
        }[kind]

        print(f"\n--- permission ---\n{question}")
        try:
            answer = await asyncio.to_thread(input, "  allow for this session? [y/N] ")
        except EOFError:  # no terminal - treat as a decline, not a crash
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            self.denied.add(kind)
            return PermissionResultDeny(message=STOP.format(kind=kind))

        self.granted.add(kind)
        return PermissionResultAllow()


async def run(project, out_dir, run_cmd=None, auto=False, model=None, max_turns=60, max_usd=None, effort=None):
    project = pathlib.Path(project).resolve()
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    tools.CTX.update({"project": project, "out": out, "run_cmd": run_cmd, "run_locked": bool(run_cmd)})
    mission = MISSION
    if run_cmd:
        mission += (f"\n\nThe run command is already set: `{run_cmd}`. Do not rediscover or "
                    "change it - start with measure.")

    options = ClaudeAgentOptions(
        cwd=str(project),
        model=model,
        system_prompt=(
            "You are a cost optimizer for LangGraph agents, run by a central AI "
            "team on another team's project. Evidence beats intuition: every claim "
            "you make must cite trace data or a measurement. Never report a saving "
            "you have not measured.\n\n" + SKILL
        ),
        mcp_servers={"fleetopt": tools.server()},
        # Only ungated tools go here. An allowed_tools entry auto-approves before
        # can_use_tool is consulted, so anything the Gate must see is left out and
        # falls through to it (the SDK warns about this: CanUseToolShadowedWarning).
        allowed_tools=[t for t in tools.TOOL_NAMES if not t.endswith(("measure", "set_run_command"))]
        + ["Read", "Grep", "Glob", "Skill"],
        plugins=[{"type": "local", "path": str(PLUGIN)}],
        can_use_tool=Gate(auto),
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[guard_bash(run_cmd)])]},
        # Optional. Lower effort cuts the optimizer's own output/thinking tokens;
        # unverified for finding quality, so off unless FLEETOPT_EFFORT is set.
        effort=effort,
        # Headless-capable: the agent must not block on a question nobody will answer.
        # And it reads the target's source, never its secrets (see config.DENY_READS).
        disallowed_tools=["AskUserQuestion", *config.DENY_READS],
        # Authenticates like Claude Code on this machine (login, settings.json env
        # block, apiKeyHelper, cloud switches) - loads user settings for that, and
        # nothing from the target repo's own .claude/. See config.py.
        setting_sources=config.SETTING_SOURCES,
        env=config.SDK_ENV,
        max_turns=max_turns,
        # Caps the optimizer's own spend. The target's API calls go through the
        # target's key and are not counted here.
        max_budget_usd=max_usd,
        permission_mode="default",
    )

    async def prompt():
        yield {"type": "user", "message": {"role": "user", "content": mission}}

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt())
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)
                    elif isinstance(block, ToolUseBlock):
                        print(f"  - {block.name.replace('mcp__fleetopt__', '')}")
            elif isinstance(message, ResultMessage):
                cost = getattr(message, "total_cost_usd", None)
                print(f"\n--- done in {message.num_turns} turns" +
                      (f", ${cost:.4f}" if cost else "") + " ---")
    return 0
