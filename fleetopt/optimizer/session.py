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
import datetime
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess

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
# Only fleetopt's skills are listed to the model; the CLI's built-in ones are noise here.
SKILLS = sorted(f"fleetopt:{p.name}" for p in (PLUGIN / "skills").iterdir() if p.is_dir())

MISSION = """Optimize the LangGraph agent in this project so it costs less to run,
without changing what it produces.

Work in this order, but use your judgement - the project decides the details:

1. Understand it. Read the source and the graph topology. What is this agent for?
2. Establish a baseline. Find how to run it once end to end, set that as the run
   command, then measure it. Without a baseline nothing you do afterwards is
   provable. Before that, look for eval cases (see fleetopt:evals) and load them
   with load_eval_cases; prefer the suite that exercises them as the run command.
3. Find the cost. Query the traces. Go where the tokens are.
4. Change one thing. Create a git branch first, then apply a single optimization.
5. Prove it. Measure again under a new label, compare, and judge equivalence.

Report at the end: what you changed, the measured difference (dollars first, then
latency, then tokens), the equivalence verdict, and the correctness pass rate before and after if eval cases were loaded
(say "correctness not checked" if none were found). If the saving was within
noise, or equivalence or correctness failed, say so plainly and leave the branch
for review. A cost reduction that broke the agent is a
regression, not a result."""

REVIEW_MISSION = """

ARCHITECTURE REVIEW IS ON. Once the baseline is measured and before any patch, call
review_architecture once with the baseline label and one sentence on what this agent is
for. It runs a separate read-only reviewer and returns its report. Put that report
verbatim under a heading "Architecture review" in your final report, separate from the
savings. Do not act on tier-two items. Act on a tier-one item only if eval cases are
loaded and cover the affected path; otherwise leave it as a recommendation and say what
would unlock it."""

STOP = ("The user declined to let you {kind}. This is final for the session: do not retry, "
        "do not look for another way to do it. Write your final report now with what you "
        "established from read-only evidence, and state plainly what you could not do.")

# Eval runners count as running the target too: a deepeval or promptfoo suite
# invokes the agent on every case and bills the team for its own graders.
RUNS_TARGET = re.compile(
    r"\bpytest\b|\blanggraph\s+dev\b|\bdeepeval\s+test\b|\bpromptfoo\s+eval\b|\bbraintrust\s+eval\b"
    r"|\bpython[\d.]*\s+(?!-c\b|-m\s+(?:pip|venv|py_compile)\b)(?:-m\s+)?[\w./-]+"
)

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



def _git(project, *args):
    """Read-only git query on the target; '' when there is no repo or no git."""
    try:
        return subprocess.run(["git", "-C", str(project), *args],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _write_record(run_dir, project, start_sha, started, meta, texts, calls, skills, result):
    """What this run did: report.md for a human, run.json for the ledger and for
    improving fleetopt, patch.diff when the branch changed anything. No prompts or
    outputs of the target are stored; the 120-char input excerpts in judge rows and
    the diff are the only target content, so sharing a run folder is the operator's
    call, not automatic."""
    (run_dir / "report.md").write_text("\n\n".join(texts), encoding="utf-8")
    (run_dir / "log.txt").write_text("\n".join(calls), encoding="utf-8")
    diff = _git(project, "diff", start_sha) if start_sha else ""
    if diff:
        (run_dir / "patch.diff").write_text(diff + "\n", encoding="utf-8")
    try:
        version = importlib.metadata.version("fleetopt")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    record = {
        "fleetopt": version,
        "project": str(project),
        "code_state_before": start_sha,
        "branch_after": _git(project, "rev-parse", "--abbrev-ref", "HEAD"),
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        **meta,
        "run_cmd": tools.CTX.get("run_cmd"),
        **result,
        "skills": skills,
        "tool_calls": calls,
        "events": tools.CTX.get("events", []),
        "patch": "patch.diff" if diff else None,
        "report": "report.md",
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"[fleetopt] run record: {run_dir}")


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


def verdict(events):
    """What the measurements support, computed from what the tools recorded. Printed
    after the agent's report and stored in the run record, so a report cannot argue
    with a failed gate. Only a judged comparison of two different code states counts,
    and every judgment of the final code counts: one failure is a failure."""
    real = [e for e in events if e["event"] == "judge" and e.get("baseline_state") != e.get("candidate_state")]
    if not real:
        return "NOTHING PROVEN: no change was judged against the code it started from."
    final = real[-1]["candidate_state"]
    judged = [e for e in real if e["candidate_state"] == final]

    def detail(e):
        ok = sum(bool(r["equivalent"]) for r in e["equivalence"])
        text = f"{e['candidate']}: equivalence {ok}/{len(e['equivalence'])}"
        c = e.get("correctness")
        if c:
            return text + f", correctness {c['baseline_pass']}/{c['matched']} before and {c['candidate_pass']}/{c['matched']} after"
        return text + ", correctness not checked (no eval cases)"

    details = "; ".join(detail(e) for e in judged)
    measured = [e for e in events if e["event"] == "compare" and e.get("candidate_state") == final
                and e.get("baseline_state") != final]
    cost = measured[-1]["result"].get("cost_usd", {}) if measured else {}
    saving = (f" Cost ${cost['before']:.4f} to ${cost['after']:.4f} per run: {cost.get('verdict')}."
              if cost.get("before") is not None and cost.get("after") is not None else "")
    if not all(e["passed"] for e in judged):
        return (f"NOT PROVEN SAFE: the judge failed ({details}).{saving}"
                " A saving with a failed gate is not a result. The branch is left for review.")
    return f"PROVEN ON THIS EVIDENCE: the judge passed ({details}).{saving}"


def build_options(project, run_cmd=None, auto=False, model=None, max_turns=60, max_usd=None, effort=None):
    """Everything a session is allowed to be. Apart from run() so the product's
    promises can be read off it in a test without starting a session
    (tests/test_invariants.py)."""
    return ClaudeAgentOptions(
        cwd=str(project),
        model=model,
        system_prompt=(
            "You are a cost optimizer for LangGraph agents, run by a central AI "
            "team on another team's project. Evidence beats intuition: every claim "
            "you make must cite trace data or a measurement. Never report a saving "
            "you have not measured.\n\n" + SKILL
        ),
        mcp_servers={"fleetopt": tools.server()},
        # Built-ins by allowlist. The CLI default is 26 tools including web fetch and
        # search, cron, worktrees, messaging and wake-up scheduling: egress and mutation
        # surfaces an optimizer has no business with, and schema tokens on every turn.
        # Observed before this: a run called ScheduleWakeup to "wait" for its reviewer.
        # The reviewer is a separate query() behind review_architecture, not a subagent.
        tools=["Read", "Grep", "Glob", "Bash", "Edit", "Write", "Skill"],
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
        # block, apiKeyHelper, cloud switches) without inheriting the operator's
        # plugins, skills or MCP servers, or the target repo's .claude/. See config.py.
        setting_sources=config.SETTING_SOURCES,
        extra_args=config.sdk_args(),
        strict_mcp_config=True,
        skills=SKILLS,
        env=config.SDK_ENV,
        max_turns=max_turns,
        # Caps the optimizer's own spend. The target's API calls go through the
        # target's key and are not counted here.
        max_budget_usd=max_usd,
        permission_mode="default",
    )


async def run(project, out_dir, run_cmd=None, auto=False, model=None, max_turns=60, max_usd=None, effort=None, evals=None, review=False):
    project = pathlib.Path(project).resolve()
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    started = datetime.datetime.now()
    run_dir = out / "runs" / f"{started:%Y%m%d-%H%M%S}-{project.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_sha = _git(project, "rev-parse", "HEAD")
    tools.CTX.update({"project": project, "out": out, "run_cmd": run_cmd, "run_locked": bool(run_cmd),
                      "events": [], "review": review, "model": model, "run_dir": run_dir})
    tools.CTX.pop("baseline_state", None)
    mission = MISSION
    if evals:
        mission += (f"\n\nEval cases were supplied at `{evals}`. Call load_eval_cases with that "
                    "path before measuring.")
    if run_cmd:
        mission += (f"\n\nThe run command is already set: `{run_cmd}`. Do not rediscover or "
                    "change it - start with measure.")
    if review:
        mission += REVIEW_MISSION

    options = build_options(project, run_cmd, auto, model, max_turns, max_usd, effort)

    async def prompt():
        yield {"type": "user", "message": {"role": "user", "content": mission}}

    texts, calls, skills, result = [], [], [], {}
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt())
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text)
                            texts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            name = block.name.replace("mcp__fleetopt__", "")
                            print(f"  - {name}")
                            calls.append(name)
                            if block.name == "Skill":
                                skills.append((block.input or {}).get("skill", "?"))
                elif isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", None)
                    result = {"turns": message.num_turns, "optimizer_cost_usd": cost,
                              "status": message.subtype,
                              "error": message.result if message.is_error else None}
                    print(f"\n--- done in {message.num_turns} turns" +
                          (f", ${cost:.4f}" if cost else "") + " ---")
    finally:
        computed = verdict(tools.CTX.get("events", []))
        print(f"\n--- fleetopt verdict (computed from the measurements, not written by the agent) ---\n{computed}")
        meta = {"model": model, "auto": auto, "evals_path": evals, "max_turns": max_turns, "max_usd": max_usd,
                "review": review, "verdict": computed}
        try:
            _write_record(run_dir, project, start_sha, started, meta, texts, calls, skills, result)
        except OSError as exc:  # never let the record mask what the run itself did
            print(f"[fleetopt] could not write the run record: {exc}")
    return 0
