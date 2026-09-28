"""The session behind `fleetopt apply`.

It is handed the findings of a review a person has seen, tries them one at a time on
a new branch, and proves whatever it claims using the tools in tools.py. What this
module owns is the boundary: where the agent runs, what it may touch, and the
verdict, which is computed from what the tools recorded and not from what the agent
wrote.

Nothing is asked. What keeps a run safe is enforced: the target is started only by
fleetopt's own driver, an edit lands inside the project on a new branch or not at
all, and nothing is installed or pushed.
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
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from fleetopt import config
from fleetopt.optimizer import tools
from fleetopt.probe import runner

_HERE = pathlib.Path(__file__).parent
SKILL = "\n\n".join((_HERE / name).read_text(encoding="utf-8") for name in ("SKILL.md", "COST.md"))
PLUGIN = _HERE / "plugin"  # decision skills, loaded by the harness, triggered by description
# Only fleetopt's skills are listed to the model; the CLI's built-in ones are noise here.
SKILLS = sorted(f"fleetopt:{p.name}" for p in (PLUGIN / "skills").iterdir() if p.is_dir())

MISSION = """Apply the findings of a review to the LangGraph agent in this project, and
prove each one.

The review below was made on this exact code. It saw one capture and measured nothing,
so it is where you start, not the last word. Try these findings first, in this order:
{ids}.

1. Understand it. Read the review, then the source it points to.
2. Establish a baseline. fleetopt already knows how to start this agent and which
   inputs to give it, so call measure with the label "baseline". Before that, look for
   eval cases (see fleetopt:evals) and load them with load_eval_cases.
3. Create one git branch for this run. Edits are refused until you are on a new branch.
4. For each finding, in the order given:
   - make that one change and commit it, the finding's id first in the message
     ("C1: cache the system prompt");
   - measure under a label that starts with the id, compare it with the code just
     before it to see what this finding did on its own, and judge it against the
     baseline;
   - if the saving is within noise or the judge fails, undo the commit with
     git reset --hard HEAD~1 before you go on, and report the finding as not proven;
   - if the baseline contradicts the finding, skip it and give the number that does.
5. {after}
6. If more than one change is left standing, measure and judge the code as you leave
   it against the baseline once more. The team merges the branch, not single commits.

{design}

Report at the end, one row per finding you were given or found: its id, what happened (stands,
undone, skipped and why), the measured difference (dollars first, then latency, then
tokens), the equivalence verdict, and the correctness pass rate before and after if
eval cases were loaded (say "correctness not checked" if none were found). A cost
reduction that broke the agent is a regression, not a result.

--- the review ---

{review}"""

# The loop stays open: some waste only shows once the first fix is in, and the review
# saw one run where the baseline has three.
LOOK_AGAIN = """Then look again. Query the traces of the code as it now stands: a fix often
   uncovers the next cost. If something the review did not list is worth trying and
   keeps the design as it is, give it the next free id (N1, N2, ...) and treat it like
   any other finding. A change to the design that the review did not list is reported,
   never tried: nobody has seen the case for it. Stop when nothing is left that the
   numbers support."""

# With --only a person chose a list, and a list is a fence.
ONLY_THESE = """Stop there. A person chose exactly these findings. If you notice anything else,
   say so at the end and leave it alone."""

DESIGN_ON = """Findings marked "needs cases" or "human decides" change the design, not just its cost.
Try one only if the eval cases you loaded cover the path it touches; if they do not,
skip it and say which cases would unlock it. For a redesign, first list what must
survive (see fleetopt:patterns) and keep every item on it."""

DESIGN_OFF = "None of the findings you were given changes the design. Keep the design as it is."

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


PUBLISH = re.compile(r"\bgit\s+push\b")


def guard_edit(project, start_branch):
    """PreToolUse hook on Edit and Write. It replaces a yes/no question with a rule:
    a change lands inside the project, on a branch made for it, or not at all."""
    root = pathlib.Path(project).resolve()

    async def hook(input_data, tool_use_id, context):
        raw = (input_data.get("tool_input") or {}).get("file_path", "")
        target = pathlib.Path(raw) if pathlib.Path(raw).is_absolute() else root / raw
        target = target.resolve()
        if not target.is_relative_to(root) or ".git" in target.relative_to(root).parts:
            return _deny(f"fleetopt changes files only inside the project it was given ({root}).")
        if _git(root, "rev-parse", "--abbrev-ref", "HEAD") == start_branch:
            return _deny(f"You are still on {start_branch!r}, the branch this run started from. Create a "
                         "new branch first (git checkout -b ...), then make the change there.")
        return {}

    return hook


def guard_bash(run_cmd):
    """PreToolUse hook on Bash. Three rules, all enforced rather than asked for.

    Nothing is published: no git push. What fleetopt changes stays on a local branch
    for the team to read.

    The target runs only through `measure`: running it by hand spends the team's
    tokens twice, captures nothing, and fed 12K tokens of pytest tracebacks into
    the optimizer's context last time.

    The target's environment is not ours to change: no installs, no downloads. A
    missing dependency is a finding about the run command, not something to fix."""
    async def hook(input_data, tool_use_id, context):
        cmd = (input_data.get("tool_input") or {}).get("command", "")
        if PUBLISH.search(cmd):
            return _deny("fleetopt never pushes. The change stays on a local branch for the team to review.")
        if ENV_MUTATION.search(cmd):
            return _deny("fleetopt never installs packages or downloads anything into the "
                         "target's environment. Finding the interpreter that already has the "
                         "project's dependencies is fleetopt's job, not yours. If something is "
                         "missing, report it - that is the team's finding.")
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


def verdict(events, final=None, start=None):
    """What the measurements support, computed from what the tools recorded. Printed
    after the agent's report and stored in the run record, so a report cannot argue
    with a failed gate. Only a judged comparison of two different code states counts,
    and every judgment of the final code counts: one failure is a failure.

    `final` is the code as the run left it and `start` the code it began on. With
    them the verdict is about what is on the branch, not about the last thing that
    happened to be judged: a change that failed and was undone does not condemn the
    ones left standing, and code nobody judged is not proven by its neighbours."""
    real = [e for e in events if e["event"] == "judge" and e.get("baseline_state") != e.get("candidate_state")]
    if not real:
        return "NOTHING PROVEN: no change was judged against the code it started from."
    if final and final == start:
        return (f"NOTHING LEFT STANDING: {len({e['candidate_state'] for e in real})} changed version(s) were "
                "judged and undone. The branch holds the code it started from.")
    final = final or real[-1]["candidate_state"]
    judged = [e for e in real if e["candidate_state"] == final]
    if not judged:
        return (f"NOT PROVEN: the code as it was left ({final}) was never judged. The last code judged was "
                f"{real[-1]['candidate_state']}. The branch is left for review.")

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


def build_options(project, run_cmd=None, model=None, max_turns=120, max_usd=None):
    """Everything a session is allowed to be. Apart from run() so the product's
    promises can be read off it in a test without starting a session
    (tests/test_invariants.py)."""
    return ClaudeAgentOptions(
        cwd=str(project),
        model=model,
        system_prompt=(
            "You apply the findings of a review to LangGraph agents, run by a central "
            "AI team on another team's project. Evidence beats intuition: every claim "
            "you make must cite trace data or a measurement. Never report a saving "
            "you have not measured. If an edit or a command is refused, that is an answer, "
            "not an obstacle: never look for another way to make the same change (another "
            "tool, a shell write, git plumbing). Say what was refused and why, skip that "
            "finding, and go on.\n\n" + SKILL
        ),
        mcp_servers={"fleetopt": tools.server()},
        # Built-ins by allowlist. The CLI default is 26 tools including web fetch and
        # search, cron, worktrees, messaging and wake-up scheduling: egress and mutation
        # surfaces an optimizer has no business with, and schema tokens on every turn.
        # Observed before this: a run called ScheduleWakeup to "wait" for a subagent.
        tools=["Read", "Grep", "Glob", "Bash", "Edit", "Write", "Skill"],
        # No questions are asked, so none of these needs a person. What keeps them safe
        # is enforced below, not confirmed: the target is started only by fleetopt's own
        # driver, Bash cannot install, publish or run the target by hand, and an edit
        # lands inside the project on a new branch or not at all.
        allowed_tools=[*tools.TOOL_NAMES, "Read", "Grep", "Glob", "Skill", "Bash", "Edit", "Write"],
        plugins=[{"type": "local", "path": str(PLUGIN)}],
        hooks={"PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[guard_bash(run_cmd)]),
            HookMatcher(matcher="Edit|Write", hooks=[guard_edit(project, _git(project, "rev-parse", "--abbrev-ref", "HEAD"))]),
        ]},
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


async def run(project, out_dir, run_cmd, review, findings, model=None, max_turns=120, max_usd=None,
              evals=None, fenced=False):
    """Try `findings` (from review.findings, already chosen) of the `review` text, then
    keep looking unless `fenced`: a person who names findings gets those and no others."""
    project = pathlib.Path(project).resolve()
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    started = datetime.datetime.now()
    run_dir = out / "runs" / f"{started:%Y%m%d-%H%M%S}-{project.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_sha = _git(project, "rev-parse", "HEAD")
    start_state = runner.code_state(project)
    tools.CTX.update({"project": project, "out": out, "run_cmd": run_cmd,
                      "events": [], "model": model, "run_dir": run_dir})
    tools.CTX.pop("baseline_state", None)
    tools.CTX.pop("eval_cases", None)
    design = any(f["apply"] != "yes" for f in findings)
    mission = MISSION.format(ids=", ".join(f["id"] for f in findings) or "none: the review listed nothing to try",
                             review=review, after=ONLY_THESE if fenced else LOOK_AGAIN,
                             design=DESIGN_ON if design else DESIGN_OFF)
    if evals:
        mission += (f"\n\nEval cases were supplied at `{evals}`. Call load_eval_cases with that "
                    "path before measuring.")

    options = build_options(project, run_cmd, model, max_turns, max_usd)

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
        computed = verdict(tools.CTX.get("events", []), runner.code_state(project), start_state)
        print(f"\n--- fleetopt verdict (computed from the measurements, not written by the agent) ---\n{computed}")
        meta = {"model": model, "evals_path": evals, "max_turns": max_turns, "max_usd": max_usd,
                "findings": findings, "fenced": fenced, "code_state_after": runner.code_state(project), "verdict": computed}
        try:
            _write_record(run_dir, project, start_sha, started, meta, texts, calls, skills, result)
        except OSError as exc:  # never let the record mask what the run itself did
            print(f"[fleetopt] could not write the run record: {exc}")
    return 0
