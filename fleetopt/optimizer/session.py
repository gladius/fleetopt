"""The edit session, and the boundary around it.

`fleetopt apply` is driven by code (loop.py): it measures, compares, judges, keeps and
undoes. A model does one thing in it: make the change for one finding. This module
builds the session that makes it, and owns what that session may touch. It also
computes the verdict, from what the tools recorded and not from what any session wrote.

Nothing is asked. What keeps a run safe is enforced: the agent is started only by
fleetopt's own driver and never by this session, an edit lands inside the project on
fleetopt's branch or not at all, git is fleetopt's alone, and nothing is installed or pushed.
"""

import datetime
import importlib.metadata
import json
import pathlib
import re
import subprocess

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from fleetopt import config
from fleetopt.optimizer import tools

_HERE = pathlib.Path(__file__).parent
SKILL = "\n\n".join((_HERE / name).read_text(encoding="utf-8") for name in ("SKILL.md", "COST.md"))
PLUGIN = _HERE / "plugin"  # decision skills, loaded by the harness, triggered by description
# Only fleetopt's skills are listed to the model; the CLI's built-in ones are noise here.
SKILLS = sorted(f"fleetopt:{p.name}" for p in (PLUGIN / "skills").iterdir() if p.is_dir())

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
# git belongs to the loop: it commits a change, and undoes it when the numbers say so.
# Observed: a session whose edits were refused got the same change in through git
# plumbing; another ran `git checkout` to fake a clean baseline and had to repair it.
GIT_WRITE = re.compile(
    r"\bgit\s+(?:-C\s+\S+\s+)?(?:commit|checkout|switch|reset|restore|revert|branch|merge|rebase|stash|"
    r"cherry-pick|tag|clean|am|apply|update-index|update-ref|hash-object|read-tree|write-tree|"
    r"commit-tree|mv|rm|add)\b")


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
        if GIT_WRITE.search(cmd):
            return _deny("fleetopt commits, undoes and branches itself. Edit the files; leave git alone.")
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


def _write_record(run_dir, project, start_sha, started, meta, texts, calls, skills, result, log=()):
    """What this run did: report.md for a human, log.txt for whoever has to find out
    why, run.json for the ledger and for improving fleetopt, patch.diff when the branch
    changed anything. No prompts or outputs of the target are stored; the 120-char
    input excerpts in judge rows and the diff are the only target content, so sharing
    a run folder is the operator's call, not automatic."""
    (run_dir / "report.md").write_text("\n\n".join(texts), encoding="utf-8")
    (run_dir / "log.txt").write_text("\n".join(log or calls), encoding="utf-8")
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
        text = f"{e['candidate']}: answers unchanged {ok}/{len(e['equivalence'])}"
        c = e.get("correctness")
        if c:
            return text + (f", correct on the team's cases {c['baseline_pass']}/{c['matched']} before and "
                           f"{c['candidate_pass']}/{c['matched']} after")
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


NAMES = {"cost_usd": "cost", "wall_ms": "time", "llm_calls": "model calls", "completed": "finished requests"}


def numbers(events, computed, start, final):
    """The run in numbers, from what the tools recorded: how many changed versions were
    measured, and what the code left on the branch gained. A gain is stated only under
    a verdict that says it is proven; an unproven number is not a saving."""
    tried = len({e["code_state"] for e in events
                 if e["event"] == "measure" and e.get("code_state") not in (None, start)})
    gained = []
    if computed.startswith("PROVEN"):
        last = [e for e in events if e["event"] == "compare"
                and e.get("baseline_state") == start and e.get("candidate_state") == final]
        for key, name in NAMES.items():
            v = (last[-1]["result"] if last else {}).get(key) or {}
            if v.get("verdict") == "improved" and v.get("delta_pct") is not None:
                gained.append(f"{name} {v['delta_pct']:+.0f}%")
    return tried, gained


READS = ["query_traces", "graph_topology", "graph_shape"]  # the edit session looks; it never measures


def build_options(project, run_cmd=None, model=None, max_turns=40, max_usd=None, start_branch=None):
    """Everything the edit session is allowed to be. Apart from the loop so the
    product's promises can be read off it in a test without starting a session
    (tests/test_invariants.py). `start_branch` is the branch the run began on: edits
    are refused there, and allowed on the branch the loop made."""
    return ClaudeAgentOptions(
        cwd=str(project),
        model=model,
        system_prompt=(
            "You make one change to another team's LangGraph agent, for one finding of a "
            "review, for a central AI team. fleetopt measures it, judges it, and keeps or "
            "undoes it after you: you never run the agent and never touch git. If an edit "
            "or a command is refused, that is an answer, not an obstacle: never look for "
            "another way to make the same change (another tool, a shell write, git plumbing). "
            "Reply CANNOT with the reason instead.\n\n" + SKILL
        ),
        mcp_servers={"fleetopt": tools.server(READS)},
        # Built-ins by allowlist. The CLI default is 26 tools including web fetch and
        # search, cron, worktrees, messaging and wake-up scheduling: egress and mutation
        # surfaces an optimizer has no business with, and schema tokens on every turn.
        # Observed before this: a run called ScheduleWakeup to "wait" for a subagent.
        tools=["Read", "Grep", "Glob", "Bash", "Edit", "Write", "Skill"],
        # No questions are asked, so none of these needs a person. What keeps them safe
        # is enforced below, not confirmed: the target is started only by fleetopt's own
        # driver, Bash cannot install, publish or run the target by hand, and an edit
        # lands inside the project on a new branch or not at all.
        allowed_tools=[*(f"mcp__fleetopt__{n}" for n in READS), "Read", "Grep", "Glob", "Skill", "Bash", "Edit", "Write"],
        plugins=[{"type": "local", "path": str(PLUGIN)}],
        hooks={"PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[guard_bash(run_cmd)]),
            HookMatcher(matcher="Edit|Write", hooks=[guard_edit(
                project, start_branch or _git(project, "rev-parse", "--abbrev-ref", "HEAD"))]),
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
        # Caps this session's own spend. The agent's API calls go through the team's
        # key and are capped by the loop.
        max_budget_usd=max_usd,
        permission_mode="default",
    )
