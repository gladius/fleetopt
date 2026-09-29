"""The session behind `fleetopt apply`, and the boundary around it.

One agent drives, as a developer would: it reads the review and the source, decides
what to try, in what order, what to do when something fails, what else to look at, and
when to stop. What it may not decide is enforced, not asked: the tools (tools.py) run the
agent, measure, judge, keep and undo, and hold the limits on money, time and steps; the
hooks here keep edits inside the project on fleetopt's branch, git fleetopt's alone,
nothing installed or pushed. The verdict is computed from what the tools recorded, never
from what the session wrote.

For a day (29 Sep 2026) a fixed procedure in code replaced the agent (loop.py). It was
safe and could not adapt: no second look after a fix, no judgement about which finding
the code already contradicts. The limits it enforced now live in the tools.
"""

import datetime
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from fleetopt import config
from fleetopt.evidence import evals as evals_mod
from fleetopt.optimizer import tools
from fleetopt.probe import runner

_HERE = pathlib.Path(__file__).parent
SKILL = "\n\n".join((_HERE / name).read_text(encoding="utf-8") for name in ("APPLY.md", "COST.md"))
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
            return _deny(f"This is {start_branch!r}, the branch the run started from. Changes go only on the "
                         "branch fleetopt made for this run; stop and say so in your report.")
        if "baseline" in tools.CTX and tools.CTX["baseline"] is None:
            return _deny("Measure the agent as it is first: call measure with no finding. Edits are allowed after that.")
        rel = str(target.relative_to(root))
        if "editing" in tools.CTX and rel not in tools.CTX["editing"]:
            tools.CTX["editing"].add(rel)
            tools.say(f"[fleetopt] changing {rel}")
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


def build_options(project, run_cmd=None, model=None, max_turns=100, max_usd=None, start_branch=None):
    """Everything the apply session is allowed to be. Apart from run() so the product's
    promises can be read off it in a test without starting a session
    (tests/test_invariants.py). `start_branch` is the branch the run began on: edits
    are refused there, and allowed on the branch fleetopt made."""
    return ClaudeAgentOptions(
        cwd=str(project),
        model=model,
        system_prompt=(
            "You make another team's LangGraph agent cheaper, for a central AI team, and prove "
            "each change. fleetopt runs the agent, measures it, judges its answers and owns git: "
            "you change the source and call its tools. If an edit, a command or a tool call is "
            "refused, that is an answer, not an obstacle: never look for another way to make the "
            "same change (another tool, a shell write, git plumbing). Say so in your report "
            "instead.\n\n" + SKILL
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
        # key and are capped by the tools.
        max_budget_usd=max_usd,
        permission_mode="default",
    )


MISSION = """Make the LangGraph agent in this project cost less to run without changing what it
produces, starting from the review below.

The findings to try: {ids}. {scope}
{design}
Eval cases: {cases}
Limits, held by the tools: ${team:.2f} on the team's key for runs of the agent, {minutes:g}
minutes in all, ${own:.2f} for you. You are on fleetopt's branch {branch}.

--- the review ---

{review}"""

SCOPE = {True: "A person chose exactly these: try nothing else, and name anything else you notice in your report.",
         False: "When they are done, look again at the traces of the code as it now stands: a fix often "
                "uncovers the next cost. Give anything new worth trying the next free id (N1, N2, ...)."}


def _design_line(design, cases):
    if not design:
        return "Cost only: a change that removes, merges or rewires the graph's nodes or edges cannot be kept."
    if not cases:
        return ("Design was asked for, but the team has no eval cases, so a change to the graph's structure "
                "cannot be kept: describe it in your report instead.")
    return ("Design changes may be tried. A change to the graph's structure is kept only when the team's "
            "cases cover every request and it passes them.")


def _report(findings, rows, total):
    """The measured table: one row per finding, from what the tools recorded."""
    order = [f["id"] for f in findings] + [k for k in rows if k not in {f["id"] for f in findings}]
    titles = {f["id"]: f["title"] for f in findings}
    lines = ["| Finding | Outcome | What was measured |", "|---|---|---|"]
    for key in order:
        outcome, detail = rows.get(key, ["not tried", ""])
        lines.append(f"| {key} {titles.get(key, '')} | {outcome} | {detail} |".replace("  |", " |"))
    return "\n".join(["## What was measured", "", *lines, "", total])


async def run(project, out_dir, run_cmd, review, findings, *, task, model=None, max_usd=5.0, evals=None,
              first_session=0, only=False, design=False):
    """The apply run: a new branch, one session that drives, then the record. Returns
    the facts the summary is computed from."""
    from claude_agent_sdk import AssistantMessage, ClaudeSDKError, ResultMessage, TextBlock, ToolUseBlock, query

    from fleetopt.progress import ticking

    project, out = pathlib.Path(project).resolve(), pathlib.Path(out_dir).resolve()
    started = datetime.datetime.now()
    run_dir = out / "runs" / f"{started:%Y%m%d-%H%M%S}-{project.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cases, _ = evals_mod.load(pathlib.Path(evals).resolve() if evals else project)
    team = float(os.environ.get("FLEETOPT_TEAM_USD") or tools.TEAM_USD)
    minutes = float(os.environ.get("FLEETOPT_MAX_MINUTES") or tools.MAX_MINUTES)
    start_branch = _git(project, "rev-parse", "--abbrev-ref", "HEAD")
    tools.start(project, out, run_cmd, findings=findings, task=task, cases=cases, only=only, design=design,
                team_usd=team, minutes=minutes, first_session=first_session, run_dir=run_dir)
    start_sha, start_state = tools.CTX["start_sha"], tools.CTX["start_state"]
    branch = f"fleetopt/{started:%Y%m%d-%H%M%S}"
    tools._git("checkout", "-q", "-b", branch)

    prompt = MISSION.format(
        ids=", ".join(f["id"] for f in findings), scope=SCOPE[bool(only)], design=_design_line(design, cases),
        cases=(f"{len(cases)} loaded; where a request matches one, the judge grades the answer against it"
               if cases else "none; the judge compares each answer with the original's"),
        team=team, minutes=minutes, own=max_usd, branch=branch, review=review)
    options = build_options(project, run_cmd, model, max_usd=max_usd, start_branch=start_branch)
    log, final, own = [], "", 0.0
    try:
        async with ticking("working on the agent", said_at=lambda: tools.CTX.get("said_at", 0)):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            log.append(block.text.strip())
                        elif isinstance(block, ToolUseBlock):
                            log.append(f"> {block.name.removeprefix('mcp__fleetopt__')} {json.dumps(block.input)[:300]}")
                elif isinstance(message, ResultMessage):
                    own = getattr(message, "total_cost_usd", None) or 0.0
                    final = (message.result or "").strip()
    except ClaudeSDKError as exc:
        # Out of turns or budget arrives as an exception. What the tools recorded still stands.
        why = str(exc).splitlines()[0][:200]
        log.append(f"session ended: {why}")
        tools.say(f"[fleetopt] the session ended early: {why}")
    tools.finish()

    kept = int(tools._git("rev-list", "--count", f"{start_sha}..HEAD") or 0)
    total = "Nothing was kept. The branch holds the code it started from."
    if kept:
        total = tools.compared("All kept changes", tools._compare("baseline", tools.CTX["kept_label"]))
    final_state = runner.code_state(project)
    events = tools.CTX["events"]
    computed = verdict(events, final_state, start_state)
    tried, gained = numbers(events, computed, start_state, final_state)
    table = _report(findings, tools.CTX["rows"], total.removeprefix("[fleetopt] "))
    if final:
        print("\n--- fleetopt's account ---\n" + final)
    print("\n--- " + table.removeprefix("## ").replace("\n", " ---\n", 1))
    print(f"\n--- fleetopt verdict (computed from the measurements) ---\n{computed}")
    facts = {"verdict": computed, "tried": tried, "kept": kept, "gained": gained, "branch": branch,
             "own_cost_usd": own, "run_dir": str(run_dir)}
    meta = {"model": model, "evals_path": evals, "max_usd": max_usd, "findings": findings,
            "code_state_after": final_state, "start_branch": start_branch, **facts,
            "rows": [{"id": k, "outcome": o, "detail": d} for k, (o, d) in tools.CTX["rows"].items()]}
    try:
        _write_record(run_dir, project, start_sha, started, meta,
                      [t for t in (final, table, f"Verdict: {computed}") if t], [], [], {}, log)
    except OSError as exc:
        print(f"[fleetopt] could not write the run record: {exc}")
    return facts
