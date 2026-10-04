"""The one session behind `fleetopt review` and `fleetopt apply`, and the boundary around it.

One Claude Agent SDK session does the whole job, as an expert would: it works out how to
start the project's agent, measures it, finds what its expertise looks for, changes the
code, proves each change, looks again and reports. Which expert it is (experts/) decides
its guide, its skills and what a change must earn; `review` is the same session without the
tools that change anything. What it may not decide is enforced, not asked: the tools
(tools.py) hold the numbers, the limits and git; the hooks here keep edits inside the
project on fleetopt's branch, and nothing installed, pushed or run by hand. What is kept is
decided by `keep`, on the measurements and on the team's own evals, never on what the
session wrote.
"""

import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from fleetopt import config, tools
from fleetopt.evidence import measure as measure_mod
from fleetopt import expert as experts
from fleetopt.expert import COST, EXPERTS
from fleetopt.probe import store

MODEL, FALLBACK = "sonnet", "opus"  # aliases: whatever this Claude Code setup provides; FLEETOPT_MODEL overrides

# Running the agent by hand spends the team's tokens twice and records nothing; eval
# runners count too, since they run the agent on every case and bill its graders. Python
# runs only as a compile check: observed, `python -c "from agents... import ..."` to size
# a prompt, which is project code outside the cap (the recorded prompts have the sizes).
RUNS_TARGET = re.compile(
    r"\bpytest\b|\blanggraph\s+dev\b|\bdeepeval\s+test\b|\bpromptfoo\s+eval\b|\bbraintrust\s+eval\b"
    r"|\b(?:uv|poetry|pdm)\s+run\b|\bpython[\d.]*\b(?!\s+-m\s+py_compile\b)|driver\.py"
    r"|(?:^|[\s;&|(])py(?:\.exe)?\s+(?!-m\s+py_compile\b)")  # Windows' py launcher
# Installs and downloads: right for a sandbox, wrong on someone else's machine (observed:
# a session pulled langgraph from the internet with `uv run --with`).
ENV_MUTATION = re.compile(
    r"\b(?:pip3?\s+(?:install|uninstall)|python[\d.]*\s+-m\s+pip\s+(?:install|uninstall)"
    r"|uv\s+(?:pip|add|sync|tool|run\s+--with)|pipx\s+(?:install|run)"
    r"|poetry\s+(?:add|install|update)|pdm\s+(?:add|install)|(?:conda|mamba)\s+(?:install|create)"
    r"|(?:npm|pnpm|yarn)\s+(?:install|add|i)\b|npx\s|apt(?:-get)?\s+install|(?:dnf|yum)\s+install"
    r"|brew\s+install|curl\s|wget\s)")
PUBLISH = re.compile(r"\bgit\s+push\b")
# git is fleetopt's: observed, a session got a refused change in through git plumbing,
# and another ran `git checkout` to fake a clean baseline.
GIT_WRITE = re.compile(
    r"\bgit\s+(?:-C\s+\S+\s+)?(?:commit|checkout|switch|reset|restore|revert|branch|merge|rebase|stash|"
    r"cherry-pick|tag|clean|am|apply|update-index|update-ref|hash-object|read-tree|write-tree|"
    r"commit-tree|mv|rm|add)\b")


def _deny(reason):
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny", "permissionDecisionReason": reason}}


def _git(project, *args):
    """Read-only git query; '' when there is no repo or no git."""
    try:
        return subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def guard_edit(project, start_branch):
    """Edit and Write: inside the project, on fleetopt's branch, after the agent as it is was measured."""
    root = pathlib.Path(project).resolve()

    async def hook(input_data, tool_use_id, context):
        raw = (input_data.get("tool_input") or {}).get("file_path", "")
        target = (pathlib.Path(raw) if pathlib.Path(raw).is_absolute() else root / raw).resolve()
        if not target.is_relative_to(root) or ".git" in target.relative_to(root).parts:
            return _deny(f"fleetopt changes files only inside the project ({root}).")
        if _git(root, "rev-parse", "--abbrev-ref", "HEAD") == start_branch:
            return _deny(f"This is {start_branch!r}, the branch the run started from; changes go only on "
                         "fleetopt's branch. Stop and say so in your report.")
        if "baseline" in tools.CTX and tools.CTX["baseline"] is None:
            return _deny("Measure the agent as it is first. Edits are allowed after that.")
        return {}

    return hook


def guard_bash():
    """Bash: no push, no git that writes, no installs or downloads, never the agent by hand."""
    async def hook(input_data, tool_use_id, context):
        cmd = (input_data.get("tool_input") or {}).get("command", "")
        if PUBLISH.search(cmd):
            return _deny("fleetopt never pushes. Changes stay on a local branch for the team.")
        if GIT_WRITE.search(cmd):
            return _deny("fleetopt commits and undoes itself (save_change, keep, undo). Leave git alone.")
        if ENV_MUTATION.search(cmd):
            return _deny("fleetopt never installs or downloads anything on this machine. If something is "
                         "missing, that is the team's to provide: report it.")
        if RUNS_TARGET.search(cmd):
            return _deny("The project's code runs only through start and measure; Python here only as "
                         "`python -m py_compile <file>`. Prompt sizes and contents are in what was recorded: query it.")
        return {}
    return hook


def build_options(project, model=None, max_usd=None, start_branch=None, look_only=False, max_turns=150, expert=COST):
    """Everything the session is allowed to be. `look_only`: no Bash, Edit, Write, save_change,
    keep or undo at all."""
    reads = ["Read", "Grep", "Glob"]
    builtins = reads if look_only else reads + ["Bash", "Edit", "Write"]
    hooks = None if look_only else {"PreToolUse": [
        HookMatcher(matcher="Bash", hooks=[guard_bash()]),
        HookMatcher(matcher="Edit|Write", hooks=[guard_edit(
            project, start_branch or _git(project, "rev-parse", "--abbrev-ref", "HEAD"))]),
    ]}
    return ClaudeAgentOptions(
        cwd=str(project), model=model or MODEL, fallback_model=FALLBACK if (model or MODEL) != FALLBACK else MODEL,
        system_prompt=expert.system(),
        mcp_servers={"fleetopt": tools.server(look_only, expert)},
        # Built-ins by allowlist: no web, no scheduler, no subagents. Nothing is asked;
        # what keeps each tool safe is enforced by the hooks and inside fleetopt's tools.
        tools=builtins, allowed_tools=[*tools.names(look_only, expert), *builtins], hooks=hooks,
        # Headless, and never a secret read (config.DENY_READS).
        disallowed_tools=["AskUserQuestion", *config.DENY_READS],
        # Authenticates like Claude Code on this machine, and inherits nothing else from
        # it or from the project's .claude/ (config.py).
        setting_sources=config.SETTING_SOURCES, extra_args=config.sdk_args(), strict_mcp_config=True, skills=[],
        env=config.SDK_ENV, max_turns=max_turns, max_budget_usd=max_usd, permission_mode="default",
    )


# --- one run ------------------------------------------------------------------------------

def _activity(block, project, shown):
    """What the agent is doing, in words, once per thing: observed, five minutes of nothing but
    'still working' while it read the code and the recorded calls."""
    name, args = block.name.removeprefix("mcp__fleetopt__"), block.input or {}
    where = lambda p: str(pathlib.Path(p).resolve().relative_to(project)) if pathlib.Path(p).resolve().is_relative_to(
        project) else pathlib.Path(p).name
    line = {"query": "looking at the recorded calls", "Grep": "searching the code", "Glob": "searching the code",
            "Read": args.get("file_path") and f"reading {where(args['file_path'])}",
            "Edit": args.get("file_path") and f"editing {where(args['file_path'])}",
            "Write": args.get("file_path") and f"editing {where(args['file_path'])}"}.get(name)
    if line and line != shown.get("last") and (line not in shown or not line.startswith("reading")):
        tools.say(f"  {line}")
        shown[line] = shown["last"] = line


def _put_back(start_branch, branch):
    """After a stopped run: nothing unproven left, the team's copy back on its own branch,
    and fleetopt's branch kept only if it holds something kept."""
    tools.finish()
    if branch:
        kept = int(tools._git("rev-list", "--count", f"{tools.CTX['start_sha']}..HEAD") or 0)
        tools._git("checkout", "-q", start_branch)
        if not kept:
            tools._git("branch", "-q", "-D", branch)
        print(f"\n  stopped: your copy is back on {start_branch}" + (f"; what was kept is on {branch}" if kept else
                                                                   ", nothing was changed"), flush=True)


def _saved_entry(path):
    """How to start this agent, if a try proved it before and its interpreter is still there."""
    if not path.exists():
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))
    return entry if entry.get("proven") and pathlib.Path(entry.get("interpreter", "")).exists() else None


def _review_path(out, project, state, expert):
    """An expert's review of one state of the code: what its own `apply` starts from."""
    return out / "reviews" / f"{project.name}-{hashlib.sha1(str(project).encode()).hexdigest()[:8]}-{state}-{expert.name}.md"


def _told(path):
    """What the team answered when asked, on this run or an earlier one."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _newest(out):
    if not (out / "fleetopt.db").exists():
        return 0
    with store.connect(out / "fleetopt.db") as conn:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM sessions").fetchone()[0]


def _prompt(expert, look_only, entry, evals, graph, team, minutes, max_usd, branch, earlier, told=()):
    ctx = tools.CTX
    lines = [expert.review if look_only else expert.apply, ""]
    if entry:
        lines.append(f"How to start it is known, from an earlier try: {entry['name']} ({entry['graph']}), "
                     f"{len(entry['inputs'])} input(s) {tools.source(entry)}. It is started: measure it.")
    else:
        python, (files, shell) = tools.interpreter(ctx["project"]), tools.env_names(ctx["project"])
        listed = "; ".join(f"{f}: {', '.join(n[:40]) or 'nothing'}" for f, n in list(files.items())[:20]) or "none"
        lines.append(f"How to start it is not known yet: start it (see the guide). Env files in the project and the "
                     f"names each sets (never the values): {listed}. Credential names set in this terminal: "
                     f"{', '.join(shell[:40]) or 'none'}. "
                     + (f"The project's interpreter: {python}." if python != sys.executable else
                        "No .venv or venv in the project: if its environment is elsewhere (poetry, conda, a path in "
                        "its README), name that python as `interpreter`."))
        if graph:
            lines.append(f"The person running fleetopt asked for this agent: {graph}.")
        for item in told:
            lines.append(f"The team was asked before: {item['question']} They answered: {item['answer']}")
    if evals:
        lines.append(f"The person running fleetopt says to check the agent with: {evals} (an eval command, or a file "
                     "of test cases, expected answers or example requests). Use it.")
    lines.append(f"Limits, held by the tools: ${team:.2f} spent by the agent on its own API key, {minutes:g} minutes, "
                 f"${max_usd:.2f} for you.")
    if branch:
        lines.append(f"Changes go on fleetopt's branch {branch}.")
    if earlier:
        lines += ["", earlier]
    return "\n".join(lines)


async def run(project, out, *, look_only=False, evals=None, graph=None, model=None, max_usd=5.0, expert="cost",
              ask=False):
    """One run of one expert. Returns the facts the summary is computed from. Raises
    RuntimeError when it cannot begin (the project is not under git), ValueError when the
    expert cannot do what was asked. `ask`: someone is at the terminal to answer a question."""
    from claude_agent_sdk import (AssistantMessage, ClaudeSDKError, ResultMessage, SystemMessage, TextBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage, query)

    from fleetopt.progress import ticking

    expert = experts.get(expert)
    if not look_only and not expert.apply:
        raise ValueError(f"the {expert.name} expert only reviews: fleetopt review --expert {expert.name}")
    project, out = pathlib.Path(project).resolve(), pathlib.Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    began = datetime.datetime.now()
    run_dir = out / "runs" / f"{began:%Y%m%d-%H%M%S}-{project.name}"
    path = tools.entry_path(out, project, graph or "agent")
    entry = _saved_entry(path)
    team = float(os.environ.get("FLEETOPT_TEAM_USD") or tools.TEAM_USD)
    minutes = float(os.environ.get("FLEETOPT_MAX_MINUTES") or tools.MAX_MINUTES)
    first = _newest(out)
    tools.begin(project, out, entry_file=path, entry=entry, run_dir=run_dir, look_only=look_only, team_usd=team,
                minutes=minutes, first_session=first, expert=expert, ask=ask)
    ctx = tools.CTX
    start_branch = _git(project, "rev-parse", "--abbrev-ref", "HEAD")
    branch = None if look_only else f"fleetopt/{began:%Y%m%d-%H%M%S}"
    if branch:
        tools._git("checkout", "-q", "-b", branch)
    earlier = None
    if not look_only:  # what every expert found on this exact code, its own first: one's findings reach the others
        found = [(e, _review_path(out, project, ctx["start_state"], e)) for e in (expert, *EXPERTS.values()) if e.name != expert.name or e is expert]
        texts = [f"The {e.name} expert reviewed this exact code earlier:\n{p.read_text(encoding='utf-8')}"
                 for e, p in dict.fromkeys(found) if p.exists()]
        earlier = "\n\n".join(texts) or None

    prompt = _prompt(expert, look_only, entry, evals, graph, team, minutes, max_usd, branch, earlier,
                     _told(tools.told_path(path)))
    options = build_options(project, model, max_usd, start_branch, look_only, expert=expert)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (run_dir / "log.txt").open("w", encoding="utf-8")

    def log(line):  # everything, as it happens: `tail -f` it to follow a run in full
        log_file.write(line + "\n")
        log_file.flush()

    tools.say(f"  full log, live: {_shown(run_dir / 'log.txt')}")
    account, own, shown, said, stopped = "", 0.0, {}, None, True
    try:
        async with ticking("working", said_at=lambda: ctx.get("said_at", 0)):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    tools.say(f"  model: {(message.data or {}).get('model') or options.model}")  # what this setup gave
                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text.strip():
                            log(block.text.strip())
                            said = block.text  # shown once a step follows, so the final report is not echoed
                        elif isinstance(block, ToolUseBlock):
                            if said:
                                tools.say(f"  › {_sentence(said)}")
                                said = None
                            log(f"> {block.name.removeprefix('mcp__fleetopt__')} {json.dumps(block.input)[:500]}")
                            _activity(block, project, shown)
                elif isinstance(message, UserMessage) and isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            text = block.content if isinstance(block.content, str) else " ".join(
                                c.get("text", "") for c in block.content or [] if isinstance(c, dict))
                            log("< " + (text or "")[:2000])
                elif isinstance(message, ResultMessage):
                    own = getattr(message, "total_cost_usd", None) or 0.0
                    log(message.result or "")
                    account = _report_only(message.result or "")
        stopped = False
    except ClaudeSDKError as exc:  # out of turns or budget: what the tools recorded still stands
        log(f"session ended: {str(exc).splitlines()[0][:200]}")
        tools.say(f"  the session ended early: {str(exc).splitlines()[0][:120]}")
        stopped = False
    finally:
        log_file.close()
        if stopped:  # Ctrl-C or a kill: observed, the team's copy left on fleetopt's branch
            _put_back(start_branch, branch)
    tools.finish()

    kept = int(tools._git("rev-list", "--count", f"{ctx['start_sha']}..HEAD") or 0) if branch else 0
    whole = tools.moved(tools._compare(ctx["base"], ctx["kept_label"])) if kept else None
    diff = tools._git("diff", ctx["start_sha"], "HEAD") if kept else ""
    if branch:  # the team's working copy goes back where it was; the branch holds what was kept
        tools._git("checkout", "-q", start_branch)
        if not kept:
            tools._git("branch", "-q", "-D", branch)
    if look_only and ctx.get("baseline") and account:
        review = _review_path(out, project, ctx["start_state"], expert)
        review.parent.mkdir(parents=True, exist_ok=True)
        review.write_text(account + "\n", encoding="utf-8")
    runs, team_cost = measure_mod.spent(out, project, first)
    facts = {
        "mode": "review" if look_only else "apply", "expert": expert.name, "expert_version": expert.version,
        "expert_from": str(expert.folder), "changes_it": bool(expert.apply),
        "project": str(project), "started": bool(ctx.get("run_cmd")),
        "agent": (ctx.get("entry") or {}).get("name"), "measured": bool(ctx.get("baseline")),
        "baseline": ctx.get("baseline"), "proof": tools.proof() if ctx.get("entry") else None, "reach": tools.reached(),
        "bill": ctx.get("bill") or [], "unaccounted": tools.unaccounted(account, ctx.get("bill") or []) if account else None,
        "checks": tools.check_facts(),
        "changes": [{"name": n, "outcome": o, "detail": d} for n, (o, d) in ctx["changes"].items()],
        "kept": kept, "whole": whole, "branch": branch if kept else None,
        "team_runs": runs, "team_cost": team_cost, "own_cost": own, "account": account, "run_dir": str(run_dir),
    }
    (run_dir / "report.md").write_text("\n\n".join(filter(None, [account, "\n".join(summary(facts))])) + "\n",
                                       encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({**facts, "events": ctx["events"]}, indent=1, default=str),
                                      encoding="utf-8")
    if diff:
        (run_dir / "patch.diff").write_text(diff + "\n", encoding="utf-8")
    return facts


def summary(facts):
    """The run in a few lines for a person, every one computed from what was recorded."""
    money = lambda usd: "not priced" if usd is None else f"${usd:.2f}"
    lines = []
    if facts["mode"] == "apply" and facts["measured"]:
        if facts["kept"]:
            lines.append(f"  Result   {facts['kept']} change(s) kept on branch {facts['branch']}: {facts['whole']}")
        else:
            lines.append("  Result   nothing kept: the agent is as it was")
        width = max((len(c["name"]) for c in facts["changes"]), default=0)
        for i, c in enumerate(facts["changes"]):
            lines.append(f"  {'Changes' if not i else '':<8} {c['name']:<{width}}  {c['outcome']}"
                         + (f": {c['detail']}" if c["detail"] else ""))
        lines.append(f"  Checked  {facts['proof']}")
    elif not facts["measured"]:
        lines.append("  Result   " + ("could not start it" if not facts["started"] else "could not measure it")
                     + ": see why above")
    if facts["measured"] and (facts.get("baseline") or {}).get("completed") == 0:
        lines.append("  Broken   the agent as it is finished none of its requests"
                     + (": what was kept makes a broken agent cheaper, not a working one" if facts.get("kept") else ""))
    if facts.get("reach"):  # a change in a node the requests never ran is proven by nothing
        lines.append(f"  Reached  {facts['reach']}, by the requests it was run on")
    if facts.get("bill") and facts.get("unaccounted") is not None:  # did the expert account for where the money goes
        big = [x["node"] for x in facts["bill"] if x["share"] >= tools.ACCOUNT_FLOOR and x["node"] != "(graph)"]
        missing = facts["unaccounted"]
        lines.append(f"  Account  {len(big) - len(missing)} of {len(big)} nodes above {tools.ACCOUNT_FLOOR:.0%} of the tokens are "
                     "in the report" + (f"; not accounted for: {', '.join(missing[:6])}" if missing else ""))
    lines.append(f"  Spent    {money(facts['team_cost'])} by the agent on its API key ({facts['team_runs']} runs) · "
                 f"{money(facts['own_cost'])} by fleetopt on your Claude login")
    checks = facts.get("checks") or {}
    if checks.get("total"):  # the expert's own list, made by code from the recordings, closed by it with its numbers
        lines.append(f"  Checks   {checks['closed']} of {checks['total']} closed (node x check)"
                     + (f"; open: {', '.join(checks['open'][:6])}" + (" ..." if len(checks["open"]) > 6 else "")
                        if checks["open"] else ""))
    if facts["mode"] == "review" and facts["measured"] and facts.get("changes_it", True):
        lines.append(f"  Next     fleetopt apply {facts['project']}"
                     + (f" --expert {facts['expert']}" if facts.get("expert", "cost") != "cost" else ""))
    lines.append(f"  Details  {_shown(pathlib.Path(facts['run_dir']) / 'report.md')}")
    return lines


def _shown(path):
    return os.path.relpath(path) if path.is_relative_to(pathlib.Path.cwd()) else str(path)


def _sentence(text):
    """The first sentence of what the agent said, for a line on screen."""
    first = re.split(r"(?<=[.!?:])\s|\n", text.strip(), maxsplit=1)[0].strip().lstrip("#*- ").rstrip(":")
    return first[:157] + "..." if len(first) > 160 else first


def _report_only(text):
    """The report from the agent's last message, without what it said before it (observed:
    a line of chatter, and a made-up explanation, ahead of the report)."""
    lines = text.strip().splitlines()
    start = next((i for i, line in enumerate(lines) if "What it is for" in line), 0)
    return "\n".join(lines[start:]).strip()
