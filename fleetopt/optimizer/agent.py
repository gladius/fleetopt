"""The one agent behind `fleetopt review` and `fleetopt apply`, and the boundary around it.

One Claude Agent SDK session does the whole job, as an expert would: it works out how to
start the project's agent, measures it, finds the waste, changes the code, proves each
change, looks again and reports. `review` is the same agent without the tools that
change anything. What it may not decide is enforced, not asked: the tools (tools.py)
hold the numbers, the limits and git; the hooks here keep edits inside the project on
fleetopt's branch, and nothing installed, pushed or run by hand. The verdict is computed
from what the tools recorded, never taken from what the session wrote.
"""

import datetime
import hashlib
import json
import os
import pathlib
import re
import subprocess

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

from fleetopt import config
from fleetopt.evidence import measure as measure_mod
from fleetopt.optimizer import tools
from fleetopt.probe import runner, store

_HERE = pathlib.Path(__file__).parent
SKILLS = ("caching", "model-tier", "prompt-growth", "redundant-work", "tool-surface")


def _body(path):
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip() if text.startswith("---") else text


# The guide, then the mechanics of each pattern it names: all in the prompt, cached after
# the first turn, so the agent never works without the one it needs.
SYSTEM = ((_HERE / "GUIDE.md").read_text(encoding="utf-8") + "\n\n## The mechanics each pattern refers to\n\n"
          + "\n\n".join(f"<!-- fleetopt:{n} -->\n" + _body(_HERE / "skills" / n / "SKILL.md") for n in SKILLS))

# Running the agent by hand spends the team's tokens twice and records nothing; eval
# runners count too, since they run the agent on every case and bill its graders. Python
# runs only as a compile check: observed, `python -c "from agents... import ..."` to size
# a prompt, which is project code outside the cap (the recorded prompts have the sizes).
RUNS_TARGET = re.compile(
    r"\bpytest\b|\blanggraph\s+dev\b|\bdeepeval\s+test\b|\bpromptfoo\s+eval\b|\bbraintrust\s+eval\b"
    r"|\b(?:uv|poetry|pdm)\s+run\b|\bpython[\d.]*\b(?!\s+-m\s+py_compile\b)|driver\.py")
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


def build_options(project, model=None, max_usd=None, start_branch=None, look_only=False, max_turns=150):
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
        cwd=str(project), model=model, system_prompt=SYSTEM,
        mcp_servers={"fleetopt": tools.server(look_only)},
        # Built-ins by allowlist: no web, no scheduler, no subagents. Nothing is asked;
        # what keeps each tool safe is enforced by the hooks and inside fleetopt's tools.
        tools=builtins, allowed_tools=[*tools.names(look_only), *builtins], hooks=hooks,
        # Headless, and never a secret read (config.DENY_READS).
        disallowed_tools=["AskUserQuestion", *config.DENY_READS],
        # Authenticates like Claude Code on this machine, and inherits nothing else from
        # it or from the project's .claude/ (config.py).
        setting_sources=config.SETTING_SOURCES, extra_args=config.sdk_args(), strict_mcp_config=True, skills=[],
        env=config.SDK_ENV, max_turns=max_turns, max_budget_usd=max_usd, permission_mode="default",
    )


# --- the verdict: computed from what the tools recorded -----------------------------------

def verdict(events, final=None, start=None):
    """What the measurements support. Only a judged comparison of two different code states
    counts, and every judgment of the code left on the branch counts: one failure is a
    failure. A change that failed and was undone does not condemn the ones kept."""
    real = [e for e in events if e["event"] == "judge" and e.get("baseline_state") != e.get("candidate_state")]
    if not real:
        return "NOTHING PROVEN: no change was judged against the code it started from."
    if final and final == start:
        return (f"NOTHING LEFT STANDING: {len({e['candidate_state'] for e in real})} changed version(s) were "
                "judged and undone. The code is as it started.")
    final = final or real[-1]["candidate_state"]
    judged = [e for e in real if e["candidate_state"] == final]
    if not judged:
        return (f"NOT PROVEN: the code as it was left ({final}) was never judged. The last code judged was "
                f"{real[-1]['candidate_state']}.")

    def detail(e):
        ok = sum(bool(r["equivalent"]) for r in e["equivalence"])
        text = f"{e['candidate']}: answers unchanged {ok}/{len(e['equivalence'])}"
        c = e.get("correctness")
        if c:
            return text + (f", correct on the team's cases {c['baseline_pass']}/{c['matched']} before and "
                           f"{c['candidate_pass']}/{c['matched']} after")
        return text + ", correctness not checked (no eval cases)"

    details = "; ".join(detail(e) for e in judged)
    if not all(e["passed"] for e in judged):
        return f"NOT PROVEN SAFE: the judge failed ({details})."
    return f"PROVEN ON THIS EVIDENCE: the judge passed ({details})."


# --- one run ------------------------------------------------------------------------------

MISSION = {
    True: "Review the LangGraph agent in this project: start it if needed, measure it once as it is, find where "
          "it wastes tokens and money, and report what is worth changing. This run changes nothing.",
    False: "Make the LangGraph agent in this project cost less without changing what it answers: start it if "
           "needed, measure it, find the waste, change it, prove each change, look again, and report.",
}


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


def _review_path(out, project, state):
    return out / "reviews" / f"{project.name}-{hashlib.sha1(str(project).encode()).hexdigest()[:8]}-{state}.md"


def _newest(out):
    if not (out / "fleetopt.db").exists():
        return 0
    with store.connect(out / "fleetopt.db") as conn:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM sessions").fetchone()[0]


def _prompt(look_only, entry, inputs, cases, graph, team, minutes, max_usd, branch, earlier):
    ctx = tools.CTX
    lines = [MISSION[look_only], ""]
    if entry:
        lines.append(f"How to start it is known, from an earlier try: {entry['name']} ({entry['graph']}), "
                     f"{len(entry['inputs'])} inputs from {entry['inputs_from']}. It is started: measure it.")
    else:
        lines.append(f"How to start it is not known yet: start it (see the guide). Provider keys that are set "
                     f"(names only): {', '.join(tools.key_names(ctx['project'])) or 'none found'}. The project's "
                     f"interpreter: {tools.interpreter(ctx['project'])}.")
        if graph:
            lines.append(f"The person running fleetopt asked for this agent: {graph}.")
        lines.append(f"Inputs: fleetopt runs it on the team's eval cases ({len(inputs)}); leave `inputs` out."
                     if inputs else "Inputs: give 4 that differ in kind (at least 3).")
    lines.append(f"Eval cases: {len(cases)}; the judge checks each answer against its case where one matches."
                 if cases else "Eval cases: none; the judge compares each answer with the original's.")
    lines.append(f"Limits, held by the tools: ${team:.2f} on the team's key, {minutes:g} minutes, ${max_usd:.2f} "
                 "for you.")
    if branch:
        lines.append(f"Changes go on fleetopt's branch {branch}.")
    if earlier:
        lines += ["", "An earlier review of this exact code:", earlier]
    return "\n".join(lines)


async def run(project, out, *, look_only=False, evals=None, graph=None, model=None, max_usd=5.0):
    """One run of the agent. Returns the facts the summary is computed from. Raises
    ValueError or RuntimeError when it cannot begin (no git, unreadable eval cases)."""
    from claude_agent_sdk import (AssistantMessage, ClaudeSDKError, ResultMessage, TextBlock, ToolResultBlock,
                                  ToolUseBlock, UserMessage, query)

    from fleetopt.progress import ticking

    project, out = pathlib.Path(project).resolve(), pathlib.Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    began = datetime.datetime.now()
    run_dir = out / "runs" / f"{began:%Y%m%d-%H%M%S}-{project.name}"
    cases, inputs, inputs_from = tools.team_inputs(project, evals)
    path = tools.entry_path(out, project, graph or "agent")
    entry = _saved_entry(path)
    if entry and not inputs and len(entry["inputs"]) < tools.MIN_INPUTS:
        entry = None  # observed: an entry from before the rule, 1 input: too few to see past the noise
    if entry and inputs and entry["inputs"] != inputs:  # the same agent, asked the team's cases now
        entry.update(inputs=inputs, inputs_from=inputs_from)
        path.write_text(json.dumps(entry, indent=1), encoding="utf-8")
    team = float(os.environ.get("FLEETOPT_TEAM_USD") or tools.TEAM_USD)
    minutes = float(os.environ.get("FLEETOPT_MAX_MINUTES") or tools.MAX_MINUTES)
    first = _newest(out)
    tools.begin(project, out, entry_file=path, entry=entry, cases=cases, inputs=inputs, inputs_from=inputs_from,
                look_only=look_only, team_usd=team, minutes=minutes, first_session=first)
    ctx = tools.CTX
    start_branch = _git(project, "rev-parse", "--abbrev-ref", "HEAD")
    branch = None if look_only else f"fleetopt/{began:%Y%m%d-%H%M%S}"
    if branch:
        tools._git("checkout", "-q", "-b", branch)
    earlier = None
    if not look_only and _review_path(out, project, ctx["start_state"]).exists():
        earlier = _review_path(out, project, ctx["start_state"]).read_text(encoding="utf-8")

    prompt = _prompt(look_only, entry, inputs, cases, graph, team, minutes, max_usd, branch, earlier)
    options = build_options(project, model, max_usd, start_branch, look_only)
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
                if isinstance(message, AssistantMessage):
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
    final_state = runner.code_state(project)
    whole = tools.moved(tools._compare(ctx["base"], ctx["kept_label"])) if kept else None
    diff = tools._git("diff", ctx["start_sha"], "HEAD") if kept else ""
    if branch:  # the team's working copy goes back where it was; the branch holds what was kept
        tools._git("checkout", "-q", start_branch)
        if not kept:
            tools._git("branch", "-q", "-D", branch)
    if look_only and ctx.get("baseline") and account:
        review = _review_path(out, project, ctx["start_state"])
        review.parent.mkdir(parents=True, exist_ok=True)
        review.write_text(account + "\n", encoding="utf-8")
    runs, team_cost = measure_mod.spent(out, project, first)
    facts = {
        "mode": "review" if look_only else "apply", "project": str(project), "started": bool(ctx.get("run_cmd")),
        "agent": (ctx.get("entry") or {}).get("name"), "measured": bool(ctx.get("baseline")),
        "baseline": ctx.get("baseline"), "cases": len(cases),
        "changes": [{"name": n, "outcome": o, "detail": d} for n, (o, d) in ctx["changes"].items()],
        "kept": kept, "whole": whole, "branch": branch if kept else None,
        "verdict": verdict(ctx["events"], final_state, ctx["start_state"]) if branch else None,
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
        lines.append("  Checked  " + (f"answers against {facts['cases']} eval cases" if facts["cases"] else
                                      "answers against the agent's original answers (no eval cases in the project)"))
        if facts["kept"] and facts["verdict"]:
            lines.append(f"  Verdict  {facts['verdict'].split(':')[0].lower()}")
    elif not facts["measured"]:
        lines.append("  Result   " + ("could not start it" if not facts["started"] else "could not measure it")
                     + ": see why above")
    lines.append(f"  Spent    {money(facts['team_cost'])} on the team's key ({facts['team_runs']} runs) · "
                 f"{money(facts['own_cost'])} by fleetopt")
    if facts["mode"] == "review" and facts["measured"]:
        lines.append(f"  Next     fleetopt apply {facts['project']}")
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
