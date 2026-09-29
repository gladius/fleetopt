"""What must stay true whatever else changes: one test per promise the product makes.

    pytest -q tests/test_invariants.py

An experiment that breaks one of these broke the product, however good its numbers.
"""

import asyncio
import pathlib
import sys

import pytest

from fleetopt import cli, config
from fleetopt.evidence import judge, measure
from fleetopt.optimizer import review, session, tools
from fleetopt.probe import store

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = (ROOT / "fleetopt" / "probe" / "hooks" / "_fleetopt_hook.py").read_text(encoding="utf-8")


def options(**kw):
    return session.build_options(ROOT / "fixture", **kw)


def _session(conn, **row):
    cols, marks = ", ".join(row), ", ".join("?" * len(row))
    return conn.execute(f"INSERT INTO sessions ({cols}) VALUES ({marks})", tuple(row.values())).lastrowid


# --- the operator's machine and the target's repo stay out of a run ------------------

def test_a_session_inherits_nothing_from_the_operator():
    o = options()
    assert o.setting_sources == [] and o.strict_mcp_config is True
    assert list(o.mcp_servers) == ["fleetopt"]
    assert [p["path"] for p in o.plugins] == [str(session.PLUGIN)]
    assert o.skills and all(s.startswith("fleetopt:") for s in o.skills)


def test_only_credential_keys_leave_the_operators_settings():
    assert set(config.AUTH_KEYS) == {"env", "apiKeyHelper", "awsAuthRefresh", "awsCredentialExport"}


def test_a_session_has_no_web_no_scheduler_no_subagents():
    assert set(options().tools) == {"Read", "Grep", "Glob", "Bash", "Edit", "Write", "Skill"}


def test_a_session_cannot_read_secrets_or_stop_to_ask():
    denied = options().disallowed_tools
    assert {"AskUserQuestion", "Read(**/.env)", "Read(**/*.pem)", "Read(**/*secret*)"} <= set(denied)


# --- what touches the target is enforced, not asked ----------------------------------

def test_fleetopt_starts_an_agent_only_through_its_own_driver(tmp_path):
    from fleetopt.drive import entry

    command = entry.command(tmp_path / "e.json", {"interpreter": "/proj/.venv/bin/python"})
    assert command.split()[:2] == ["/proj/.venv/bin/python", str(entry.DRIVER)]
    for gone in ("--run", "--auto"):  # nobody hands fleetopt a command, and nobody is asked to approve one
        with pytest.raises(SystemExit):
            cli._parser().parse_args(["apply", "repo", gone, "x"])
    assert "set_run_command" not in {t.name for t in tools._TOOLS}


def test_the_driver_needs_nothing_but_the_projects_own_packages():
    from fleetopt.drive import entry

    import ast

    tree = ast.parse(entry.DRIVER.read_text(encoding="utf-8"))
    # The one exception: the project's own framework, tried and done without. It is how a
    # user message is made the way the project makes one, and where a store comes from.
    optional = {id(n) for t in ast.walk(tree) if isinstance(t, ast.Try)
                and any(isinstance(h.type, ast.Name) and h.type.id == "ImportError" for h in t.handlers)
                for n in t.body}
    imported, tried = set(), set()
    for node in ast.walk(tree):
        names = ({alias.name.split(".")[0] for alias in node.names} if isinstance(node, ast.Import) else
                 {(node.module or "").split(".")[0]} if isinstance(node, ast.ImportFrom) else set())
        (tried if id(node) in optional else imported).update(names)
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)
    assert tried == {"langchain_core", "langgraph"}


def test_an_edit_lands_inside_the_project_on_a_new_branch_or_not_at_all(tmp_path):
    import subprocess

    project = tmp_path / "repo"
    project.mkdir()
    (project / "agent.py").write_text("x = 1\n", encoding="utf-8")
    git = lambda *a: subprocess.run(["git", "-C", str(project), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                                    check=True, capture_output=True)
    git("init", "-q", "-b", "main"); git("add", "-A"); git("commit", "-qm", "base")
    tools.CTX.clear()
    guard = session.guard_edit(project, "main")
    ask = lambda path: asyncio.run(guard({"tool_input": {"file_path": str(path)}}, "id", None))
    denied = lambda v: v.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"

    assert denied(ask(project / "agent.py"))            # still on the branch the run started from
    git("checkout", "-q", "-b", "fleetopt/change")
    tools.CTX["baseline"] = None
    assert denied(ask(project / "agent.py"))            # the agent as it is is measured before anything changes
    tools.CTX.clear()
    assert ask(project / "agent.py") == {}              # fleetopt's branch: go ahead
    assert ask("agent.py") == {}                        # relative paths are the project's
    assert denied(ask(tmp_path / "elsewhere.py"))       # outside the project
    assert denied(ask(project / ".." / "elsewhere.py"))
    assert denied(ask(project / ".git" / "config"))     # not the repository's own files


def test_nothing_is_published():
    guard = session.guard_bash("driver entry.json")
    verdict = asyncio.run(guard({"tool_input": {"command": "git push origin fleetopt/change"}}, "id", None))
    assert verdict["hookSpecificOutput"]["permissionDecision"] == "deny"
    # git is the loop's: the edit session can look, not commit, branch, undo or go round a refusal
    for cmd in ("git checkout -b x", "git commit -am x", "git reset --hard HEAD~1", "git -C . update-index --add x",
                "git hash-object -w f", "git stash", "git add -A"):
        denied = asyncio.run(guard({"tool_input": {"command": cmd}}, "id", None))
        assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", cmd
    for cmd in ("git status", "git diff HEAD", "git log --oneline -3", "python -m py_compile agent.py"):
        assert asyncio.run(guard({"tool_input": {"command": cmd}}, "id", None)) == {}, cmd


def test_the_optimizers_own_spend_is_capped_by_default():
    assert cli._parser().parse_args(["apply", "repo"]).max_usd == 5.0


def test_the_probe_only_observes():
    assert "set_llm_cache" not in HOOK and "_generate" not in HOOK and "bind_tools" not in HOOK
    assert HOOK.count("StateGraph.compile = ") == 1  # the one patch: snapshot the graph as compiled


def test_the_target_never_inherits_fleetopts_own_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLEETOPT_INVARIANT_KEY", raising=False)
    (tmp_path / ".env").write_text("FLEETOPT_INVARIANT_KEY=abc\n", encoding="utf-8")
    config.load_env()
    try:
        assert "FLEETOPT_INVARIANT_KEY" not in config.child_env()
    finally:
        config._injected.discard("FLEETOPT_INVARIANT_KEY")
        monkeypatch.delenv("FLEETOPT_INVARIANT_KEY", raising=False)


# --- a number is only a claim when it has earned it ------------------------------------

def _measured(conn, tokens, exit_code=0):
    sid = _session(conn, project="p", label="x", code_state="v1", exit_code=exit_code)
    conn.execute("INSERT INTO runs (session_id, run_type, model, input_tokens, output_tokens, duration_ms)"
                 " VALUES (?, 'llm', 'claude-haiku-4-5', ?, 10, 100)", (sid, tokens))
    return sid


def test_a_change_inside_the_noise_is_not_a_saving(tmp_path):
    conn = store.connect(tmp_path / "n.db")
    steady = [_measured(conn, 1000) for _ in range(3)]  # reruns that agree exactly: spread 0
    verdict = lambda tokens: measure.compare(conn, steady, [_measured(conn, tokens)])["input_tokens"]["verdict"]
    assert verdict(990) == "within noise"   # 1%: under the minimum margin
    assert verdict(1015) == "within noise"
    assert verdict(900) == "improved"
    assert verdict(1100) == "regressed"


def test_a_crashed_run_never_reaches_a_median(tmp_path):
    tools.CTX.clear()
    tools.CTX.update(out=tmp_path, project=tmp_path / "a")
    with store.connect(tmp_path / "fleetopt.db") as conn:
        _session(conn, project=str(tmp_path / "a"), label="baseline", code_state="v1", exit_code=1)
        good = _session(conn, project=str(tmp_path / "a"), label="baseline", code_state="v1", exit_code=0)
    assert tools._ids("baseline") == [good]


def test_versions_of_the_source_are_never_pooled(tmp_path):
    conn = store.connect(tmp_path / "v.db")
    a = _session(conn, project="p", label="x", code_state="v1", exit_code=0)
    b = _session(conn, project="p", label="x", code_state="v2", exit_code=0)
    with pytest.raises(RuntimeError, match="different versions"):
        measure.aggregate(conn, [a, b])


def test_the_judge_fails_closed(monkeypatch):
    async def garbage(prompt, model=None):
        return {"_unparseable": "well, maybe"}

    monkeypatch.setattr(judge, "_ask", garbage)
    assert asyncio.run(judge.judge("task", "in", "before", "after"))["equivalent"] is False
    assert asyncio.run(judge.judge_expected("task", "in", "expected", "out"))["pass"] is False


# --- the review looks, and a structural change waits for the team's cases --------------

def test_the_reviewer_can_only_look():
    assert not {"Bash", "Edit", "Write", "Skill"} & set(review.READ_ONLY)
    served = {t.name for t in tools._TOOLS if f"mcp__fleetopt__{t.name}" in review.READ_ONLY}
    assert served == {"graph_shape", "graph_topology", "query_traces"}


def test_the_reviewer_gets_no_network_only_local_references():
    assert "create_agent" in review.REFERENCES and "Checked 20" in review.REFERENCES
    assert "WebFetch" not in review.READ_ONLY and "WebSearch" not in review.READ_ONLY


def test_structural_patches_wait_for_eval_cases():
    guide = " ".join((session.PLUGIN / "skills" / "patterns" / "SKILL.md").read_text(encoding="utf-8").split())
    design = [{"id": "D1", "title": "t", "kind": "redesign"}, {"id": "D2", "title": "t", "kind": "design"}]
    assert cli.chosen(design, None, has_cases=False)[0] == []      # decided in code, before any session starts
    assert cli.chosen(design, None, has_cases=True)[0] == design
    assert "only when eval cases are loaded" in guide
    assert "what must survive" in guide


def test_starting_an_agent_is_read_not_guessed_and_only_checked_in_code():
    from fleetopt.drive import entry, setup

    for gone in ("scanned", "candidates", "choose", "inputs_for", "other_provider"):
        assert not hasattr(entry, gone), gone        # the rules that each came from one agent
    skill = " ".join(setup.SKILL.split())
    assert "List each in `missing`" in skill and "never run anything yourself" in skill.lower()
    assert "Never put a key, token or password anywhere in your answer" in skill


def test_nothing_about_the_operator_goes_into_what_is_sent_to_the_agent():
    # Observed 2026-09-28: inputs written for an agent opened with the operator's first
    # name, and the same name was set as the agent's user id.
    from fleetopt.drive import setup

    assert "never use anything about whoever runs this tool" in " ".join(setup.SKILL.split())


def test_a_refusal_is_an_answer_not_an_obstacle():
    # Observed 2026-09-28: edits were refused, and the session spent 50 turns getting the
    # same change in through sed, a glob and git plumbing. It succeeded. That is the fault.
    prompt = " ".join(options().system_prompt.split())
    assert "never look for another way to make the same change" in prompt


def test_design_is_reviewed_only_when_asked_for():
    cost_only, both = review.system(), review.system(design=True)
    assert "# Part two: design" not in cost_only and "## Design" not in cost_only
    assert "# Part two: design" in both and "tier: <one | two>" in both
    assert "a cost change that alters the graph is undone" in " ".join(cost_only.split())
    args = cli._parser().parse_args(["apply", "repo"])
    assert args.design is False and cli._parser().parse_args(["review", "repo", "--design"]).design is True


def test_the_agent_drives_and_the_limits_live_in_its_tools():
    assert (tools.RUNS, tools.ATTEMPTS, tools.TEAM_USD, tools.STEP_FACTOR) == (3, 2, 2.0, 3)
    served = {n.removeprefix("mcp__fleetopt__") for n in options().allowed_tools if n.startswith("mcp__")}
    # it measures, keeps and undoes through fleetopt; there is no tool that judges on its say-so
    assert served == {"measure", "keep", "undo", "query_traces", "graph_topology", "graph_shape"}
    guide = " ".join(session.SKILL.split()).lower()
    assert "never remove what ends a loop" in guide and "when you cannot help" in guide
    assert "review_architecture" not in {t.name for t in tools._TOOLS}  # whoever patches does not also review


def test_the_reviewer_covers_cost_and_design_and_measures_nothing():
    assert "# Part one: cost" in review.SYSTEM and "# Part two: design" in review.SYSTEM
    assert "minimum prefix" in review.SYSTEM.lower()          # the mechanics travel with it, no skill to load
    rules = " ".join(review.system().split())
    assert "never a measured saving" in rules
    assert "never hold a finding back" in rules  # caution goes on the risk line, not in a veto
    assert review.KINDS == ("cost", "design", "redesign")


# --- the verdict belongs to the measurements, not to the agent that wants it --------------

def _judged(candidate, passed, base_state="v1", cand_state="v2", before=4, after=4):
    return {"event": "judge", "baseline": "baseline", "candidate": candidate, "passed": passed,
            "baseline_state": base_state, "candidate_state": cand_state,
            "equivalence": [{"equivalent": True}, {"equivalent": passed}],
            "correctness": {"cases": 12, "matched": 4, "baseline_pass": before, "candidate_pass": after}}


def test_a_failed_gate_cannot_be_argued_away():
    events = [_judged("patched", False, after=3),
              _judged("patched-again", True),                                       # one pass does not erase a failure
              _judged("baseline-retest", False, base_state="v2", cand_state="v2")]  # same code: not a before/after
    text = session.verdict(events)
    assert text.startswith("NOT PROVEN SAFE") and "4/4 before and 3/4 after" in text
    assert "patched-again" in text and "baseline-retest" not in text


def test_nothing_is_proven_without_a_judged_change():
    assert session.verdict([]).startswith("NOTHING PROVEN")
    assert session.verdict([_judged("again", True, cand_state="v1")]).startswith("NOTHING PROVEN")
    assert session.verdict([_judged("patched", True)]).startswith("PROVEN ON THIS EVIDENCE")


def test_the_verdict_is_about_the_code_left_on_the_branch():
    events = [_judged("C1", True, cand_state="v2"), _judged("C2", False, cand_state="v3", after=2)]
    # C2 failed and was undone: the branch holds v2, and v2 passed
    assert session.verdict(events, final="v2", start="v1").startswith("PROVEN ON THIS EVIDENCE")
    # the same events with the failed change still on the branch
    assert session.verdict(events, final="v3", start="v1").startswith("NOT PROVEN SAFE")
    # code nobody judged is not proven by its neighbours
    text = session.verdict(events, final="v4", start="v1")
    assert text.startswith("NOT PROVEN") and "never judged" in text and "v4" in text
    # everything undone
    assert session.verdict(events, final="v1", start="v1").startswith("NOTHING LEFT STANDING")


def test_a_run_of_the_agent_that_does_not_end_is_stopped_with_what_it_started(tmp_path):
    import time

    from fleetopt.probe import runner

    marker = tmp_path / "still-alive"
    # a shell that starts a child which would write a file after 3 seconds: if only the
    # shell were killed, the child would live on and the file would appear
    child = f'{sys.executable} -c "import time, pathlib; time.sleep(3); pathlib.Path(r\'{marker}\').write_text(\'x\')"'
    began = time.time()
    raw, _, _, code = runner.execute(tmp_path, child, tmp_path / "out", timeout=1)
    assert code == runner.TIMED_OUT and time.time() - began < 3
    assert "had not ended" in (raw / "target.log").read_text(encoding="utf-8")
    time.sleep(3)
    assert not marker.exists()


def test_the_teams_money_and_the_clock_both_end_a_run(tmp_path, monkeypatch, capsys):
    import time

    project = tmp_path / "p"
    tools.CTX.clear()
    tools.CTX.update(out=tmp_path, project=project, events=[], first_session=0, max_team_usd=5.0,
                     deadline=time.time() + 60, max_minutes=120)
    monkeypatch.setattr(measure, "spent", lambda out, proj, after: (12, 4.99))
    assert tools.over() is None
    monkeypatch.setattr(measure, "spent", lambda out, proj, after: (13, 5.01))
    assert "limit on the team's key is reached: $5.01 spent in 13 runs" in tools.over()
    monkeypatch.setattr(measure, "spent", lambda out, proj, after: (3, None))   # no price: the clock still holds
    assert tools.over() is None
    tools.CTX["deadline"] = time.time() - 1
    assert "time limit for a run is reached: 120 minutes" in tools.over()

    tools.CTX.clear()


def test_what_is_shown_is_for_a_person():
    assert measure.plain("baseline") == measure.plain("baseline-clean") == "the agent as it is"
    assert measure.plain("review-20260928-173438") == "a first look" and measure.plain("D1-n5") == "D1-n5"
    line = tools.compared("D1", {
        "cost_usd": {"delta_pct": -14.6, "verdict": "within noise"},
        "wall_ms": {"delta_pct": -23.4, "verdict": "improved"},
        "llm_calls": {"delta_pct": None, "verdict": "unpriced"},
        "completed": {"before": 0, "after": 2, "verdict": "improved"}})
    assert line == ("[fleetopt] D1 against the original: cost -15% (no real change), time -23% (better), "
                    "requests finished per run 0 to 2 (better)")
