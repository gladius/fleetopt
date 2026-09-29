"""fleetopt's own tests: the deterministic parts, plus captures of the fixture.

    pytest -q

No model is called anywhere here (conftest.py fails any test that tries), so no key and no
spend. The agent's judgement is tested by runs on real agents.
"""

import asyncio
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

from fleetopt import cli, config
from fleetopt.evidence import evals, measure
from fleetopt.optimizer import agent, tools
from fleetopt.probe import runner, store

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- eval discovery: a team's expected answers, three formats ------------------

def test_evals_load_three_formats_and_skip_venv(tmp_path):
    (tmp_path / "cases.jsonl").write_text(
        '{"input": "capital of France?", "expected": "Paris"}\n'
        '{"query": "2+2", "answer": 4}\n'
        '{"input": "no expected answer here"}\n', encoding="utf-8")
    (tmp_path / "suite.json").write_text(json.dumps(
        {"cases": [{"question": "largest planet", "reference": "Jupiter"}]}), encoding="utf-8")
    (tmp_path / "test_agent.py").write_text(
        "from deepeval.test_case import LLMTestCase\n"
        'case = LLMTestCase(input="who wrote Hamlet", expected_output="Shakespeare")\n'
        'dynamic = LLMTestCase(input=make_input(), expected_output="not a literal, skipped")\n',
        encoding="utf-8")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "ignored.jsonl").write_text('{"input": "x", "expected": "y"}\n', encoding="utf-8")

    (tmp_path / "langsmith_export.json").write_text(json.dumps([
        {"inputs": {"question": "total spent on keyboards?"}, "outputs": {"answer": "$1,118.00"}, "metadata": {}},
        {"inputs": {"text": "only key, odd name"}, "outputs": {"label": "still a case"}},
    ]), encoding="utf-8")

    cases, notes = evals.load(tmp_path)
    assert sorted(c["expected"] for c in cases) == ["$1,118.00", "4", "Jupiter", "Paris", "Shakespeare", "still a case"]
    assert all(".venv" not in c["source"] for c in cases)
    assert len(notes) == 4


def test_evals_match_ignores_case_and_whitespace():
    cases = [{"input": "Capital of  France?", "expected": "Paris", "source": "x"}]
    run_inputs = json.dumps({"messages": [{"role": "user", "content": "What is the capital of france?"}]})
    assert evals.match(cases, run_inputs) is cases[0]
    assert evals.match(cases, json.dumps({"messages": "capital of Spain?"})) is None
    assert evals.match(cases, "") is None


# --- labels: a 'baseline' is this project's, completed, at the newest code state --

def _session(conn, **row):
    cols, marks = ", ".join(row), ", ".join("?" * len(row))
    return conn.execute(f"INSERT INTO sessions ({cols}) VALUES ({marks})", tuple(row.values())).lastrowid


def test_ids_scope_to_project_and_newest_code_state(tmp_path):
    tools.CTX.update(out=tmp_path, project=tmp_path / "a")
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    with store.connect(tmp_path / "fleetopt.db") as conn:
        _session(conn, project=a, label="baseline", code_state="v1", exit_code=0)  # stale source
        _session(conn, project=a, label="baseline", code_state="v1", exit_code=1)  # crashed
        _session(conn, project=b, label="baseline", code_state="v2", exit_code=0)  # another repo
        fresh = [_session(conn, project=a, label="baseline", code_state="v2", exit_code=0) for _ in range(2)]
    assert tools._ids("baseline") == fresh
    assert tools._ids("candidate") == []
# --- the Bash guard: enforced, not asked ------------------------------------------

def _guard(cmd):
    return asyncio.run(agent.guard_bash()({"tool_input": {"command": cmd}}, "id", None))


@pytest.mark.parametrize("cmd", [
    "pip install rich", "python -m pip install -q x", "uv run --with rich python x.py", "uv add rich",
    "poetry add rich", "npm install", "npx something", "curl https://example.com -o f", "brew install jq",
    "python agent.py", "pytest tests/", "langgraph dev", "deepeval test run tests/", "promptfoo eval",
    "/proj/.venv/bin/python /x/fleetopt/probe/driver.py e.json --limit 1",
])
def test_guard_denies_installs_and_running_the_target(cmd):
    assert _guard(cmd)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("cmd", ["git status", "ls -la", "wc -l README.md", "grep -rn StateGraph src"])
def test_guard_allows_everything_else(cmd):
    assert _guard(cmd) == {}


# --- measurement: the baseline's own spread is the noise floor --------------------

def _measured(conn, input_tokens, model="claude-haiku-4-5", code_state="v1"):
    sid = _session(conn, project="p", label="x", code_state=code_state, exit_code=0)
    conn.execute(
        "INSERT INTO runs (session_id, run_type, model, input_tokens, output_tokens, duration_ms)"
        " VALUES (?, 'llm', ?, ?, 10, 100)", (sid, model, input_tokens))
    return sid


def test_compare_applies_noise_floor_and_flags_unpriced(tmp_path):
    conn = store.connect(tmp_path / "m.db")
    base = [_measured(conn, t) for t in (1000, 1100, 1200)]  # median 1100, spread 200

    def verdict(candidate):
        return measure.compare(conn, base, [candidate])["input_tokens"]["verdict"]

    assert verdict(_measured(conn, 1050)) == "within noise"
    assert verdict(_measured(conn, 700)) == "improved"
    assert verdict(_measured(conn, 1500)) == "regressed"
    mystery = measure.compare(conn, base, [_measured(conn, 700, model="mystery-model")])
    assert mystery["cost_usd"]["verdict"] == "unpriced"
    assert list(mystery)[:2] == ["cost_usd", "wall_ms"]  # billed cost is the headline, then latency


def test_compare_refuses_to_pool_different_source_versions(tmp_path):
    conn = store.connect(tmp_path / "m.db")
    ids = [_measured(conn, 1000, code_state="v1"), _measured(conn, 1000, code_state="v2")]
    with pytest.raises(RuntimeError, match="different versions"):
        measure.aggregate(conn, ids)


# --- the judge's bookkeeping, with the LLM calls stubbed --------------------------

def test_judge_sessions_pairs_by_position_and_counts_correctness(tmp_path, monkeypatch):
    from fleetopt.evidence import judge as judge_mod

    conn = store.connect(tmp_path / "j.db")

    def captured(pairs):
        sid = _session(conn, project="p", label="x", code_state="v1", exit_code=0)
        for i, (inputs, outputs) in enumerate(pairs):
            conn.execute(
                "INSERT INTO runs (session_id, run_type, inputs, outputs, start_time)"
                " VALUES (?, 'chain', ?, ?, ?)", (sid, inputs, outputs, f"2026-01-01T00:00:0{i}"))
        return sid

    base = captured([("capital of France?", "Paris"), ("2+2", "4")])
    cand = captured([("capital of France?", "Paris."), ("2+2", "5")])

    async def fake_judge(task, inputs, before, after, model=None):
        return {"equivalent": before.rstrip(".") == after.rstrip("."), "reason": "stub"}

    async def fake_expected(task, inputs, expected, output, model=None):
        return {"pass": output.rstrip(".") == expected, "reason": "stub"}

    monkeypatch.setattr(judge_mod, "judge", fake_judge)
    monkeypatch.setattr(judge_mod, "judge_expected", fake_expected)
    cases = [{"input": "capital of France?", "expected": "Paris", "source": "s"},
             {"input": "2+2", "expected": "4", "source": "s"}]

    passed, results, correctness = asyncio.run(judge_mod.judge_sessions(conn, "qa", base, cand, cases))
    assert not passed
    assert [r["equivalent"] for r in results] == [True, False]
    assert (correctness["matched"], correctness["baseline_pass"], correctness["candidate_pass"]) == (2, 2, 1)

    # What passes, request by request. Reworded and still right by the team's case: passes.
    reworded = captured([("capital of France?", "It is Paris"), ("2+2", "4")])

    async def expected_contains(task, inputs, expected, output, model=None):
        return {"pass": expected in output, "reason": "stub"}

    monkeypatch.setattr(judge_mod, "judge_expected", expected_contains)
    passed, results, _ = asyncio.run(judge_mod.judge_sessions(conn, "qa", base, reworded, cases))
    assert passed and [r["equivalent"] for r in results] == [False, True]
    assert results[0]["kept_on"] == "correct on the team's case"
    # The same rewording with no case to say it is right: nothing to go on but the old answer.
    passed, results, _ = asyncio.run(judge_mod.judge_sessions(conn, "qa", base, reworded))
    assert not passed and results[0]["kept_on"] is None
    # Both wrong: unchanged is no worse, a different wrong answer is not evidence of anything.
    wrong = captured([("capital of France?", "Lyon"), ("2+2", "4")])
    assert asyncio.run(judge_mod.judge_sessions(conn, "qa", wrong, captured([("capital of France?", "Lyon"), ("2+2", "4")]), cases))[0]
    assert not asyncio.run(judge_mod.judge_sessions(conn, "qa", wrong, captured([("capital of France?", "Nice"), ("2+2", "4")]), cases))[0]
    # A request the original never finished: judged on the team's case, since there is no old answer.
    def with_failure(pairs):
        sid = captured(pairs)
        conn.execute("UPDATE runs SET outputs = NULL, error = 'IndexError()' WHERE session_id = ? AND inputs = '2+2'", (sid,))
        return sid

    broken = with_failure([("capital of France?", "Paris"), ("2+2", "x")])
    fixed = captured([("capital of France?", "Paris"), ("2+2", "4")])
    passed, results, correctness = asyncio.run(judge_mod.judge_sessions(conn, "qa", broken, fixed, cases))
    assert passed and results[1]["kept_on"] == "correct on the team's case" and "did not finish" in results[1]["reason"]
    assert (correctness["baseline_pass"], correctness["candidate_pass"]) == (1, 2)
    assert not asyncio.run(judge_mod.judge_sessions(conn, "qa", broken, fixed))[0]     # no case: finishing is not yet being right
    assert not asyncio.run(judge_mod.judge_sessions(conn, "qa", fixed, broken, cases))[0]  # and breaking a request never passes
    assert not asyncio.run(judge_mod.judge_sessions(conn, "qa", broken, broken, cases))[0]  # still broken is not proven

    # It was right and now it is not: fails, however alike a judge finds the two.
    monkeypatch.setattr(judge_mod, "judge", lambda *a, **k: _always_equivalent())
    assert not asyncio.run(judge_mod.judge_sessions(conn, "qa", base, captured([("capital of France?", "Lyon"), ("2+2", "4")]), cases))[0]
    with pytest.raises(RuntimeError, match="different number"):
        asyncio.run(judge_mod.judge_sessions(conn, "qa", base, captured([("x", "y")])))


async def _always_equivalent():
    return {"equivalent": True, "reason": "stub"}


# --- small things that each broke a real run once ---------------------------------

def test_output_tail_survives_undecodable_bytes(tmp_path):
    body = b"first\n\xff\xfe not utf-8\n" + b"\n".join(f"l{i}".encode() for i in range(30))
    (tmp_path / "target.log").write_bytes(body)
    assert runner.output_tail(tmp_path, lines=3).splitlines() == ["l27", "l28", "l29"]
    assert runner.output_tail(tmp_path / "missing") == ""


def test_child_env_never_carries_fleetopts_own_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLEETOPT_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("# comment\nFLEETOPT_TEST_KEY=abc\n", encoding="utf-8")
    config.load_env()
    try:
        assert os.environ["FLEETOPT_TEST_KEY"] == "abc"
        assert "FLEETOPT_TEST_KEY" not in config.child_env()
    finally:
        config._injected.discard("FLEETOPT_TEST_KEY")
        monkeypatch.delenv("FLEETOPT_TEST_KEY", raising=False)


# --- isolation: authenticate like Claude Code, inherit nothing else ---------------

def test_sdk_args_pass_only_credential_keys_from_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "settings.json").write_text(json.dumps({
        "env": {"ANTHROPIC_BASE_URL": "https://gateway.example", "ANTHROPIC_AUTH_TOKEN": "t"},
        "apiKeyHelper": "/usr/local/bin/key.sh",
        "enabledPlugins": {"someone-elses@marketplace": True},
        "hooks": {"SessionStart": []},
        "model": "opus",
    }), encoding="utf-8")
    assert set(json.loads(config.sdk_args()["settings"])) == {"env", "apiKeyHelper"}
    assert {"enabledPlugins", "hooks", "model"}.isdisjoint(config.AUTH_KEYS)
    (tmp_path / "settings.json").unlink()
    assert config.sdk_args() == {}




def test_a_capture_that_fails_late_keeps_the_targets_last_words(tmp_path):
    target = tmp_path / "fixture"
    shutil.copytree(ROOT / "fixture", target)
    cmd = f'{sys.executable} agent.py && {sys.executable} -c "print(\'the reason it failed\'); raise SystemExit(3)"'
    with pytest.raises(RuntimeError, match="exited 3 after .* runs(.|\\n)*the reason it failed"):
        measure.collect(target, cmd, tmp_path / "out", 1, "x")


def test_compare_counts_finished_requests_and_prices_only_those(tmp_path):
    conn = store.connect(tmp_path / "c.db")

    def run_of(finished, tokens):
        sid = _session(conn, project="p", label="x", code_state="v1", exit_code=0)
        for i in range(3):
            ok = i < finished
            conn.execute("INSERT INTO runs (session_id, run_type, name, outputs, error, duration_ms) VALUES (?, 'chain', 'root', ?, ?, 100)",
                         (sid, "answer" if ok else None, None if ok else "IndexError('x')"))
        conn.execute("INSERT INTO runs (session_id, run_type, model, input_tokens, output_tokens, parent_run_id)"
                     " VALUES (?, 'llm', 'claude-haiku-4-5', ?, 100, 'r')", (sid, tokens))
        return sid

    crashing = [run_of(0, 10_000) for _ in range(3)]   # cheap, and does nothing
    working = [run_of(3, 20_000) for _ in range(3)]    # costs more, finishes everything
    result = measure.compare(conn, crashing, working)
    assert result["cost_usd"]["verdict"] == "regressed"                       # true, and misleading alone
    assert (result["completed"]["before"], result["completed"]["after"], result["completed"]["verdict"]) == (0, 3, "improved")
    assert result["cost_per_completed"]["verdict"] == "baseline finished nothing"
    assert "completed" in measure.render(result)


# --- the probe on the bundled fixture: no key, no model, a few seconds ---------------------

def _fixture(tmp_path):
    target = tmp_path / "fixture"
    if not target.exists():
        shutil.copytree(ROOT / "fixture", target)
    return target.resolve()


def _captured(tmp_path, graph, inputs=("battery degradation", "route optimization")):
    """Run one of the fixture's agents the way fleetopt does: its driver, under the probe."""
    project = _fixture(tmp_path)
    entry = {"project": str(project), "graph": graph, "paths": ["."], "interpreter": sys.executable,
             "env_file": None, "env": {}, "config": {}, "input_template": None, "inputs": list(inputs)}
    path = tmp_path / "entry.json"
    path.write_text(json.dumps(entry), encoding="utf-8")
    return runner.run(project, tools.command(path, entry), tmp_path / "out", with_io=True, label="x")


def test_the_probe_records_an_agent_it_never_edited(tmp_path):
    _, code, n_runs, n_graphs = _captured(tmp_path, "agent.py:graph")
    assert code == 0 and n_runs > 0 and n_graphs == 1
    conn = store.connect(tmp_path / "out" / "fleetopt.db")
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE run_type = 'llm' AND input_tokens > 0").fetchone()[0] > 0


def test_the_probe_sees_every_node_that_ran_and_none_that_did_not(tmp_path):
    assert _captured(tmp_path, "supervisor.py:graph")[1] == 0
    conn = store.connect(tmp_path / "out" / "fleetopt.db")
    nodes = {r[0] for r in conn.execute("SELECT DISTINCT node FROM runs WHERE node IS NOT NULL")}
    assert {"route", "technical", "supervisor", "worker_a", "worker_b", "worker_c", "draft", "reflect"} <= nodes
    assert not {"billing", "other"} & nodes  # in the graph, never taken


# --- the one agent: its guide, its command line ----------------------------------------------

def test_the_agent_reads_one_guide_and_every_cost_skill():
    assert config.SETTING_SOURCES == []
    on_disk = {p.name for p in (ROOT / "fleetopt" / "optimizer" / "skills").iterdir() if p.is_dir()}
    assert on_disk == set(agent.SKILLS)
    assert all(f"fleetopt:{name}" in agent.SYSTEM for name in agent.SKILLS)
    assert "minimum prefix" in agent.SYSTEM and "## The flow" in agent.SYSTEM


def test_the_command_line_is_two_commands_and_a_few_flags():
    parse = cli._parser().parse_args
    assert parse(["apply", "repo"]).out == ".fleetopt" and parse(["apply", "repo"]).max_usd == 5.0
    assert parse(["apply", "repo", "--out", "after"]).out == "after"
    assert parse(["--out", "before", "review", "repo"]).out == "before"
    args = parse(["review", "repo", "--evals", "cases.jsonl", "--graph", "supervisor"])
    assert (args.cmd, args.evals, args.graph, args.max_usd) == ("review", "cases.jsonl", "supervisor", 2.0)
    for gone in (["apply", "repo", "--only", "C1"], ["review", "repo", "--design"], ["capture", "repo"]):
        with pytest.raises(SystemExit):
            parse(gone)


# --- the tools: what is kept has earned it ---------------------------------------------------

def _repo(tmp_path):
    project = tmp_path / "agent"
    project.mkdir()
    (project / "agent.py").write_text("x = 0\n", encoding="utf-8")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"]):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    (tmp_path / "out").mkdir()
    return project.resolve()


ENTRY = {"name": "agent", "graph": "agent.py:graph", "inputs": ["x"], "interpreter": sys.executable,
         "inputs_from": "the test"}


def _fake_runs(monkeypatch, failing=(), flat=(), wrong=(), reshaped=()):
    """Running, comparing, judging and the graph's structure, faked by the change saved last."""
    seen = {"measured": [], "judged": []}
    last = lambda: tools.CTX["saved"][-1] if tools.CTX["saved"] else ""

    def run(label, max_steps=None, probe=False):
        seen["measured"].append((label, max_steps, probe))
        if last() in failing:
            return None, "run 1 took more than 150 steps, far more than the original, and was stopped."
        return {"steps": 40, "completed": 3, "cost_usd": 0.01, "llm_calls": 5}, None

    async def judge(label):
        seen["judged"].append(last())
        ok = last() not in wrong
        return ok, [{"kept_on": "unchanged answer" if ok else None, "reason": "an answer was cut off"}]

    monkeypatch.setattr(tools, "_measure", run)
    monkeypatch.setattr(tools, "_compare", lambda before, after: {"cost_usd": {
        "before": 0.0125, "after": 0.01, "delta_pct": -20.0, "verdict": "within noise" if last() in flat else "improved"}})
    monkeypatch.setattr(tools, "_judge", judge)
    monkeypatch.setattr(tools, "_shape", lambda label: ["changed"] if label != "baseline" and last() in reshaped
                        else ["same"])
    return seen


def _call(tool_name, **args):
    return asyncio.run(getattr(tools, tool_name).handler(args))["content"][0]["text"]


def test_the_tools_keep_what_earns_it_and_undo_the_rest(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    seen = _fake_runs(monkeypatch, failing={"loop forever"}, flat={"trim notes"}, wrong={"smaller model"},
                      reshaped={"merge two steps"})
    cases = [{"input": "when is my refund due?", "expected": "Refunds are issued within 14 days of the request.",
              "source": "x"}]
    edit = lambda text, name="agent.py": (project / name).write_text(text, encoding="utf-8")
    git = lambda *a: subprocess.run(["git", "-C", str(project), *a], capture_output=True, text=True, check=True).stdout
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry={**ENTRY, "project": str(project)},
                cases=cases)
    git("checkout", "-q", "-b", "fleetopt/test")
    try:
        assert "measure the agent as it is first" in _call("save_change", name="too early")
        assert "Edits are allowed now" in _call("measure")                 # the agent as it is, once
        assert "Nothing saved" in _call("measure")

        edit("x = 1\n"), _call("save_change", name="cache the system prompt")   # a bundle of two, measured once
        edit("y = 1\n", "helper.py"), _call("save_change", name="bound the output")
        assert "got better past the noise" in _call("measure")
        assert "Kept" in _call("keep")

        edit("x = 2\n"), edit("z = 2\n", "scratch.py"), _call("save_change", name="loop forever")
        assert "could not be measured" in _call("measure")                 # it ran away and was stopped
        assert "Undone" in _call("undo", why="it looped")
        assert not (project / "scratch.py").exists()                       # nothing half-made is left

        edit("x = 3\n"), _call("save_change", name="trim notes"), _call("measure")
        assert "nothing got better past the noise" in _call("keep")
        _call("undo", why="")
        edit("x = 4\n"), _call("save_change", name="smaller model"), _call("measure")
        assert "answers changed on 1 of 1 requests: an answer was cut off" in _call("keep")
        _call("undo", why="")
        edit("x = 5\n"), _call("save_change", name="merge two steps"), _call("measure")
        assert "nodes or edges" in _call("keep")                            # design is not this run's to change
        _call("undo", why="")
        edit("x = 6\n")
        assert "unsaved edits" in _call("measure")
        edit('ANSWER = "Refunds are issued within 14 days of the request."\n')
        _call("save_change", name="hardcode the answer"), _call("measure")
        assert "writes an answer from the team's eval cases" in _call("keep")
        tools.finish()                                                     # left unproven: undone for it

        changes = {k: v[0] for k, v in tools.CTX["changes"].items()}
        assert changes == {"cache the system prompt": "kept", "bound the output": "kept", "loop forever": "undone",
                           "trim notes": "undone", "smaller model": "undone", "merge two steps": "undone",
                           "hardcode the answer": "undone"}
        assert "150 steps" in tools.CTX["changes"]["loop forever"][1]
        assert seen["judged"] == ["bound the output", "smaller model"]     # nothing is judged that did not gain
        assert seen["measured"][0] == ("baseline", None, False)
        assert all(probe and steps == 150 for _, steps, probe in seen["measured"][1:])  # changed code is watched
        assert git("log", "--format=%s").split("\n")[:3] == ["bound the output", "cache the system prompt", "base"]
        assert (project / "agent.py").read_text(encoding="utf-8") == "x = 1\n" and (project / "helper.py").exists()

        tools.CTX["deadline"] = 1
        edit("x = 9\n"), _call("save_change", name="late")
        assert "time limit" in _call("measure")
    finally:
        tools.CTX.clear()


def test_an_agent_whose_model_calls_cannot_be_seen_is_said_so_and_nothing_more_is_spent(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    runs = []
    monkeypatch.setattr(tools, "_measure", lambda label, max_steps=None, probe=False: runs.append(label) or (
        {"steps": 12, "completed": 2, "llm_calls": 0}, None))
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry={**ENTRY, "project": str(project)})
    try:
        assert "without LangChain" in _call("measure")
        assert "Refused" in _call("measure") and runs == ["baseline"]
    finally:
        tools.CTX.clear()


# --- the one session, end to end, with a scripted stand-in for the model ----------------------

def _tried(monkeypatch):
    monkeypatch.setattr(tools, "trial", lambda path, entry, timeout=600: {
        "exit": 0, "requests": 1, "finished": 1, "model_calls": 3, "answered": 3, "error": None, "tail": "ok"})


def test_one_agent_starts_measures_changes_and_leaves_nothing_unproven(tmp_path, monkeypatch):
    import claude_agent_sdk

    project = _repo(tmp_path)
    _fake_runs(monkeypatch)
    _tried(monkeypatch)
    seen = {}

    async def the_agent(prompt, options):  # starts it, keeps one change, walks away from another
        seen["prompt"], seen["tools"] = prompt, options.allowed_tools
        call = lambda tool_name, **a: getattr(tools, tool_name).handler(a)
        await call("start", entry=json.dumps({"graph": "agent.py:graph", "agent": "researcher", "job": "research",
                                              "inputs": ["battery degradation"]}))
        await call("measure")
        (project / "agent.py").write_text("x = 1\n", encoding="utf-8")
        await call("save_change", name="cache the system prompt")
        await call("measure")
        await call("keep")
        (project / "agent.py").write_text("x = 2\n", encoding="utf-8")
        await call("save_change", name="trim notes")
        await call("measure")
        return
        yield

    monkeypatch.setattr(claude_agent_sdk, "query", the_agent)
    try:
        facts = asyncio.run(agent.run(project, tmp_path / "out"))
    finally:
        tools.CTX.clear()
    git = lambda *a: subprocess.run(["git", "-C", str(project), *a], capture_output=True, text=True).stdout.strip()
    assert "How to start it is not known yet" in seen["prompt"] and "Eval cases: none" in seen["prompt"]
    assert "mcp__fleetopt__keep" in seen["tools"]
    assert facts["kept"] == 1 and facts["branch"].startswith("fleetopt/")
    assert [(c["name"], c["outcome"]) for c in facts["changes"]] == [("cache the system prompt", "kept"),
                                                                    ("trim notes", "undone")]
    assert git("rev-parse", "--abbrev-ref", "HEAD") == "main"                # the team's copy is where it was
    assert git("show", f"{facts['branch']}:agent.py") == "x = 1"             # the branch holds what was kept
    text = "\n".join(agent.summary(facts))
    assert "1 change(s) kept on branch fleetopt/" in text and "no eval cases" in text
    entry = json.loads(tools.entry_path(tmp_path / "out", project).read_text(encoding="utf-8"))
    assert entry["proven"] and entry["name"] == "researcher"                 # the next run knows how to start it


def test_review_is_the_same_agent_without_anything_that_changes_code(tmp_path, monkeypatch):
    import claude_agent_sdk

    project = _repo(tmp_path)
    _fake_runs(monkeypatch)
    _tried(monkeypatch)
    seen = {}

    async def the_agent(prompt, options):
        seen["tools"] = set(options.allowed_tools)
        await tools.start.handler({"entry": json.dumps({"graph": "agent.py:graph", "inputs": ["x"]})})
        await tools.measure.handler({})
        yield claude_agent_sdk.ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                                             num_turns=3, session_id="s", total_cost_usd=0.1,
                                             result="Worth changing:\n- cache the system prompt: 0 cache reads")

    monkeypatch.setattr(claude_agent_sdk, "query", the_agent)
    try:
        facts = asyncio.run(agent.run(project, tmp_path / "out", look_only=True))
    finally:
        tools.CTX.clear()
    assert not seen["tools"] & {"Bash", "Edit", "Write", "mcp__fleetopt__save_change", "mcp__fleetopt__keep",
                                "mcp__fleetopt__undo"}
    assert facts["branch"] is None and facts["kept"] == 0
    branches = subprocess.run(["git", "-C", str(project), "branch"], capture_output=True, text=True).stdout
    assert "fleetopt" not in branches
    reviews = list((tmp_path / "out" / "reviews").glob("*.md"))
    assert len(reviews) == 1 and "cache the system prompt" in reviews[0].read_text(encoding="utf-8")
    assert "Next     fleetopt apply" in "\n".join(agent.summary(facts))
