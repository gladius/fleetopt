"""fleetopt's own tests: the deterministic parts, plus one capture of the fixture.

    pytest -q

No LLM call anywhere here, so no key and no spend. The optimizer's judgement is
tested separately: the skills by `claude plugin eval` (see
fleetopt/optimizer/plugin/evals/), the whole loop by runs on real repos recorded
in tests/corpus/ledger.md. README, "Testing".
"""

import asyncio
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from fleetopt import cli, config
from fleetopt.evidence import evals, measure, shape
from fleetopt.optimizer import review, session, tools
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
    tools.CTX["include_failed"] = True   # a review, never a measurement
    try:
        assert len(tools._ids("baseline")) == 2  # newest code state has no crashed run; still scoped to it
    finally:
        tools.CTX.pop("include_failed")


# --- the Bash guard: enforced, not asked ------------------------------------------

def _guard(cmd, run_cmd="python agent.py"):
    return asyncio.run(session.guard_bash(run_cmd)({"tool_input": {"command": cmd}}, "id", None))


@pytest.mark.parametrize("cmd", [
    "pip install rich", "python -m pip install -q x", "uv run --with rich python x.py", "uv add rich",
    "poetry add rich", "npm install", "npx something", "curl https://example.com -o f", "brew install jq",
    "python agent.py", "pytest tests/", "langgraph dev", "deepeval test run tests/", "promptfoo eval",
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


def test_sessions_load_no_operator_settings_and_only_fleetopt_skills():
    assert config.SETTING_SOURCES == []
    on_disk = sorted(p.name for p in (session.PLUGIN / "skills").iterdir() if p.is_dir())
    assert session.SKILLS == [f"fleetopt:{name}" for name in on_disk]
    assert len(on_disk) >= 6


# --- the probe on the bundled fixture: no key, no LLM, a few seconds --------------

def test_capture_fixture_end_to_end(tmp_path):
    target = tmp_path / "fixture"
    shutil.copytree(ROOT / "fixture", target)

    def git(*args):
        subprocess.run(["git", "-C", str(target), *args], check=True, capture_output=True)

    git("init", "-q")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    out = subprocess.run(
        [sys.executable, "-c", "from fleetopt.cli import main; main()",
         "capture", str(target), "--out", str(tmp_path / "out")],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "out" / "fleetopt.db").exists()
    runs = re.search(r"(\d+) runs, (\d+) graphs", out.stdout)
    assert runs and int(runs.group(1)) > 0 and int(runs.group(2)) == 1, out.stdout


# --- the second fixture ------------------------------------------------------------------

def _capture_fixture(tmp_path, script, *extra):
    """Capture one of the fixture's two agents, named by its file: agent.py or supervisor.py."""
    target = tmp_path / "fixture"
    if not target.exists():
        shutil.copytree(ROOT / "fixture", target)
        for args in (["init", "-q"], ["add", "-A"],
                     ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"]):
            subprocess.run(["git", "-C", str(target), *args], check=True, capture_output=True)
    return subprocess.run(
        [sys.executable, "-c", "from fleetopt.cli import main; main()", "capture", str(target),
         "--graph", f"{script}:graph", "--out", str(tmp_path / "out"), *extra],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )


def test_supervisor_fixture_captures_its_planted_smells(tmp_path):
    out = _capture_fixture(tmp_path, "supervisor.py")
    assert out.returncode == 0, out.stdout + out.stderr
    conn = store.connect(tmp_path / "out" / "fleetopt.db")
    nodes = {r[0] for r in conn.execute("SELECT DISTINCT node FROM runs WHERE node IS NOT NULL")}
    assert {"route", "technical", "supervisor", "worker_a", "worker_b", "worker_c", "draft", "reflect"} <= nodes
    assert not {"billing", "other"} & nodes  # in the graph, never taken


def test_out_is_accepted_before_and_after_the_subcommand():
    parse = cli._parser().parse_args
    assert parse(["apply", "repo"]).out == ".fleetopt"
    assert parse(["apply", "repo", "--out", "after"]).out == "after"
    assert parse(["--out", "before", "apply", "repo"]).out == "before"
    assert parse(["--out", "before", "capture", "repo", "--out", "after"]).out == "after"
    args = parse(["review", "repo"])
    assert (args.out, args.fresh, args.graph, args.max_usd, args.fn.__name__) == (".fleetopt", False, None, 1.0, "review")
    assert parse(["review", "repo", "--fresh", "--graph", "supervisor"]).graph == "supervisor"
    assert parse(["review", "repo", "--evals", "cases.jsonl"]).evals == "cases.jsonl"  # the same cases to look and to change
    args = parse(["apply", "repo", "--only", "C1,D2"])
    assert (args.only, args.evals, args.max_usd, args.fn.__name__) == ("C1,D2", None, 5.0, "apply")


# --- structural smells: numbers, not opinions ------------------------------------------

def test_shape_finds_the_supervisor_fixtures_planted_smells_and_nothing_else(tmp_path):
    assert _capture_fixture(tmp_path, "supervisor.py").returncode == 0
    conn = store.connect(tmp_path / "out" / "fleetopt.db")
    ids = [r[0] for r in conn.execute("SELECT id FROM sessions")]
    result = shape.analyze(conn, ids)
    kinds = {(f["kind"], f["node"]) for f in result["findings"]}
    assert result["traces"] == 2 and result["distinct_inputs"] == 2
    assert ("branch_never_taken", "route") in kinds          # billing, other exist and are never taken
    assert ("fixed_dispatch", "supervisor") in kinds         # worker_a -> worker_b -> worker_c -> draft, every time
    assert ("constant_rounds", "reflect") in kinds           # three rounds, always
    assert ("repeated_identical_reply", "reflect") in kinds  # the critic never says anything new
    never = next(f for f in result["findings"] if f["kind"] == "branch_never_taken")
    assert sorted(never["targets"]) == ["billing", "other"]
    dispatches = [f for f in result["findings"] if f["kind"] == "fixed_dispatch"]
    assert [d["node"] for d in dispatches] == ["supervisor"]  # reflect's self-loop is rounds, not dispatch
    assert dispatches[0]["order"] == ["worker_a", "worker_b", "worker_c", "draft"] and dispatches[0]["calls_model"]
    assert not {n for _, n in kinds} - {"route", "supervisor", "reflect"}  # no finding on a healthy node
    text = shape.render(result)
    assert "2 traces" in text and "never taken in 2 traces" in text


def test_shape_on_the_cost_fixture_sees_only_the_research_loop(tmp_path):
    assert _capture_fixture(tmp_path, "agent.py").returncode == 0
    conn = store.connect(tmp_path / "out" / "fleetopt.db")
    result = shape.analyze(conn, [r[0] for r in conn.execute("SELECT id FROM sessions")])
    assert {f["node"] for f in result["findings"]} == {"research"}  # runs 3 rounds to its cap
    assert shape.render({"traces": 0, "nodes": [], "findings": []}) == "no traces to analyze"


def test_graph_shape_tool_renders_the_fixtures_smells(tmp_path):
    assert _capture_fixture(tmp_path, "supervisor.py").returncode == 0
    tools.CTX.update(out=tmp_path / "out", project=(tmp_path / "fixture").resolve(), events=[])
    reply = json.dumps(asyncio.run(tools.graph_shape.handler({"label": "manual"})))
    assert "2 traces" in reply and "never taken" in reply and "same order" in reply
    assert tools.CTX["events"][-1]["event"] == "graph_shape"
    assert "no completed measurement" in json.dumps(asyncio.run(tools.graph_shape.handler({"label": "nope"})))


def test_shape_counts_distinct_inputs_not_just_traces(tmp_path):
    for _ in range(3):  # a baseline: the same command, three times
        assert _capture_fixture(tmp_path, "supervisor.py").returncode == 0
    conn = store.connect(tmp_path / "out" / "fleetopt.db")
    result = shape.analyze(conn, [r[0] for r in conn.execute("SELECT id FROM sessions")])
    assert (result["traces"], result["distinct_inputs"]) == (6, 2)
    assert "2 inputs wide" in shape.render(result)


def test_shape_reports_swallowed_errors_and_human_pauses(tmp_path):
    conn = store.connect(tmp_path / "e.db")
    sid = _session(conn, project="p", label="x", code_state="v1", exit_code=0)

    def run(trace, step, name, node, run_type="chain", error=None):
        conn.execute(
            "INSERT INTO runs (session_id, trace_id, run_type, name, node, step, error, start_time)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (sid, trace, run_type, name, node, step, error, f"{trace}-{step:02d}-{name}"))

    for trace in ("t1", "t2"):
        run(trace, 1, "draft", "draft")
        run(trace, 2, "critic", "critic")                       # the node itself did not fail...
        run(trace, 2, "ChatAnthropic", "critic", "llm",         # ...the call inside it did
            error="BadRequestError(\"Error code: 400 - temperature is deprecated\")Traceback (most recent call last): ...")
        run(trace, 3, "human_gate", "human_gate", error="GraphInterrupt((Interrupt(value={}),))Traceback ...")
    result = shape.analyze(conn, [sid])
    first, *_ = result["findings"]
    assert (first["kind"], first["node"], first["count"], first["swallowed"]) == ("node_error", "critic", 2, True)
    assert "temperature is deprecated" in first["text"] and "caught inside the node" in first["text"]
    pause = next(f for f in result["findings"] if f["kind"] == "interrupt")
    assert (pause["node"], pause["count"]) == ("human_gate", 2) and "paused for a human in 2/2" in pause["text"]


def test_a_capture_that_fails_late_keeps_the_targets_last_words(tmp_path):
    target = tmp_path / "fixture"
    shutil.copytree(ROOT / "fixture", target)
    cmd = f'{sys.executable} agent.py && {sys.executable} -c "print(\'the reason it failed\'); raise SystemExit(3)"'
    with pytest.raises(RuntimeError, match="exited 3 after .* runs(.|\\n)*the reason it failed"):
        measure.collect(target, cmd, tmp_path / "out", 1, "x")


def test_shape_counts_model_calls_per_tool_round_and_stays_quiet_on_one_trace(tmp_path):
    conn = store.connect(tmp_path / "r.db")
    sid = _session(conn, project="p", label="x", code_state="v1", exit_code=0)
    rows = [(1, "plan", "plan", "chain"), (1, "M", "plan", "llm"), (2, "decide", "decide", "chain"), (2, "M", "decide", "llm"),
            (3, "act", "act", "chain"), (3, "M", "act", "llm"), (4, "tools", "tools", "chain"), (4, "search", "tools", "tool"),
            (5, "reflect", "reflect", "chain"), (5, "M", "reflect", "llm"), (6, "decide", "decide", "chain"), (6, "M", "decide", "llm")]
    for step, name, node, kind in rows:
        conn.execute("INSERT INTO runs (session_id, trace_id, run_type, name, node, step, completion, start_time)"
                     " VALUES (?, 't1', ?, ?, ?, ?, 'search', ?)", (sid, kind, name, node, step, f"{step:02d}{kind}"))
    result = shape.analyze(conn, [sid])
    kinds = {f["kind"] for f in result["findings"]}
    ratio = next(f for f in result["findings"] if f["kind"] == "calls_per_tool_round")
    assert (ratio["model_calls"], ratio["tool_rounds"], ratio["ratio"]) == (4, 1, 4.0)  # 5 calls, less the answer
    assert not kinds & {"constant_rounds", "fixed_dispatch", "repeated_identical_reply"}  # one trace proves no habit


def test_a_plain_tool_calling_agent_is_not_called_over_built(tmp_path):
    conn = store.connect(tmp_path / "p.db")
    sid = _session(conn, project="p", label="x", code_state="v1", exit_code=0)

    def run(trace, step, kind, node):
        conn.execute("INSERT INTO runs (session_id, trace_id, run_type, name, node, step, start_time)"
                     " VALUES (?, ?, ?, ?, ?, ?, ?)", (sid, trace, kind, node, node, step, f"{trace}{step:02d}{kind}"))

    for trace in ("with-tool-1", "with-tool-2"):      # call the tool, then answer: 2 model calls, 1 round
        for step, kind, node in ((1, "chain", "assistant"), (1, "llm", "assistant"), (2, "chain", "tools"),
                                 (2, "tool", "tools"), (3, "chain", "assistant"), (3, "llm", "assistant")):
            run(trace, step, kind, node)
    for trace in ("no-tool-1", "no-tool-2", "no-tool-3"):  # answers directly: these used to inflate the ratio
        run(trace, 1, "chain", "assistant")
        run(trace, 1, "llm", "assistant")
    kinds = {f["kind"] for f in shape.analyze(conn, [sid])["findings"]}
    assert "calls_per_tool_round" not in kinds
    assert "fixed_dispatch" not in kinds  # going round the same loop is not choosing among workers


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


def test_review_stops_when_there_is_no_agent_to_start(tmp_path, capsys, monkeypatch):
    target = tmp_path / "t"
    target.mkdir()
    (target / "notes.py").write_text("print('no graph in here')\n", encoding="utf-8")
    monkeypatch.setattr(config, "auth_summary", lambda: "test")

    def never(*a, **k):
        raise AssertionError("the reviewer must not be started when the agent never ran")

    from fleetopt.optimizer import review as review_mod
    monkeypatch.setattr(review_mod, "run", never)
    code = cli.main(["review", str(target), "--out", str(tmp_path / "out")])
    assert code == 1 and "no graph found" in capsys.readouterr().out


# --- a review is a list of numbered findings, kept against the code it saw -----------------

REPORT = """Job: answers questions about orders.
Evidence: 4 traces of 4 distinct inputs, from one capture. Eval cases: 12 found in the repository

## Cost

### C1 - System prompt is never cached
pattern: caching not used on assistant
evidence: cache_read_tokens = 0 across 9 calls, 2,310-token prefix repeated
source: agent.py:41
risk: none seen

### C2 - Router on a frontier model
evidence: 4 calls, completions of 1 token
risk: it may route differently; I would want the team's cases before trusting this
apply: needs cases

## Design

### D1 — Hand-built agent loop
evidence: calls_per_tool_round 3.0
**tier:** two

### D2 - Reflection never changes the draft
evidence: repeated_identical_reply on reflect in 4/4 traces
change: drop the round
tier: one

### D3 - Planner nobody reads
change: drop the planner call

### C3 - Nothing to cache here
change: none - there is no stable prefix

## Checked and fine

- supervisor: 3 distinct worker orders in 4 traces
"""


def test_what_may_be_tried_is_a_rule_not_the_reviewers_opinion():
    found = review.findings(REPORT)
    assert [(f["id"], f["kind"]) for f in found] == [
        ("C1", "cost"), ("C2", "cost"),          # cost stays cost, however cautious the reviewer felt about it
        ("D1", "redesign"), ("D2", "design"),
        ("D3", "redesign")]                      # a design finding without a readable tier is taken as the larger change
    assert "C3" not in {f["id"] for f in found}  # something checked and cleared is not a finding to try
    assert review.level(found) == 3 and review.LEVELS[3] == "wrong shape"
    assert review.level(found[:2]) == 1 and review.level([]) == 0
    assert review.level([], unfinished=2) == 4   # an agent that does not finish its requests comes before everything
    assert found[0]["title"] == "System prompt is never cached"
    assert review.findings("Nothing here has a number.") == []


def test_what_apply_tries_is_decided_in_code_not_by_the_session():
    found = review.findings(REPORT)
    ids = lambda picked: [f["id"] for f in picked]
    picked, left = cli.chosen(found, None, has_cases=True)
    assert ids(picked) == ["C1", "C2", "D1", "D2", "D3"] and not left   # the team's cases judge a design change
    picked, left = cli.chosen(found, None, has_cases=False)
    assert ids(picked) == ["C1", "C2"] and "no eval cases" in left[0] and "D1" in left[0]  # no cases, no change to the design
    assert ids(cli.chosen(found, "d1, c1", has_cases=True)[0]) == ["C1", "D1"]  # a fence: these and no others
    assert ids(cli.chosen(found, "D1", has_cases=False)[0]) == []
    with pytest.raises(ValueError, match="no finding C9"):
        cli.chosen(found, "C1,C9", has_cases=True)


def _a_saved_review(tmp_path, state="abc+1"):
    project, out = tmp_path / "p", tmp_path / "out"
    run_dir = out / "runs" / "review-x-p"
    project.mkdir()
    run_dir.mkdir(parents=True)
    (run_dir / "review.md").write_text(REPORT, encoding="utf-8")
    review.remember(out, project, "cmd", state, "review-x", run_dir, review.findings(REPORT))
    return project, out


def _nothing_may_run(monkeypatch, state):
    def never(*a, **k):
        raise AssertionError("nothing may be run or spent here")

    monkeypatch.setattr(cli, "_start", lambda args: setattr(args, "asked_anew", False) or setattr(args, "agent", "a") or "cmd")
    monkeypatch.setattr(runner, "code_state", lambda project: state)
    monkeypatch.setattr(config, "auth_summary", lambda: "test")
    monkeypatch.setattr(review, "run", never)
    monkeypatch.setattr(measure, "collect", never)
    from fleetopt.optimizer import loop
    monkeypatch.setattr(loop, "run", never)


def test_a_review_is_reused_while_the_code_has_not_changed(tmp_path, capsys, monkeypatch):
    project, out = _a_saved_review(tmp_path)
    _nothing_may_run(monkeypatch, "abc+1")
    assert cli.main(["review", str(project), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "has not changed since the review" in text and "C1 - System prompt is never cached" in text
    assert f"fleetopt apply {project}" in text and "D1 (Hand-built agent loop)" in text
    assert "level 3 of 4, wrong shape: 2 cost, 1 design, 2 redesign" in text
    assert review.saved(out, project, "cmd", "abc+2") is None       # the code moved: that review no longer answers
    assert review.saved(out, project, "other agent", "abc+1") is None  # and it was a review of one agent, not the project


def test_apply_stops_before_spending_when_a_named_finding_does_not_exist(tmp_path, capsys, monkeypatch):
    project, out = _a_saved_review(tmp_path)
    _nothing_may_run(monkeypatch, "abc+1")
    assert cli.main(["apply", str(project), "--out", str(out), "--only", "C7"]) == 1
    assert "no finding C7" in capsys.readouterr().out
    assert cli.main(["apply", str(project), "--out", str(out), "--only", "D1"]) == 1  # a redesign, and no cases
    assert "nothing to try" in capsys.readouterr().out


def test_a_capture_that_never_got_its_review_is_not_run_again(tmp_path):
    out, project = tmp_path, tmp_path / "p"
    with store.connect(out / "fleetopt.db") as conn:
        row = dict(project=str(project), run_cmd="cmd", code_state="abc+1", exit_code=0)
        _session(conn, label="review-1", **row)
        _session(conn, label="review-2", **{**row, "exit_code": 1})          # crashed: not evidence to reuse
        _session(conn, label="baseline", **row)                              # a measurement, not a review capture
        _session(conn, label="review-3", **{**row, "run_cmd": "another agent"})
    assert cli._captured(out, project, "cmd", "abc+1") == "review-1"
    assert cli._captured(out, project, "cmd", "abc+2") is None               # the code moved
    assert cli._captured(out, project, "cmd", None) is None                  # not a git repo: nothing to go by
    assert cli._captured(tmp_path / "nowhere", project, "cmd", "abc+1") is None


def test_the_summary_is_computed_and_claims_a_gain_only_under_a_proven_verdict(tmp_path):
    def measured(label, state):
        return {"event": "measure", "label": label, "code_state": state}

    def compared(state, **verdicts):
        return {"event": "compare", "baseline_state": "v1", "candidate_state": state,
                "result": {k: {"verdict": v, "delta_pct": pct} for k, (v, pct) in verdicts.items()}}

    events = [measured("baseline", "v1"), measured("C1", "v2"), measured("C1-n5", "v2"), measured("D1", "v3"),
              compared("v3", cost_usd=("within noise", -14.6), wall_ms=("improved", -23.4), llm_calls=("improved", -33.3))]
    assert session.numbers(events, "PROVEN ON THIS EVIDENCE: ...", "v1", "v3") == (2, ["time -23%", "model calls -33%"])
    assert session.numbers(events, "NOT PROVEN SAFE: ...", "v1", "v3") == (2, [])    # measured, and not a gain
    assert session.numbers(events, "NOTHING LEFT STANDING: ...", "v1", "v1") == (2, [])

    record = {"findings": review.findings(REPORT), "level": 3}
    facts = {"verdict": "NOTHING LEFT STANDING: 1 changed version(s) were judged and undone.", "tried": 3, "kept": 0,
             "gained": [], "branch": "fleetopt/c1-c2-d1", "run_dir": "/runs/x"}
    text = cli.summary("supervisor", record, facts, 27, 1.22, 2.31)
    assert "Level    3 of 4, wrong shape" in text and "Found    2 cost, 1 design, 2 redesign" in text
    assert "3 changed version(s): 0 kept, 3 undone" in text and "Gained   nothing proven" in text
    assert "$1.22 on the team's key in 27 run(s)" in text and "the same code it started from" in text
    assert "not priced" in cli.summary("a", record, facts, 1, None, 0.5)

    out, project = tmp_path, tmp_path / "p"
    with store.connect(out / "fleetopt.db") as conn:
        for label in ("old", "baseline", "C1"):
            sid = _session(conn, project=str(project), label=label, run_cmd="cmd", code_state="v1", exit_code=0)
            conn.execute("INSERT INTO runs (session_id, run_type, model, input_tokens, output_tokens)"
                         " VALUES (?, 'llm', 'claude-haiku-4-5', 1000000, 0)", (sid,))
        _session(conn, project="another project", label="baseline", run_cmd="cmd", code_state="v1", exit_code=0)
    runs, cost = cli.spent(out, project, after=1)
    assert runs == 2 and cost == pytest.approx(2 * measure.session_stats(store.connect(out / "fleetopt.db"), 1)["cost_usd"])


def test_the_loop_keeps_what_passes_undoes_the_rest_and_tries_twice_at_most(tmp_path, monkeypatch):
    from fleetopt.optimizer import loop

    project = tmp_path / "agent"
    project.mkdir()
    (project / "agent.py").write_text("x = 0\n", encoding="utf-8")
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"]):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)

    edits, judged = [], []

    async def edit(project_, finding, review, feedback, model, max_usd, start_branch):
        edits.append((finding["id"], feedback))
        if finding["id"] == "C3":
            return "CANNOT: the finding is wrong about the code", 0.1
        (project_ / "agent.py").write_text(f"x = {len(edits)}\n", encoding="utf-8")
        return f"DONE: change {len(edits)}", 0.1

    def measure(label, max_steps=None, probe=False):
        if label == "C2-2":
            return None, "run 1 took more than 150 steps, far more than the original, and was stopped."
        return {"steps": 40, "completed": 3}, None

    def compare(before, after):
        return {"cost_usd": {"verdict": "within noise" if after.startswith("D1") else "improved", "delta_pct": -20.0}}

    async def judge(task, label):
        judged.append(label)
        ok = label != "C2"
        return ok, [{"kept_on": "unchanged answer" if ok else None, "reason": "an answer was cut off"}]

    for name, fake in (("_edit", edit), ("_measure", measure), ("_compare", compare), ("_judge", judge)):
        monkeypatch.setattr(loop, name, fake)
    monkeypatch.setattr(tools, "say", lambda line: None)
    findings = [{"id": i, "title": i, "kind": "cost"} for i in ("C1", "C2", "C3")] + [{"id": "D1", "title": "D1", "kind": "design"}]
    facts = asyncio.run(loop.run(project, tmp_path / "out", "cmd", "review", findings, task="t"))

    rows = {r["id"]: (r["outcome"], r["detail"]) for r in json.loads(
        (pathlib.Path(facts["run_dir"]) / "run.json").read_text(encoding="utf-8"))["rows"]}
    assert rows["C1"][0] == "kept"
    assert rows["C2"] == ("undone", "run 1 took more than 150 steps, far more than the original, and was stopped.")
    assert rows["C3"] == ("not changed", "the finding is wrong about the code")
    assert rows["D1"] == ("undone", "no real gain")
    assert [e for e in edits if e[0] == "C2"] == [("C2", None), ("C2", "the judge failed 1 of 1 requests: an answer was cut off")]
    assert len([e for e in edits if e[0] == "D1"]) == 2                       # two attempts, never a third
    assert judged == ["C1", "C2"]                                             # nothing is judged that did not gain
    log = subprocess.run(["git", "-C", str(project), "log", "--format=%s"], capture_output=True, text=True).stdout.split("\n")
    assert log[0] == "C1: change 1" and facts["kept"] == 1                    # one commit per kept finding, nothing else
    assert facts["branch"].startswith("fleetopt/")
    assert (project / "agent.py").read_text(encoding="utf-8") == "x = 1\n"   # what was undone is gone
