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
from fleetopt.evidence import evals, measure
from fleetopt.optimizer import session, tools
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

    cases, notes = evals.load(tmp_path)
    assert sorted(c["expected"] for c in cases) == ["4", "Jupiter", "Paris", "Shakespeare"]
    assert all(".venv" not in c["source"] for c in cases)
    assert len(notes) == 3


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
    with pytest.raises(RuntimeError, match="different number"):
        asyncio.run(judge_mod.judge_sessions(conn, "qa", base, captured([("x", "y")])))


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
         "capture", str(target), "--run", f"{sys.executable} agent.py", "--out", str(tmp_path / "out")],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert (tmp_path / "out" / "fleetopt.db").exists()
    runs = re.search(r"(\d+) runs, (\d+) graphs", out.stdout)
    assert runs and int(runs.group(1)) > 0 and int(runs.group(2)) == 1, out.stdout


# --- the second fixture ------------------------------------------------------------------

def _capture_fixture(tmp_path, script, *extra):
    target = tmp_path / "fixture"
    if not target.exists():
        shutil.copytree(ROOT / "fixture", target)
        for args in (["init", "-q"], ["add", "-A"],
                     ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"]):
            subprocess.run(["git", "-C", str(target), *args], check=True, capture_output=True)
    return subprocess.run(
        [sys.executable, "-c", "from fleetopt.cli import main; main()", "capture", str(target),
         "--run", f"{sys.executable} {script}", "--out", str(tmp_path / "out"), *extra],
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
    assert parse(["optimize", "repo"]).out == ".fleetopt"
    assert parse(["optimize", "repo", "--out", "after"]).out == "after"
    assert parse(["--out", "before", "optimize", "repo"]).out == "before"
    assert parse(["--out", "before", "capture", "repo", "--run", "x", "--out", "after"]).out == "after"
