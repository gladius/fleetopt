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
from fleetopt.evidence import measure
from fleetopt import experts, session, tools
from fleetopt.probe import runner, store

ROOT = pathlib.Path(__file__).resolve().parents[1]


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
    return asyncio.run(session.guard_bash()({"tool_input": {"command": cmd}}, "id", None))


@pytest.mark.parametrize("cmd", [
    "pip install rich", "python -m pip install -q x", "uv run --with rich python x.py", "uv add rich",
    "poetry add rich", "npm install", "npx something", "curl https://example.com -o f", "brew install jq",
    "python agent.py", "pytest tests/", "langgraph dev", "deepeval test run tests/", "promptfoo eval",
    "/proj/.venv/bin/python /x/fleetopt/probe/driver.py e.json --limit 1",
    '.venv/bin/python -c "from agents.sql_agent import build; print(len(build()))"',  # project code, outside the cap
    "uv run python -m agents.main", "poetry run agent", 'py -c "import agent"', r".venv\Scripts\python.exe agent.py",
])
def test_guard_denies_installs_and_running_the_target(cmd):
    assert _guard(cmd)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("cmd", ["git status", "ls -la", "wc -l README.md", "grep -rn StateGraph src", "cat agent.py",
                                 "py -m py_compile agent.py", "poetry env info -p"])
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


# --- small things that each broke a real run once ---------------------------------

def test_output_tail_survives_undecodable_bytes(tmp_path):
    body = b"first\n\xff\xfe not utf-8\n" + b"\n".join(f"l{i}".encode() for i in range(30))
    (tmp_path / "target.log").write_bytes(body)
    assert runner.output_tail(tmp_path, lines=3).splitlines() == ["l27", "l28", "l29"]
    assert runner.output_tail(tmp_path / "missing") == ""


def test_child_env_never_carries_fleetopts_own_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLEETOPT_TEST_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("# comment\nFLEETOPT_TEST_KEY=abc\nOPENAI_API_KEY=theirs\n", encoding="utf-8")
    config.load_env()
    try:
        assert os.environ["FLEETOPT_TEST_KEY"] == "abc"
        assert "OPENAI_API_KEY" not in os.environ      # run from inside a team's project, its .env stays theirs
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
    tools.CTX.clear()
    tools.CTX.update(out=tmp_path / "out", project=_fixture(tmp_path))
    try:
        tools.CTX["reach"] = tools._reach("x")
        assert tools.CTX["reach"][1] == ["billing", "other"]              # what the inputs never reached, by name
        assert tools.reached() == "8 of 10 nodes; never ran: billing, other"
        facts = {"mode": "review", "measured": True, "reach": tools.reached(), "team_cost": 0, "team_runs": 3,
                 "own_cost": 0, "project": "p", "run_dir": str(tmp_path)}
        assert "Reached  8 of 10 nodes; never ran: billing, other" in "\n".join(session.summary(facts))
    finally:
        tools.CTX.clear()


def test_a_project_with_several_graphs_is_judged_by_the_one_that_ran(tmp_path):
    tools.CTX.clear()
    tools.CTX.update(out=tmp_path, project=tmp_path / "a")
    graph = lambda conn, sid, nodes, driven=0: conn.execute(
        "INSERT INTO graphs (session_id, name, nodes, edges, driven) VALUES (?, 'LangGraph', ?, '[]', ?)",
        (sid, json.dumps(["__start__", "__end__", *nodes]), driven))
    with store.connect(tmp_path / "fleetopt.db") as conn:
        old = _session(conn, project=str(tmp_path / "a"), label="before-paths", code_state="v1", exit_code=0)
        new = _session(conn, project=str(tmp_path / "a"), label="with-paths", code_state="v1", exit_code=0)
        for sid in (old, new):
            graph(conn, sid, ["model", "retrieve"]), graph(conn, sid, ["guard", "model", "tools", "block"])
            graph(conn, sid, ["team:__start__", "team:model", "team:tools", "solo:model"])
        for node in ("guard", "model", "tools"):
            conn.execute("INSERT INTO runs (session_id, node) VALUES (?, ?)", (old, node))
        graph(conn, new, ["team:model", "team:tools", "solo:model", "solo:tools"], driven=1)   # the driver said which
        for node, path in (("team", "team"), ("model", "team:model"), ("solo", "solo"), ("tools", "solo:tools")):
            conn.execute("INSERT INTO runs (session_id, node, path) VALUES (?, ?, ?)", (new, node, path))
    try:
        # recorded before paths and the driven graph: the graph with the most nodes that ran
        assert tools._reach("before-paths")[:2] == (["guard", "model", "tools"], ["block"])
        # with them: exact, though 'model' and 'tools' both ran somewhere and 'team' and 'solo' both ran
        assert tools._reach("with-paths")[:2] == (["team:model", "solo:tools"], ["team:tools", "solo:model"])
        assert tools._reach("never-measured") == ([], [], {}) and tools.reached() == ""
    finally:
        tools.CTX.clear()


def test_a_change_to_a_node_the_requests_never_ran_is_refused(tmp_path):
    project = _fixture(tmp_path)
    git = lambda *a: subprocess.run(["git", "-C", str(project), "-c", "user.email=t@t", "-c", "user.name=t", *a],
                                    check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "main"), git("add", "-A"), git("commit", "-qm", "base")
    # a sum goes to the math expert; only the research request asks for a check, so only its tools run
    assert _captured(tmp_path, "nested.py:graph", inputs=("sum of 2 and 3", "check the history of rail"))[1] == 0
    file = project / "nested.py"
    source = file.read_text(encoding="utf-8")

    def changed(old, new):
        git("reset", "-q", "--hard", tools.CTX["start_sha"])
        file.write_text(source.replace(old, new), encoding="utf-8")
        git("commit", "-qam", "a change")
        return tools._unproven()

    tools.CTX.clear()
    tools.CTX.update(out=tmp_path / "out", project=project, start_sha=git("rev-parse", "HEAD"))
    try:
        tools.CTX["reach"] = tools._reach("x")
        assert tools.CTX["reach"][1] == ["math:tools"]                  # research's tools ran; the same name here did not
        assert tools.reached() == "3 of 4 nodes; never ran: math:tools"
        assert changed("(checked with the calculator)", "(checked)") == [("math:tools", "nested.py")]
        assert "math:tools (nested.py), which the requests never ran" in tools._unproven_words(tools._unproven())
        assert changed("(checked against the archive)", "(checked)") == []     # the same node name, in the graph that ran
        assert changed('f"Work out: ', 'f"Compute: ') == []                    # a node that ran
        assert changed('reply="an answer"', 'reply="a reply"') == []           # shared code: not tied to a node
    finally:
        tools.CTX.clear()


# --- the design expert: structure, in numbers ------------------------------------------------

def _shape(tmp_path, graph, inputs, runs=2):
    from fleetopt.evidence import shape

    ids = [_captured(tmp_path, graph, inputs)[0] for _ in range(runs)]
    result = shape.analyze(store.connect(tmp_path / "out" / "fleetopt.db"), ids)
    return result, {(f["kind"], f["node"]) for f in result["findings"]}, shape.render(result)


def test_what_a_graph_declared_is_set_against_what_it_did(tmp_path):
    result, found, text = _shape(tmp_path, "supervisor.py:graph", ("VPN drops every hour", "invoice shows a double charge"))
    assert {("branch_never_taken", "route"), ("fixed_dispatch", "supervisor"), ("constant_rounds", "reflect"),
            ("repeated_identical_reply", "reflect")} <= found           # the three things planted in that fixture
    assert "route: 3 branches declared, never taken in 4 runs of it: billing, other; always goes to technical" in text
    assert "dispatches worker_a -> worker_b -> worker_c -> draft in the same order in 4/4 runs while calling a model" in text
    assert (result["requests"], result["distinct_inputs"]) == (4, 2) and "the evidence is 2 inputs wide" in text


def test_each_nested_graph_is_judged_on_its_own_and_a_sound_design_gets_no_finding(tmp_path):
    _, found, text = _shape(tmp_path, "nested.py:graph", ("sum of 2 and 3", "check the history of rail"))
    assert found == {("branch_never_taken", "math:model")}             # research's tools ran; math's never did
    assert "math:model: 1 branches declared, never taken in 2 runs of it: math:tools" in text
    assert "math: ran 2 times, 2 nodes, 1 branch points" in text
    (tmp_path / "out" / "fleetopt.db").unlink()
    _, found, text = _shape(tmp_path, "nested.py:graph", ("sum of 2 and 3", "check 12 times 12", "the history of rail",
                                                          "check the history of rail"))
    assert not found and "nothing structural stands out" in text       # every branch taken: nothing to report


def test_a_branch_point_that_does_not_declare_its_targets_is_said_so_not_guessed(tmp_path):
    from fleetopt.evidence import shape

    conn = store.connect(tmp_path / "s.db")
    sid = _session(conn, project="p", label="x", code_state="v1", exit_code=0)
    edges = [{"source": "__start__", "target": "pick", "conditional": False},
             {"source": "pick", "target": "__end__", "conditional": True}]      # how an unannotated branch is drawn
    conn.execute("INSERT INTO graphs (session_id, name, nodes, edges, driven) VALUES (?, 'g', ?, ?, 1)",
                 (sid, json.dumps(["__start__", "pick", "a", "b", "__end__"]), json.dumps(edges)))
    for trace in ("t1", "t2"):
        conn.execute("INSERT INTO runs (session_id, run_id, trace_id, name, run_type) VALUES (?, ?, ?, 'g', 'chain')",
                     (sid, trace, trace))
        for step, node in enumerate(("pick", "a"), 1):
            conn.execute("INSERT INTO runs (session_id, parent_run_id, trace_id, name, run_type, node, path, step)"
                         " VALUES (?, ?, ?, ?, 'chain', ?, ?, ?)", (sid, trace, trace, node, node, node, step))
    found = shape.analyze(conn, [sid])["findings"]
    assert [f["kind"] for f in found] == ["targets_not_declared"]
    assert "cannot be counted: read its function. On these runs it went to a" in found[0]["text"]


def test_the_design_expert_only_looks_and_has_the_numbers_on_structure(tmp_path, monkeypatch):
    design = experts.EXPERTS["design"]
    system = design.system()
    assert system.index("## Starting the agent") < system.index("# Your expertise: whether the design fits the job")
    assert all(f"fleetopt:{name}" in system for name in ("patterns", "langgraph")) and "Supervisor / orchestrator" in system
    assert "token and cost waste" not in system and "whether the design fits" not in experts.COST.system()   # one expertise each
    o = session.build_options(ROOT / "fixture", look_only=True, expert=design)
    assert set(o.tools) == {"Read", "Grep", "Glob"} and "mcp__fleetopt__shape" in o.allowed_tools
    assert "mcp__fleetopt__shape" not in session.build_options(ROOT / "fixture", look_only=True).allowed_tools
    assert design.apply is None                                         # it changes nothing, for now
    with pytest.raises(ValueError, match="the design expert only reviews: fleetopt review --expert design"):
        asyncio.run(session.run(_repo(tmp_path), tmp_path / "out", expert="design"))
    project = tmp_path / "agent"
    _fake_runs(monkeypatch)
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry={**ENTRY, "project": str(project)},
                look_only=True, expert=design)
    try:
        assert "measure the agent as it is first" in _call("shape")
        _call("measure")
        assert _call("shape") == "no runs to analyze"                   # nothing recorded by the faked runs
    finally:
        tools.CTX.clear()


# --- the one agent: its guide, its command line ----------------------------------------------

def test_an_expert_is_the_shared_guide_its_own_and_every_skill_in_its_folder():
    assert config.SETTING_SOURCES == []
    cost = experts.EXPERTS["cost"]
    on_disk = {p.name for p in (ROOT / "fleetopt" / "experts" / "cost" / "skills").iterdir() if p.is_dir()}
    assert on_disk == set(cost.skills) == {"caching", "handoffs", "model-tier", "prompt-growth", "redundant-work", "tool-surface"}
    assert all(f"fleetopt:{name}" in cost.system() for name in cost.skills)
    system = cost.system()
    assert system.index("## Starting the agent") < system.index("# Your expertise: token and cost waste")  # shared, then its own
    assert "minimum prefix" in system and "## The flow" in system
    assert cost.apply and cost.earned is experts.gain and cost.keeps_shape   # what its changes must earn


def test_while_it_investigates_a_person_sees_what_it_is_looking_at(tmp_path, capsys):
    from claude_agent_sdk import ToolUseBlock

    shown = {}
    for name, args in (("Read", {"file_path": str(tmp_path / "agent.py")}), ("Read", {"file_path": str(tmp_path / "agent.py")}),
                       ("mcp__fleetopt__query", {"sql": "SELECT 1"}), ("mcp__fleetopt__query", {"sql": "SELECT 2"}),
                       ("Grep", {"pattern": "x"}), ("mcp__fleetopt__measure", {}),
                       ("Edit", {"file_path": str(tmp_path / "agent.py")})):
        session._activity(ToolUseBlock(id="t", name=name, input=args), tmp_path, shown)
    assert capsys.readouterr().out.split("\n")[:-1] == [
        "  reading agent.py", "  looking at the recorded calls", "  searching the code", "  editing agent.py"]


def test_the_report_is_what_is_printed_and_nothing_said_before_it():
    said = ("Confirmed: repo is back to the original state.\n\nI ran 4 differing inputs total...\n\n"
            "**What it is for:** research.\n**What it spends:** $0.0086 a run")
    assert session._report_only(said) == "**What it is for:** research.\n**What it spends:** $0.0086 a run"
    assert session._report_only("no report heading at all") == "no report heading at all"
    assert session._sentence("The research step re-sends every note. I will trim it.") == \
        "The research step re-sends every note."
    assert session._sentence("## Plan\nfirst trim") == "Plan"


def test_the_command_line_is_two_commands_and_a_few_flags():
    parse = cli._parser().parse_args
    assert parse(["apply", "repo"]).out == ".fleetopt" and parse(["apply", "repo"]).max_usd == 5.0
    assert parse(["apply", "repo", "--out", "after"]).out == "after"
    assert parse(["--out", "before", "review", "repo"]).out == "before"
    args = parse(["review", "repo", "--evals", "cases.jsonl", "--graph", "supervisor"])
    assert (args.cmd, args.evals, args.graph, args.max_usd) == ("review", "cases.jsonl", "supervisor", 2.0)
    assert parse(["review", "repo"]).expert == "cost" and not parse(["review", "repo"]).no_ask
    assert parse(["review", "repo", "--expert", "design"]).expert == "design"
    assert parse(["apply", "repo", "--expert", "cost", "--no-ask"]).no_ask
    for gone in (["apply", "repo", "--only", "C1"], ["review", "repo", "--design"], ["capture", "repo"],
                 ["review", "repo", "--expert", "nobody"]):
        with pytest.raises(SystemExit):
            parse(gone)


def test_an_expert_that_only_reviews_is_never_asked_to_change_anything(tmp_path, monkeypatch):
    looks = experts.Expert(name="cost", does="looks", review="Look at it.")
    monkeypatch.setitem(experts.EXPERTS, "looks", looks)
    monkeypatch.setitem(session.EXPERTS, "looks", looks)
    with pytest.raises(ValueError, match="only reviews: fleetopt review --expert cost"):
        asyncio.run(session.run(_repo(tmp_path), tmp_path / "out", expert="looks"))
    facts = {"mode": "review", "expert": "cost", "changes_it": False, "measured": True, "team_cost": 0, "team_runs": 3,
             "own_cost": 0, "project": "p", "run_dir": str(tmp_path)}
    assert "fleetopt apply" not in "\n".join(session.summary(facts))       # nothing to run next
    assert "Next     fleetopt apply p" in "\n".join(session.summary({**facts, "changes_it": True}))


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


def _fake_runs(monkeypatch, failing=(), flat=(), broke=(), reshaped=()):
    """Running the agent, comparing, the team's evals and the graph's structure, faked by the
    change saved last. Nothing runs and no model is called."""
    seen = {"measured": [], "evals": [], "read": []}
    last = lambda: tools.CTX["saved"][-1] if tools.CTX["saved"] else ""

    def run(label, max_steps=None, probe=False):
        seen["measured"].append((label, max_steps, probe))
        if last() in failing:
            return None, "run 1 took more than 150 steps, far more than the original, and was stopped."
        return {"steps": 40, "completed": 3, "cost_usd": 0.01, "llm_calls": 5}, None

    def evals(command):
        seen["evals"].append(command)
        tools.CTX["n_evals"] += 1
        return 0, 1.0, f"4 passed ({last() or 'the original'})"

    async def read(command, before, after):
        seen["read"].append(last())
        return (False, "test_summary[cold-chain]: the summary no longer names the topic") if last() in broke else (True, "")

    monkeypatch.setattr(tools, "_measure", run)
    monkeypatch.setattr(tools, "_run_evals", evals)
    monkeypatch.setattr(tools, "_read_evals", read)
    monkeypatch.setattr(tools, "_evals_called_a_model", lambda: True)   # a real suite, which runs the agent
    monkeypatch.setattr(tools, "_compare", lambda before, after: {"cost_usd": {
        "before": 0.0125, "after": 0.01, "delta_pct": -20.0, "verdict": "within noise" if last() in flat else "improved"}})
    monkeypatch.setattr(tools, "_shape", lambda label: ["changed"] if not label.startswith("baseline") and last() in reshaped
                        else ["same"])
    return seen


def _call(tool_name, **args):
    return asyncio.run(getattr(tools, tool_name).handler(args))["content"][0]["text"]


def test_the_tools_keep_what_earns_it_on_the_teams_evals_and_undo_the_rest(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    seen = _fake_runs(monkeypatch, failing={"loop forever"}, flat={"trim notes"}, broke={"smaller model"},
                      reshaped={"merge two steps"})
    edit = lambda text, name="agent.py": (project / name).write_text(text, encoding="utf-8")
    git = lambda *a: subprocess.run(["git", "-C", str(project), *a], capture_output=True, text=True, check=True).stdout
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry={**ENTRY, "project": str(project)},
                run_dir=tmp_path / "run")
    git("checkout", "-q", "-b", "fleetopt/test")
    try:
        assert "nothing is installed" in _call("run_evals", command="pip install deepeval && pytest")
        assert "Exit 0" in _call("run_evals", command="pytest evals")      # what passes today
        assert "measure the agent as it is first" in _call("save_change", name="too early")
        assert "Edits are allowed now" in _call("measure")                 # the agent as it is, once
        assert "Nothing saved" in _call("measure")

        edit("x = 1\n"), _call("save_change", name="cache the system prompt")   # a bundle of two, measured once
        edit("y = 1\n", "helper.py"), _call("save_change", name="bound the output")
        assert "got better past the noise" in _call("measure")
        assert "run the team's evals on this code first" in _call("keep")  # never kept on the numbers alone
        assert "the same command" in _call("run_evals", command="pytest")
        _call("run_evals", command="pytest evals")
        assert "Kept" in _call("keep")

        edit("x = 2\n"), edit("z = 2\n", "scratch.py"), _call("save_change", name="loop forever")
        assert "could not be measured" in _call("measure")                 # it ran away and was stopped
        assert "Undone" in _call("undo", why="it looped")
        assert not (project / "scratch.py").exists()                       # nothing half-made is left

        edit("x = 3\n"), _call("save_change", name="trim notes"), _call("measure")
        assert "nothing got better past the noise" in _call("keep")
        _call("undo", why="")
        edit("x = 4\n"), _call("save_change", name="smaller model"), _call("measure"), _call("run_evals", command="pytest evals")
        assert "the team's evals: test_summary[cold-chain]" in _call("keep")
        _call("undo", why="")
        edit("x = 5\n"), _call("save_change", name="merge two steps"), _call("measure")
        assert "nodes or edges" in _call("keep")                            # design is not this run's to change
        _call("undo", why="")
        edit("x = 6\n")
        assert "unsaved edits" in _call("measure")
        (project / "evals").mkdir(), edit("def test_x():\n    pass\n", "evals/test_summary.py")
        _call("save_change", name="loosen a test"), _call("measure"), _call("run_evals", command="pytest evals")
        assert "changes the team's tests or evals (evals/test_summary.py)" in _call("keep")
        tools.finish()                                                     # left unproven: undone for it

        changes = {k: v[0] for k, v in tools.CTX["changes"].items()}
        assert changes == {"cache the system prompt": "kept", "bound the output": "kept", "loop forever": "undone",
                           "trim notes": "undone", "smaller model": "undone", "merge two steps": "undone",
                           "loosen a test": "undone"}
        assert "150 steps" in tools.CTX["changes"]["loop forever"][1]
        assert seen["read"] == ["bound the output", "smaller model"]      # read only for what gained, on its own run
        assert seen["measured"][0][0].startswith("baseline-") and seen["measured"][0][1:] == (None, False)
        assert all(probe and steps == 150 for _, steps, probe in seen["measured"][1:])  # changed code is watched
        assert git("log", "--format=%s").split("\n")[:3] == ["bound the output", "cache the system prompt", "base"]
        assert (project / "agent.py").read_text(encoding="utf-8") == "x = 1\n" and (project / "helper.py").exists()
        assert (tmp_path / "run" / "evals-1.log").read_text(encoding="utf-8").startswith("4 passed")  # every run kept

        tools.CTX["deadline"] = 1
        edit("x = 9\n"), _call("save_change", name="late")
        assert "time limit" in _call("measure")
    finally:
        tools.CTX.clear()


def test_what_one_expert_found_reaches_the_others_apply(tmp_path, monkeypatch):
    import claude_agent_sdk

    project = _repo(tmp_path)
    _fake_runs(monkeypatch), _tried(monkeypatch)
    reviews = tmp_path / "out" / "reviews"
    reviews.mkdir(parents=True)
    key = session.hashlib.sha1(str(project).encode()).hexdigest()[:8]
    (reviews / f"{project.name}-{key}-{runner.code_state(project)}-design.md").write_text(
        "What it is for: x\nFor the cost expert:\n- the specialists' closing call only rewords\n", encoding="utf-8")
    seen = {}

    async def the_agent(prompt, options):
        seen["prompt"] = prompt
        return
        yield

    monkeypatch.setattr(claude_agent_sdk, "query", the_agent)
    try:
        asyncio.run(session.run(project, tmp_path / "out"))
    finally:
        tools.CTX.clear()
    assert "The design expert reviewed this exact code earlier:" in seen["prompt"]
    assert "the specialists' closing call only rewords" in seen["prompt"]


def test_evals_that_did_not_run_or_called_no_model_are_not_the_proof(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    seen = _fake_runs(monkeypatch)
    monkeypatch.setattr(tools, "_run_evals", lambda command: (127, 0.1, "/bin/sh: line 1: pytest: command not found"))
    read = []

    async def answers(pairs):
        read.append("answers")
        return True, ""

    monkeypatch.setattr(tools, "_read_answers", answers)
    monkeypatch.setattr(tools, "_pairs", lambda label: [{"input": "q", "expected": None, "before": "a", "after": "b"}])
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry={**ENTRY, "project": str(project)},
                run_dir=tmp_path / "run")
    subprocess.run(["git", "-C", str(project), "checkout", "-q", "-b", "fleetopt/test"], check=True)
    try:
        text = _call("run_evals", command="pytest tests")
        assert "Not recorded: the command did not run (exit 127: /bin/sh: line 1: pytest: command not found)" in text
        assert tools.CTX["evals_before"] is None                                   # observed: kept on two "not found"s
        _call("measure")
        monkeypatch.setattr(tools, "_run_evals", lambda command: (0, 1.0, "3 passed"))
        _call("run_evals", command="pytest tests")                                # a suite that drives a fake model
        monkeypatch.setattr(tools, "_evals_called_a_model", lambda: False)
        (project / "agent.py").write_text("x = 1\n", encoding="utf-8")
        _call("save_change", name="trim the prompt"), _call("measure"), _call("run_evals", command="pytest tests")
        assert "Kept" in _call("keep") and read == ["answers"]                     # the answers were read as well
        assert tools.CTX["changes"]["trim the prompt"][1].endswith("the team's evals pass as before, and answers as good as before")
        assert seen["read"] == ["trim the prompt"]
    finally:
        tools.CTX.clear()


def test_a_fake_model_in_the_teams_evals_is_not_a_model_call_and_costs_nothing(tmp_path):
    tools.CTX.clear()
    tools.CTX.update(out=tmp_path, project=tmp_path / "a", first_session=0)
    with store.connect(tmp_path / "fleetopt.db") as conn:
        fake = _session(conn, project=str(tmp_path / "a"), label="evals-1", code_state="v1", exit_code=0)
        conn.execute("INSERT INTO runs (session_id, run_type, provider, input_tokens, output_tokens) VALUES (?, 'llm', 'faketoolmodel', 50, 5)", (fake,))
        conn.commit()
        assert not tools._evals_called_a_model()                                  # observed: counted as a model call
        assert measure.session_stats(conn, fake)["cost_usd"] == 0.0                # and made the whole run "not priced"
        real = _session(conn, project=str(tmp_path / "a"), label="evals-2", code_state="v1", exit_code=0)
        conn.execute("INSERT INTO runs (session_id, run_type, model, provider, input_tokens, output_tokens) VALUES (?, 'llm', 'gpt-5-nano', 'openai', 50, 5)", (real,))
        conn.commit()
        assert tools._evals_called_a_model()
        odd = _session(conn, project=str(tmp_path / "a"), label="x", code_state="v1", exit_code=0)
        conn.execute("INSERT INTO runs (session_id, run_type, model, input_tokens, output_tokens) VALUES (?, 'llm', 'some-new-model', 50, 5)", (odd,))
        assert measure.session_stats(conn, odd)["cost_usd"] is None                # a named model with no price stays unpriced
    tools.CTX.clear()


def test_one_change_is_dropped_by_name_and_the_rest_stay_saved(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    _fake_runs(monkeypatch)
    edit = lambda text, name="agent.py": (project / name).write_text(text, encoding="utf-8")
    git = lambda *a: subprocess.run(["git", "-C", str(project), *a], capture_output=True, text=True, check=True).stdout
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry={**ENTRY, "project": str(project)},
                run_dir=tmp_path / "run")
    git("checkout", "-q", "-b", "fleetopt/test")
    try:
        _call("measure")
        edit("x = 1\n"), _call("save_change", name="cache the system prompt")
        edit("y = 1\n", "helper.py"), _call("save_change", name="bound the output")
        edit("z = 1\n", "other.py"), _call("save_change", name="smaller model")
        _call("measure")
        assert "no saved change is named 'trim notes'" in _call("undo", why="", names=["trim notes"])
        text = _call("undo", why="it reads worse", names=["bound the output"])
        assert "Still saved: cache the system prompt, smaller model. Measure them again." in text
        assert not (project / "helper.py").exists() and (project / "other.py").exists()
        assert git("log", "--format=%s").split("\n")[:3] == ["smaller model", "cache the system prompt", "base"]
        assert tools.CTX["changes"]["bound the output"] == ["undone", "it reads worse"]
        assert "Nothing" not in _call("measure") and "Kept" not in _call("keep")   # measured again; evals not run: refused
        _call("undo", why="start over")

        edit("x = 1\n"), _call("save_change", name="first edit of a line")      # the second is built on the first
        edit("x = 2\n"), _call("save_change", name="second edit of the same line")
        assert "everything is undone" in _call("undo", why="", names=["first edit of a line"])
        assert (project / "agent.py").read_text(encoding="utf-8") == "x = 0\n" and not tools.CTX["saved"]
        assert tools.CTX["changes"]["second edit of the same line"][1] == "it does not fit without a change that was undone"
        edit("x = 3\n"), _call("save_change", name="one"), edit("y = 3\n", "b.py"), _call("save_change", name="two")
        edit("y = 4\n", "b.py")                                                # not saved yet
        assert "unsaved edits" in _call("undo", why="", names=["one"])          # dropping one must not lose them
    finally:
        tools.CTX.clear()


def test_a_question_goes_to_someone_at_the_terminal_only_while_getting_started(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    _fake_runs(monkeypatch)
    typed = iter(["the one in langgraph.json", "", None])
    monkeypatch.setattr(tools, "_typed", lambda seconds: next(typed))
    entry_file = tmp_path / "out" / "e.json"
    tools.begin(project, tmp_path / "out", entry_file=entry_file, entry={**ENTRY, "project": str(project)})
    try:
        assert "No one is there to ask" in _call("ask", question="Which graph do you ship?")   # no terminal: never asked
        tools.CTX["ask"] = True
        assert "They answered: the one in langgraph.json" in _call("ask", question="Which graph do you ship?")
        assert json.loads(tools.told_path(entry_file).read_text(encoding="utf-8")) == [
            {"question": "Which graph do you ship?", "answer": "the one in langgraph.json"}]
        assert "No answer came" in _call("ask", question="Is data/x.csv your test data?") and tools.CTX["ask"]   # skipped
        assert "No answer came" in _call("ask", question="Which env file?") and not tools.CTX["ask"]             # nobody there
        tools.CTX.update(ask=True, asked=0)
        for _ in range(tools.ASKS):
            monkeypatch.setattr(tools, "_typed", lambda seconds: "yes")
            _call("ask", question="Again?")
        assert "3 questions, the most a run asks" in _call("ask", question="One more?")
        tools.CTX["asked"] = 0
        _call("measure")
        assert "questions are for getting the agent started" in _call("ask", question="Should I keep it?")
        told = session._told(tools.told_path(entry_file))                      # the next run is told, and need not ask
        prompt = session._prompt(experts.COST, True, None, None, None, 2.0, 120, 2.0, None, None, told)
        assert "The team was asked before: Which graph do you ship? They answered: the one in langgraph.json" in prompt
    finally:
        tools.CTX.clear()


def test_an_agent_whose_model_calls_cannot_be_seen_is_said_so_and_nothing_more_is_spent(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    runs = []
    monkeypatch.setattr(tools, "_measure", lambda label, max_steps=None, probe=False: runs.append(label) or (
        {"steps": 12, "completed": 2, "llm_calls": 0}, None))
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry={**ENTRY, "project": str(project)},
                look_only=True)
    try:
        assert "without LangChain" in _call("measure")
        assert "Refused" in _call("measure") and len(runs) == 1 and runs[0].startswith("baseline-")
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
                                              "inputs": ["battery degradation", "route optimization", "cold chain"]}))
        await call("run_evals", command="pytest evals")
        await call("measure")
        (project / "agent.py").write_text("x = 1\n", encoding="utf-8")
        await call("save_change", name="cache the system prompt")
        await call("measure")
        await call("run_evals", command="pytest evals")
        await call("keep")
        (project / "agent.py").write_text("x = 2\n", encoding="utf-8")
        await call("save_change", name="trim notes")
        await call("measure")
        return
        yield

    monkeypatch.setattr(claude_agent_sdk, "query", the_agent)
    try:
        facts = asyncio.run(session.run(project, tmp_path / "out"))
    finally:
        tools.CTX.clear()
    git = lambda *a: subprocess.run(["git", "-C", str(project), *a], capture_output=True, text=True).stdout.strip()
    assert "How to start it is not known yet" in seen["prompt"]
    assert {"mcp__fleetopt__keep", "mcp__fleetopt__run_evals"} <= set(seen["tools"])
    assert facts["kept"] == 1 and facts["branch"].startswith("fleetopt/")
    assert [(c["name"], c["outcome"]) for c in facts["changes"]] == [("cache the system prompt", "kept"),
                                                                    ("trim notes", "undone")]
    assert git("rev-parse", "--abbrev-ref", "HEAD") == "main"                # the team's copy is where it was
    assert git("show", f"{facts['branch']}:agent.py") == "x = 1"             # the branch holds what was kept
    text = "\n".join(session.summary(facts))
    assert "1 change(s) kept on branch fleetopt/" in text and "the team's evals (pytest evals), before and after" in text
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
        await tools.start.handler({"entry": json.dumps({"graph": "agent.py:graph", "inputs": ["x", "y", "z"]})})
        await tools.measure.handler({})
        yield claude_agent_sdk.ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                                             num_turns=3, session_id="s", total_cost_usd=0.1,
                                             result="Worth changing:\n- cache the system prompt: 0 cache reads")

    monkeypatch.setattr(claude_agent_sdk, "query", the_agent)
    try:
        facts = asyncio.run(session.run(project, tmp_path / "out", look_only=True))
    finally:
        tools.CTX.clear()
    assert not seen["tools"] & {"Bash", "Edit", "Write", "mcp__fleetopt__save_change", "mcp__fleetopt__keep",
                                "mcp__fleetopt__undo", "mcp__fleetopt__run_evals"}
    assert facts["branch"] is None and facts["kept"] == 0
    branches = subprocess.run(["git", "-C", str(project), "branch"], capture_output=True, text=True).stdout
    assert "fleetopt" not in branches
    reviews = list((tmp_path / "out" / "reviews").glob("*.md"))
    assert len(reviews) == 1 and "cache the system prompt" in reviews[0].read_text(encoding="utf-8")
    assert "Next     fleetopt apply" in "\n".join(session.summary(facts))


def test_a_stopped_run_puts_the_teams_copy_back(tmp_path, monkeypatch, capsys):
    import claude_agent_sdk

    project = _repo(tmp_path)
    _fake_runs(monkeypatch)
    _tried(monkeypatch)

    async def stopped_midway(prompt, options):  # Ctrl-C after a change was saved, before it was proven
        await tools.start.handler({"entry": json.dumps({"graph": "agent.py:graph", "inputs": ["x", "y", "z"]})})
        await tools.run_evals.handler({"command": "pytest evals"})
        await tools.measure.handler({})
        (project / "agent.py").write_text("x = 1\n", encoding="utf-8")
        await tools.save_change.handler({"name": "trim notes"})
        raise KeyboardInterrupt
        yield

    monkeypatch.setattr(claude_agent_sdk, "query", stopped_midway)
    try:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(session.run(project, tmp_path / "out"))
    finally:
        tools.CTX.clear()
    git = lambda *a: subprocess.run(["git", "-C", str(project), *a], capture_output=True, text=True).stdout.strip()
    assert git("rev-parse", "--abbrev-ref", "HEAD") == "main" and "fleetopt" not in git("branch")
    assert (project / "agent.py").read_text(encoding="utf-8") == "x = 0\n"
    assert "your copy is back on main" in capsys.readouterr().out


def test_a_saved_start_is_used_again_even_with_one_input(tmp_path, monkeypatch):
    import claude_agent_sdk

    project = _repo(tmp_path)
    path = tools.entry_path(tmp_path / "out", project)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**ENTRY, "project": str(project), "proven": "earlier"}), encoding="utf-8")  # 1 input
    seen = {}

    async def the_agent(prompt, options):
        seen["prompt"] = prompt
        return
        yield

    monkeypatch.setattr(claude_agent_sdk, "query", the_agent)
    try:
        asyncio.run(session.run(project, tmp_path / "out"))
    finally:
        tools.CTX.clear()
    assert "How to start it is known" in seen["prompt"]            # one example is enough for a simple agent


def test_runs_of_the_same_code_on_other_inputs_are_never_pooled(tmp_path):
    tools.CTX.clear()
    tools.CTX["entry_path"] = tmp_path / "e.json"
    try:
        four = {**ENTRY, "inputs": ["a", "b", "c", "d"]}
        tools.started(four)
        base = tools.CTX["base"]
        tools.started({**four, "inputs": ["a"]})
        assert tools.CTX["base"] != base                                   # observed: 1 and 4 inputs, one median
        tools.started({**four, "config": {"tenant_id": "t2"}})
        assert tools.CTX["base"] != base                                   # nor another setting
        tools.started({**four, "proven": "later", "job": "reworded"})
        assert tools.CTX["base"] == base and tools.CTX["kept_label"] == base  # the same run: reused, not rerun
    finally:
        tools.CTX.clear()


def test_the_file_the_expected_answers_come_from_counts_as_the_teams_evals(tmp_path):
    project = _repo(tmp_path)
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", run_dir=tmp_path / "r",
                entry={**ENTRY, "project": str(project), "expected": ["a"], "expected_from": "metrics/golden.csv"})
    try:
        (project / "metrics").mkdir()
        (project / "metrics" / "golden.csv").write_text("q,a\n", encoding="utf-8")
        tools._git("add", "-A")
        tools._git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
        assert tools._tests_touched() == ["metrics/golden.csv"]      # keep refuses it, wherever the file lives
    finally:
        tools.CTX.clear()


def test_the_teams_evals_run_in_their_own_setup_and_are_recorded(tmp_path):
    project = _repo(tmp_path)
    (project / ".env").write_text("EVAL_SECRET=from-their-env-file\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "x",
                    "--allow-empty"], check=True, capture_output=True)
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", run_dir=tmp_path / "run",
                entry={**ENTRY, "project": str(project), "env_file": ".env"})
    try:
        text = _call("run_evals", command='python -c "import os; print(\'3 passed\', os.environ[\'EVAL_SECRET\'])"')
        assert "Exit 0" in text and "3 passed from-their-env-file" in text   # loaded by their shell, not by fleetopt
        assert tools.CTX["evals_before"]["exit"] == 0
        with store.connect(tmp_path / "out" / "fleetopt.db") as conn:     # recorded, so it counts toward the cap
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE label = 'evals-1'").fetchone()[0] == 1
    finally:
        tools.CTX.clear()


def test_without_a_suite_the_answers_are_the_proof(tmp_path, monkeypatch):
    project = _repo(tmp_path)
    _fake_runs(monkeypatch)
    verdicts = iter([(True, ""), (False, "INC017302340: the root cause is gone from the answer")])
    seen = []

    async def read(pairs):
        seen.append(pairs)
        return next(verdicts)

    monkeypatch.setattr(tools, "_read_answers", read)
    monkeypatch.setattr(tools, "_pairs", lambda label: [{"input": "INC017302340", "expected": None, "before": "a",
                                                        "after": "b"}])
    entry = {**ENTRY, "project": str(project), "inputs": ["INC017302340", "INC017303854", "INC017302993"],
             "inputs_from": "the team's test incidents"}
    tools.begin(project, tmp_path / "out", entry_file=tmp_path / "out" / "e.json", entry=entry, run_dir=tmp_path / "r")
    subprocess.run(["git", "-C", str(project), "checkout", "-q", "-b", "fleetopt/test"], check=True)
    try:
        assert "Edits are allowed now" in _call("measure")               # no suite: nothing blocks the baseline
        (project / "agent.py").write_text("x = 1\n", encoding="utf-8")
        _call("save_change", name="trim the context"), _call("measure")
        assert "Kept" in _call("keep") and "answers as good as before" in tools.CTX["changes"]["trim the context"][1]
        (project / "agent.py").write_text("x = 2\n", encoding="utf-8")
        _call("save_change", name="smaller model"), _call("measure")
        assert "the answers: INC017302340: the root cause is gone" in _call("keep")
        assert tools.proof() == ("3 example requests from the team's test incidents, answers compared with the "
                                 "original's (no expected answers)")
    finally:
        tools.CTX.clear()


def test_answers_are_paired_request_by_request_with_what_the_team_expects(tmp_path):
    tools.CTX.clear()
    tools.CTX.update(out=tmp_path, project=tmp_path / "a", base="baseline-x",
                     entry={"expected": ["refund in 14 days", None]})
    with store.connect(tmp_path / "fleetopt.db") as conn:
        for label, answers in (("baseline-x", ["14 days", "ok"]), ("baseline-x", ["a fortnight", "fine"]),
                               ("change-1-x", ["two weeks", None])):
            sid = _session(conn, project=str(tmp_path / "a"), label=label, code_state="v1", exit_code=0)
            for i, answer in enumerate(answers):
                conn.execute("INSERT INTO runs (session_id, inputs, outputs, error, start_time) VALUES (?, ?, ?, ?, ?)",
                             (sid, f"q{i}", answer, None if answer else "TimeoutError()", f"t{i}"))
    try:
        pairs = tools._pairs("change-1-x")
        assert [(p["input"], p["expected"], p["before"]) for p in pairs] == [("q0", "refund in 14 days", "14 days"),
                                                                             ("q1", None, "ok")]
        assert pairs[1]["after"].startswith("(did not finish")               # an unfinished answer is shown as one
        assert [p["again"] for p in pairs] == ["a fortnight", "fine"]        # the original's own variation, beside it
    finally:
        tools.CTX.clear()


def test_the_reader_reads_each_request_whole_beside_the_originals_second_run(monkeypatch):
    from fleetopt.evidence import judge

    seen = []

    async def ask(prompt, model=None):
        seen.append(prompt)
        return {"broke": "restart the valve" in prompt, "reason": "the fix now names another part"}

    monkeypatch.setattr(judge, "_ask", ask)
    state = json.dumps({"messages": ["the request", "tool output " * 1000, "FINAL ANSWER: restart the pump"]})
    pairs = [{"input": "INC-1", "expected": None, "before": state, "again": state, "after": state},
             {"input": "INC-2", "expected": None, "before": state, "again": state, "after": state.replace("pump", "valve")},
             {"input": "INC-3", "expected": None, "before": "a", "after": "b"}]
    held, why = asyncio.run(judge.compare_answers("triage", pairs))
    assert len(seen) == 3 and all(p.count("--- ") == 0 and "INPUT\nINC-" in p for p in seen)   # one call a request
    assert seen[0].count("FINAL ANSWER: restart the pump") == 3 and "characters left out" not in seen[0]   # read whole
    assert "BEFORE AGAIN" in seen[0].split("WHAT THE AGENT IS FOR")[1]
    assert "BEFORE AGAIN" not in seen[2].split("WHAT THE AGENT IS FOR")[1]   # one run of the original: shown as before
    assert held is False and why == "request 2 (INC-2): the fix now names another part"   # named by code
    assert asyncio.run(judge.compare_answers("triage", pairs[:1])) == (True, "1 read, none worse")
    huge = "x" * (judge.ANSWER_CHARS + 5) + "FINAL ANSWER: restart the pump"
    asyncio.run(judge.compare_answers("triage", [{"input": "q", "expected": None, "before": huge, "after": "b"}]))
    assert "characters left out" in seen[-1] and "FINAL ANSWER: restart the pump" in seen[-1]   # past the limit: both ends
    assert judge.clipped([{"before": huge, "again": None, "after": "b"}]) == 1                   # and the report says so


def test_the_agent_runs_on_whatever_model_this_setup_has(tmp_path, monkeypatch, capsys):
    import claude_agent_sdk

    o = session.build_options(ROOT / "fixture")
    assert (o.model, o.fallback_model) == ("sonnet", "opus")              # aliases, not version numbers
    assert session.build_options(ROOT / "fixture", model="opus").fallback_model == "sonnet"
    project = _repo(tmp_path)

    async def the_agent(prompt, options):
        yield claude_agent_sdk.SystemMessage(subtype="init", data={"model": "claude-sonnet-9-20270101"})

    monkeypatch.setattr(claude_agent_sdk, "query", the_agent)
    try:
        asyncio.run(session.run(project, tmp_path / "out"))
    finally:
        tools.CTX.clear()
    assert "model: claude-sonnet-9-20270101" in capsys.readouterr().out   # what it picked is shown
