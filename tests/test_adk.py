"""An ADK agent tree is started, recorded and read the same way as a LangGraph graph.

The fixture's model is a fake: no network, no cost.
"""

import json
import pathlib
import shutil
import subprocess
import sys

from fleetopt import census, session, tools
from fleetopt.evidence import measure, shape
from fleetopt.probe import runner, store
from tests.test_start import _trying

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _project(tmp_path):
    target = tmp_path / "adk"
    shutil.copytree(ROOT / "fixture-adk", target, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q"], cwd=target, check=True)
    subprocess.run(["git", "add", "-A"], cwd=target, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"], cwd=target, check=True)
    return target.resolve()


def test_the_census_lists_an_adk_root_agent_and_the_run_starts_from_it(tmp_path):
    project = _project(tmp_path)
    taken = census.take(project)
    assert taken["adk_agents"] == ["agent.py:root_agent"] and taken["builders"] == []
    assert "ADK agents (what `adk run` starts: an app, else a root_agent): agent.py:root_agent" in census.render(taken)
    assert tools.declared(project) == ["agent.py:root_agent"]
    # the surveyor's proven entry for it is the one a later expert starts from
    out = tmp_path / "out"
    trying = _trying(project, tmp_path)
    try:
        text = trying({"graph": "agent.py:root_agent", "agent": "fleet",
                       "inputs": ["why is battery degradation up this quarter"]})
        assert "It started" in text and "9 model call(s) seen" in text
        path, entry = session._entry_for(out, project, None, taken)
        assert entry["graph"] == "agent.py:root_agent" and entry["proven"]
    finally:
        tools.CTX.clear()


def test_an_adk_tree_is_recorded_as_nested_graphs_and_its_shape_read_like_a_graphs(tmp_path):
    project = _project(tmp_path)
    out = project / ".fleetopt"
    out.mkdir()
    entry = {"project": str(project), "graph": "agent.py:root_agent", "name": "fleet", "interpreter": sys.executable,
             "inputs": [json.loads(line)["input"] for line in (project / "inputs.jsonl").read_text().splitlines()]}
    path = out / "entry.json"
    path.write_text(json.dumps(entry), encoding="utf-8")
    sid, code, n_runs, n_graphs = runner.run(project, tools.command(path, entry), out, with_io=True, label="t")
    assert (code, n_graphs) == (0, 1) and n_runs > 20
    with store.connect(out / "fleetopt.db") as conn:
        stats = measure.session_stats(conn, sid)
        assert stats["llm_calls"] == 18 and stats["completed"] == 2 and stats["input_tokens"] > 10_000
        paths = {r[0] for r in conn.execute("SELECT DISTINCT path FROM runs WHERE session_id = ? AND run_type = 'llm'", (sid,))}
        assert paths == {"coordinator", "pipeline:draft", "pipeline:review_loop:critic", "pipeline:review_loop:reviser"}
        tool_calls = [r[0] for r in conn.execute("SELECT name FROM runs WHERE session_id = ? AND run_type = 'tool'", (sid,))]
        assert sorted(tool_calls) == ["fetch_notes", "fetch_notes", "transfer_to_agent", "transfer_to_agent"]
        root = conn.execute("SELECT inputs, outputs FROM runs WHERE session_id = ? AND parent_run_id IS NULL", (sid,)).fetchone()
        assert "battery" in root["inputs"] and "revised" in root["outputs"]
        found = {f["kind"] for f in shape.analyze(conn, [sid])["findings"]}
        rendered = shape.render(shape.analyze(conn, [sid]))
    assert {"branch_never_taken", "constant_rounds", "repeated_identical_reply"} <= found
    assert "never taken in 2 runs of it: lookup" in rendered and "critic: ran exactly 3 times" in rendered
    tools.CTX.clear()
    tools.CTX.update(project=project, out=out)
    try:  # what the requests reached, as the keep gate and the checks list read it
        ran, missed, _, _ = tools._reach("t")
    finally:
        tools.CTX.clear()
    assert "lookup" in missed and "pipeline:review_loop:critic" in ran


def test_a_change_to_an_agent_the_requests_never_ran_is_tied_to_its_lines_and_an_app_is_run_as_it_is(tmp_path):
    project = _project(tmp_path)
    (project / "app.py").write_text("from google.adk.apps import App\nfrom agent import root_agent\n"
                                    "app = App(name='fleet', root_agent=root_agent)\n", encoding="utf-8")
    assert census.take(project)["adk_agents"] == ["agent.py:root_agent", "app.py:app"]
    out = project / ".fleetopt"
    out.mkdir()
    entry = {"project": str(project), "graph": "app.py:app", "name": "fleet", "interpreter": sys.executable,
             "inputs": ["why is battery degradation up this quarter"]}
    path = out / "entry.json"
    path.write_text(json.dumps(entry), encoding="utf-8")
    sid, code, n_runs, _ = runner.run(project, tools.command(path, entry), out, with_io=True, label="t")
    assert code == 0 and n_runs > 10
    with store.connect(out / "fleetopt.db") as conn:
        sources = json.loads(conn.execute("SELECT sources FROM graphs WHERE session_id = ?", (sid,)).fetchone()[0])
    agent_py = (project / "agent.py").read_text(encoding="utf-8").splitlines()
    file, first, last = sources["lookup"]
    assert pathlib.Path(file).name == "agent.py" and agent_py[first - 1].startswith("lookup = LlmAgent(")
    assert "tools=[lookup_fleet]" in agent_py[last - 1]
    assert set(sources) == {"coordinator", "lookup", "pipeline", "pipeline:draft", "pipeline:review_loop",
                            "pipeline:review_loop:critic", "pipeline:review_loop:reviser"}
    # an edit to the agent the request never ran is proven by nothing; one to an agent that ran is not flagged
    text = (project / "agent.py").read_text(encoding="utf-8")
    (project / "agent.py").write_text(text.replace('"Look a vehicle up."', '"Look a vehicle up, briefly."')
                                      .replace('"Critique the draft: {draft?}"', '"Critique the draft briefly: {draft?}"'), encoding="utf-8")
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "edit"], cwd=project, check=True)
    tools.CTX.clear()
    tools.CTX.update(project=project, out=out, start_sha=subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=project,
                                                                        capture_output=True, text=True).stdout.strip())
    try:
        tools.CTX["reach"] = tools._reach("t")
        assert tools._unproven() == [("lookup", "agent.py")]
    finally:
        tools.CTX.clear()


def test_a_skill_named_after_a_framework_goes_only_to_a_project_on_it():
    from fleetopt import expert as expert_mod

    design = expert_mod.get("design")
    assert "adk" in design.skills and "langgraph" in design.skills
    both, lg, adk, none = (design.system(f) for f in (None, {"langgraph"}, {"adk"}, set()))
    assert "fleetopt:adk" in both and "fleetopt:langgraph" in both
    assert "fleetopt:langgraph" in lg and "fleetopt:adk" not in lg
    assert "fleetopt:adk" in adk and "fleetopt:langgraph" not in adk
    assert "fleetopt:adk" not in none and "fleetopt:patterns" in none
    taken = census.take(ROOT / "fixture-adk")
    assert census.frameworks(taken) == {"adk"} and census.frameworks(census.take(ROOT / "fixture")) == {"langgraph"}
