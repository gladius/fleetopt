"""Starting an agent: what fleetopt finds, what it runs, what it remembers. No LLM."""

import json
import pathlib
import shutil

import pytest

from fleetopt.drive import driver, entry, setup

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    target = tmp_path / "fixture"
    shutil.copytree(ROOT / "fixture", target)
    return target


def test_what_the_project_declares_comes_first_then_what_its_source_shows(project):
    assert [name for name, _ in entry.candidates(project)] == ["agent", "supervisor"]
    (project / "langgraph.json").unlink()
    (project / "patterns.py").write_text("import re  # used by langgraph code\nWORD = re.compile('a+')\n", encoding="utf-8")
    (project / "factory.py").write_text(
        "from langgraph.graph import StateGraph\n"
        "def build_graph(checkpointer=None):\n    return StateGraph(dict).compile()\n"
        "def build_graph_for(tenant):\n    return None\n", encoding="utf-8")
    assert [spec for _, spec in entry.candidates(project)] == ["agent.py:graph", "supervisor.py:graph", "factory.py:build_graph()"]


def test_asking_for_an_agent_that_is_not_there_lists_the_ones_that_are(project):
    found = entry.candidates(project)
    assert entry.choose(found) == ("agent", "agent.py:graph")
    assert entry.choose(found, "supervisor")[1] == "supervisor.py:graph"
    assert entry.choose(found, "supervisor.py:graph")[0] == "supervisor"
    with pytest.raises(entry.Unstartable, match="agent, supervisor"):
        entry.choose(found, "billing")
    with pytest.raises(entry.Unstartable, match="no graph found"):
        entry.choose([])


def test_inputs_come_from_the_teams_cases_before_anything_else(project):
    assert entry.inputs_for(project) == (["battery degradation", "route optimization"], "the project's inputs.jsonl")
    (project / "evals").mkdir()
    (project / "evals" / "dataset.json").write_text(json.dumps(
        [{"inputs": {"question": f"question {i}"}, "outputs": {"answer": f"answer {i}"}} for i in range(12)]), encoding="utf-8")
    inputs, source = entry.inputs_for(project)
    assert inputs == ["question 0", "question 3", "question 6", "question 9"]  # four, spread across the file
    assert source == "the team's eval cases, evals/dataset.json"


class _Graph:
    def __init__(self, properties):
        self.properties = properties

    def get_input_jsonschema(self):
        return {"properties": self.properties}


def test_the_driver_puts_the_text_where_the_graph_takes_it():
    chat = _Graph({"messages": {"type": "array"}, "plan": {"type": "string"}})
    assert driver.build_input(chat, "hi") == {"messages": [{"role": "user", "content": "hi"}]}
    state = _Graph({"topic": {"type": "string"}, "notes": {"type": "array"}, "rounds": {"type": "integer"},
                    "summary": {"type": "string"}, "mode": {"type": "string", "default": "fast"}})
    assert driver.build_input(state, "hi") == {"topic": "hi", "notes": [], "rounds": 0, "summary": ""}
    with pytest.raises(ValueError, match="no text field"):
        driver.build_input(_Graph({"profile": {"type": "object"}}), "hi")
    template = {"jd_text": "{input}", "profile": {"name": "demo"}, "targets": ["resume"]}
    assert driver.build_input(state, "hi", template) == {"jd_text": "hi", "profile": {"name": "demo"}, "targets": ["resume"]}


def test_an_entry_is_proven_once_and_remembered(project, tmp_path, monkeypatch):
    said = []
    path, first = entry.ensure(project, tmp_path / "out", say=said.append)
    assert first["proven"] and first["graph"] == "agent.py:graph" and first["inputs"] == ["battery degradation", "route optimization"]
    assert path.is_relative_to(tmp_path / "out") and not (project / "entries").exists()  # fleetopt's folder, not the repo
    assert any("also here: supervisor" in line for line in said) and any("it runs" in line for line in said)

    def never(*a, **k):
        raise AssertionError("a proven entry is not proven again")

    monkeypatch.setattr(entry, "prove", never)
    _, again = entry.ensure(project, tmp_path / "out", say=said.append)
    assert again == first


def test_an_agent_that_cannot_be_started_says_what_was_tried(project, tmp_path, monkeypatch):
    (project / "agent.py").write_text("import langgraph\nraise RuntimeError('a service this agent needs is down')\ngraph = None\n"
                                      "graph = __import__('x').compile()\n", encoding="utf-8")
    asked = []
    monkeypatch.setattr(setup, "repair", lambda project, found, failure: asked.append(failure) or None)
    with pytest.raises(entry.Unstartable, match="a service this agent needs is down") as caught:
        entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert "What was tried is in" in str(caught.value) and len(asked) == 1


def test_a_repair_is_tried_by_running_it(project, tmp_path, monkeypatch):
    wrong = json.loads((project / "langgraph.json").read_text(encoding="utf-8"))
    wrong["graphs"] = {"agent": "./agent.py:not_the_name"}
    (project / "langgraph.json").write_text(json.dumps(wrong), encoding="utf-8")
    monkeypatch.setattr(setup, "repair", lambda project, found, failure: {"graph": "agent.py:graph"})
    _, found = entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert found["graph"] == "agent.py:graph" and found["proven"]


def test_a_credential_never_travels_through_an_entry(monkeypatch):
    async def answer(*a, **k):
        return json.dumps({"env": {"MODEL_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-leaked"}, "config": {"tenant_id": "demo"},
                           "interpreter": "/usr/bin/evil"})

    monkeypatch.setattr(setup, "_ask", answer)
    fix = setup.repair(pathlib.Path("."), {"graph": "a.py:g", "env": {}}, "failed")
    assert fix == {"config": {"tenant_id": "demo"}}  # the whole env is dropped, and the interpreter is not the model's to choose
