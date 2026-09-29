"""Starting an agent: what fleetopt tries, what it checks, what it remembers.

Trials run for real on the fixture, whose model is a fake: no network, no cost. The
session that reads the project (start.settle) is replaced by a stand-in answer, so no
model is called.
"""

import asyncio
import json
import pathlib
import shutil
import sys

import pytest

from fleetopt.drive import driver, entry, start

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRY = {"adapter": "langgraph", "name": "agent", "graph": "agent.py:graph", "paths": ["."], "env_file": None,
         "env": {}, "config": {}, "context": {}, "store": None, "input_template": None,
         "inputs": ["battery degradation", "route optimization"], "inputs_source": "the project's inputs.jsonl",
         "proven": None, "broken": None}
PLAIN = """from typing import TypedDict
from langgraph.graph import END, START, StateGraph
class State(TypedDict):
    text: str
builder = StateGraph(State)
builder.add_node("shout", lambda state: {"text": state["text"].upper()})
builder.add_edge(START, "shout")
builder.add_edge("shout", END)
graph = builder.compile()
"""


@pytest.fixture
def project(tmp_path):
    target = tmp_path / "fixture"
    shutil.copytree(ROOT / "fixture", target)
    return target.resolve()


def _settles(monkeypatch, answer, proven=True, **extra):
    """The session's answer and the entry its trials proved, standing in for the session."""
    asked = []

    def settle(project, path, **kw):
        asked.append(kw)
        found = {**ENTRY, "project": str(project), "interpreter": kw["python"], **extra}
        if kw["inputs"]:
            found.update(inputs=kw["inputs"], inputs_source=kw["source"])
        return answer, (found if proven else None)

    monkeypatch.setattr(start, "settle", settle)
    return asked


def _trying(project, tmp_path, inputs=()):
    start.RUN.clear()
    start.RUN.update(project=project, path=entry.path_for(tmp_path / "out", project, "agent"), python=sys.executable,
                     inputs=list(inputs), source=None, trials=0, proven=None)
    return lambda plan: asyncio.run(start.try_start.handler({"entry": json.dumps(plan)}))["content"][0]["text"]


class _Graph:
    def __init__(self, properties):
        self.properties = properties

    def get_input_jsonschema(self):
        return {"properties": self.properties}


def test_the_driver_puts_the_text_where_the_graph_takes_it():
    chat = _Graph({"messages": {"type": "array"}, "plan": {"type": "string"}})
    (message,) = driver.build_input(chat, "hi")["messages"]
    assert (message.type, message.content) == ("human", "hi")  # a real message: a graph without a reducer reads .content
    state = _Graph({"topic": {"type": "string"}, "notes": {"type": "array"}, "rounds": {"type": "integer"},
                    "summary": {"type": "string"}, "mode": {"type": "string", "default": "fast"}})
    assert driver.build_input(state, "hi") == {"topic": "hi", "notes": [], "rounds": 0, "summary": ""}
    with pytest.raises(ValueError, match="no text field"):
        driver.build_input(_Graph({"profile": {"type": "object"}}), "hi")
    template = {"jd_text": "{input}", "profile": {"name": "demo"}, "targets": ["resume"]}
    assert driver.build_input(state, "hi", template) == {"jd_text": "hi", "profile": {"name": "demo"}, "targets": ["resume"]}


def test_a_trial_says_what_the_agent_did_and_leaves_nothing_behind(project, tmp_path):
    path = entry.path_for(tmp_path / "out", project, "agent")
    found = {**ENTRY, "project": str(project), "interpreter": sys.executable, "inputs": ["battery degradation"]}
    entry.save(path, found)
    result = start.trial(path, found)
    assert (result["exit"], result["requests"], result["finished"]) == (0, 1, 1)
    assert result["model_calls"] >= 1 and result["answered"] == result["model_calls"] and result["error"] is None
    assert not list((tmp_path / "out").glob("fleetopt-*"))


def test_try_start_proves_an_entry_by_running_it_and_says_what_it_saw(project, tmp_path):
    (project / "plain.py").write_text(PLAIN, encoding="utf-8")
    trying = _trying(project, tmp_path)
    assert "not in the project" in trying({**ENTRY, "graph": "made_up.py:graph"})   # checked before anything runs
    assert "no inputs" in trying({**ENTRY, "inputs": []})
    assert start.RUN["trials"] == 0

    assert "It did not start" in trying({**ENTRY, "graph": "agent.py:no_such_graph"})
    assert start.RUN["proven"] is None
    assert "saw no model call" in trying({**ENTRY, "graph": "plain.py:graph"})      # it ran, and nothing to see
    assert start.RUN["proven"] is None

    text = trying({**ENTRY, "env": {"MODEL_PROVIDER": "fake", "ANTHROPIC_API_KEY": "sk-x", "db_password": "x"}})
    assert "It started" in text and "ANTHROPIC_API_KEY, db_password" in text
    assert start.RUN["proven"]["env"] == {"MODEL_PROVIDER": "fake"}                  # a credential never travels
    assert "sk-x" not in start.RUN["path"].read_text(encoding="utf-8")
    start.RUN["trials"] = start.TRIALS
    assert "Refused" in trying(ENTRY)                                                # four trials on the team's key


def test_the_teams_eval_cases_are_the_inputs_whatever_the_session_wrote(project, tmp_path):
    trying = _trying(project, tmp_path, inputs=["what is 17% of 2,340?"])
    trying({**ENTRY, "inputs": ["something the session made up"]})
    assert start.RUN["proven"]["inputs"] == ["what is 17% of 2,340?"]


def test_an_entry_is_worked_out_once_and_remembered(project, tmp_path, monkeypatch):
    asked = _settles(monkeypatch, {"status": "started", "agent": "researcher", "why": "the one the tests build"})
    said = []
    path, first = entry.ensure(project, tmp_path / "out", say=said.append)
    assert first["proven"] and first["name"] == "researcher" and first["inputs"] == ENTRY["inputs"]
    assert path.is_relative_to(tmp_path / "out") and not (project / "entries").exists()  # fleetopt's folder, not the repo
    assert any("because the one the tests build" in line for line in said)
    assert any(line.startswith("    - battery") for line in said)
    assert asked[0]["inputs"] == []  # the fixture keeps inputs but no expected answers: not eval cases

    def never(*a, **k):
        raise AssertionError("a proven entry is not worked out again")

    monkeypatch.setattr(start, "settle", never)
    _, again = entry.ensure(project, tmp_path / "out", say=said.append)
    assert again == first


def test_what_only_the_team_can_provide_is_listed(project, tmp_path, monkeypatch):
    _settles(monkeypatch, {"status": "missing", "missing": ["A Postgres database at DATABASE_URL; start it with "
                                                            "docker compose up db", "ANTHROPIC_API_KEY in .env"]},
             proven=False)
    with pytest.raises(entry.NotReady, match="(?s)2 things to set up.*Postgres database.*ANTHROPIC_API_KEY"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)


def test_an_agent_fleetopt_cannot_see_is_said_so_plainly(project, tmp_path, monkeypatch):
    _settles(monkeypatch, {"status": "cannot_see", "explanation": "it calls the claude program in llm.py:42"},
             proven=False)
    with pytest.raises(entry.Unstartable, match="cannot see its model calls.*llm.py:42"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)
    _settles(monkeypatch, {"status": "started"}, proven=False)          # a claim no trial backs is not a start
    with pytest.raises(entry.Unstartable, match="could not start the agent"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)


def test_an_agent_that_starts_and_never_finishes_is_carried_on_with_as_broken(project, tmp_path, monkeypatch):
    _settles(monkeypatch, {"status": "started"}, broken="BadRequestError: tool_result without a tool_use before it")
    said = []
    _, found = entry.ensure(project, tmp_path / "out", say=said.append)
    assert found["proven"] and "tool_result" in found["broken"] and any("no request finishes" in line for line in said)


def test_a_relative_output_folder_still_finds_its_entry(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _settles(monkeypatch, {"status": "started"})
    path, found = entry.ensure(project, "out", say=lambda line: None)
    assert path.is_absolute() and found["proven"]


def test_an_agent_built_to_be_hosted_gets_a_store_and_its_context():
    class Hosted:
        store = None

    graph = Hosted()
    assert driver.platform(graph, {"context": {"user_id": "u1"}, "store": "memory"}) == {"context": {"user_id": "u1"}}
    assert type(graph.store).__name__ == "InMemoryStore"
    kept = graph.store
    driver.platform(graph, {"store": "memory"})
    assert graph.store is kept                      # the agent's own store is never replaced
    plain = Hosted()
    assert driver.platform(plain, {}) == {} and plain.store is None  # nothing is handed to an agent that did not need it


def test_eval_cases_are_the_inputs_and_the_session_is_told_so(project, tmp_path, monkeypatch):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"input": "what is 17% of 2,340?", "expected": "397.8"}\n'
                     '{"input": "draft an outreach email", "expected": "a short email"}\n', encoding="utf-8")
    inputs, source = entry.team_inputs(project, cases)
    assert inputs == ["what is 17% of 2,340?", "draft an outreach email"] and source == "the eval cases you supplied, cases.jsonl"
    with pytest.raises(entry.Unstartable, match="no eval cases could be read"):
        entry.team_inputs(project, tmp_path / "empty-folder-or-missing")

    asked = _settles(monkeypatch, {"status": "started"})
    out = tmp_path / "out"
    _, first = entry.ensure(project, out, say=lambda line: None)                       # settled on the session's inputs
    assert not first.get("asked_anew") and asked[0]["inputs"] == []
    _, found = entry.ensure(project, out, say=lambda line: None, supplied=cases)       # then cases arrive
    assert found["inputs"] == inputs and found["asked_anew"] and found["proven"] == first["proven"]
    _, again = entry.ensure(project, out, say=lambda line: None, supplied=cases)
    assert again["inputs"] == inputs and not again.get("asked_anew")                   # the same cases: nothing is new
    assert "asked_anew" not in json.loads(entry.path_for(out, project, "agent").read_text(encoding="utf-8"))
    fresh = _settles(monkeypatch, {"status": "started"})
    entry.ensure(project, tmp_path / "out2", say=lambda line: None, supplied=cases)
    assert fresh[0]["inputs"] == inputs                                                # it is not asked to write any
