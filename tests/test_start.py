"""Starting an agent: the driver, and the `start` tool's tries.

Tries run for real on the fixture, whose model is a fake: no network, no cost.
"""

import asyncio
import json
import pathlib
import shutil

import pytest

from fleetopt.optimizer import tools
from fleetopt.probe import driver

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRY = {"graph": "agent.py:graph", "agent": "agent",
         "inputs": ["battery degradation", "route optimization", "cold-chain monitoring"]}
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


def _trying(project, tmp_path):
    tools.CTX.clear()
    tools.CTX.update(project=project, out=(tmp_path / "out").resolve(), entry_path=tools.entry_path(tmp_path / "out", project),
                     tries=0, run_cmd=None)
    return lambda plan: asyncio.run(tools.start.handler({"entry": json.dumps(plan)}))["content"][0]["text"]


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
    assert driver.platform(plain, {}) == {} and plain.store is None


def test_start_proves_an_entry_by_running_it_and_says_what_it_saw(project, tmp_path):
    (project / "plain.py").write_text(PLAIN, encoding="utf-8")
    trying = _trying(project, tmp_path)
    try:
        assert "not in the project" in trying({**ENTRY, "graph": "made_up.py:graph"})   # checked before anything runs
        assert "1 input(s) given" in trying({**ENTRY, "inputs": ["only one"]})    # too few to see past the noise
        assert tools.CTX["tries"] == 0

        assert "It did not start" in trying({**ENTRY, "graph": "agent.py:no_such_graph"})
        assert "saw no model call" in trying({**ENTRY, "graph": "plain.py:graph"})      # it ran, and nothing to see
        assert tools.CTX["run_cmd"] is None

        text = trying({**ENTRY, "env": {"MODEL_PROVIDER": "fake", "ANTHROPIC_API_KEY": "sk-x", "db_password": "x"},
                       "input_template": '{"topic": "{input}", "notes": [], "rounds": 0, "summary": ""}'})
        assert "It started" in text and tools.CTX["run_cmd"]
        saved = json.loads(tools.CTX["entry_path"].read_text(encoding="utf-8"))
        assert saved["proven"] and saved["env"] == {"MODEL_PROVIDER": "fake"}           # a credential never travels
        assert saved["input_template"]["notes"] == []   # sent as JSON text, used as the object it describes
        assert not list((tmp_path / "out").glob("fleetopt-*"))                          # a try leaves nothing behind
        assert "started already" in trying(ENTRY)
        tools.CTX.update(run_cmd=None, tries=tools.TRIES)
        assert "Refused" in trying(ENTRY)                                               # four tries on the team's key
    finally:
        tools.CTX.clear()


def test_only_key_names_are_read_never_values(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (project / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\nOPENAI_API_KEY=\nMODEL=x\n", encoding="utf-8")
    names = tools.key_names(project)
    assert "ANTHROPIC_API_KEY" in names and "OPENAI_API_KEY" not in names and "sk-secret" not in " ".join(names)


def test_expected_answers_are_the_teams_own_never_written_by_the_session(project, tmp_path):
    (project / "golden.csv").write_text('input,expected\n"battery degradation","Capacity fades with, heat and cycles"\n'
                                        '"route optimization","Shorter routes cut fuel"\n"cold-chain","Keep 2-8C"\n',
                                        encoding="utf-8")
    trying = _trying(project, tmp_path)
    try:
        made_up = {**ENTRY, "expected": ["Anything plausible", "Shorter routes cut fuel", "Keep 2-8C"],
                   "expected_from": "golden.csv"}
        assert "not in golden.csv" in trying(made_up)
        assert "expected_from" in trying({**made_up, "expected_from": None})
        copied = {**ENTRY, "expected": ["Capacity fades with, heat and cycles", "Shorter routes cut fuel", "Keep 2-8C"],
                  "expected_from": "golden.csv"}
        assert tools._entry(copied)["expected"][0] == "Capacity fades with, heat and cycles"   # CSV quoting is no bar
        assert tools._entry(ENTRY)["expected"] is None                                        # none given: none kept
    finally:
        tools.CTX.clear()
