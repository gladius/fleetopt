"""Starting an agent: the driver, and the `start` tool's tries.

Tries run for real on the fixture, whose model is a fake: no network, no cost.
"""

import asyncio
import json
import pathlib
import shutil
import sys

import pytest

from fleetopt import tools
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
    record = {"incident": {"id": "INC1", "text": "disk full"}, "priority": 2}
    assert driver.build_input(state, record, template) == record   # a whole record per request goes in as it is


def test_a_graph_built_in_a_function_and_compiled_at_start_up_is_started_as_the_service_would(tmp_path):
    (tmp_path / "svc.py").write_text(PLAIN.replace("graph = builder.compile()", "def get_graph():\n    return builder"),
                                     encoding="utf-8")
    for spec in ("svc.py:get_graph", "svc.py:get_graph()", "svc.py:builder"):   # with or without (), or the builder
        graph = driver.resolve(spec, tmp_path, [str(tmp_path)])
        assert graph.checkpointer is not None and hasattr(graph, "ainvoke"), spec


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
        assert "no input given" in trying({**ENTRY, "inputs": ["", {}]})
        assert tools.CTX["tries"] == 0
        assert tools._entry({**ENTRY, "inputs": ["the one example"]})["inputs"] == ["the one example"]  # one is enough
        record = {"incident": {"id": "INC1"}}
        assert tools._entry({**ENTRY, "inputs": [record, record, "x"]})["inputs"] == [record, "x"]

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


def test_the_projects_python_is_used_as_named_wherever_it_lives(project, tmp_path):
    env = tmp_path / "poetry-env" / "bin"
    env.mkdir(parents=True)
    try:
        (env / "python").symlink_to(sys.executable)              # a venv's python is a symlink (Linux, macOS)
    except OSError:
        shutil.copy(sys.executable, env / "python")               # Windows without symlink rights
    _trying(project, tmp_path)
    try:
        assert tools._entry({**ENTRY, "interpreter": str(env / "python")})["interpreter"] == str(env / "python")
        with pytest.raises(ValueError, match="not one"):
            tools._entry({**ENTRY, "interpreter": "agent.py"})
    finally:
        tools.CTX.clear()


def test_every_env_file_is_shown_by_its_names_never_its_values(project, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "from-a-login")
    (project / ".env").write_text("export AZURE_OPENAI_KEY=sk-secret\n# OLD=1\nMODEL=x\n", encoding="utf-8")
    (project / "app").mkdir()
    (project / "app" / ".env.local").write_text("GOOGLE_APPLICATION_CREDENTIALS=./sa.json\n", encoding="utf-8")
    (project / ".venv").mkdir()
    (project / ".venv" / ".env").write_text("NOT_THE_PROJECTS=1\n", encoding="utf-8")
    files, shell = tools.env_names(project)
    assert files == {".env": ["AZURE_OPENAI_KEY", "MODEL"], "app/.env.local": ["GOOGLE_APPLICATION_CREDENTIALS"]}
    assert "AWS_SECRET_ACCESS_KEY" in shell
    assert "sk-secret" not in str(files) and "from-a-login" not in str(shell)


def test_the_driver_runs_it_from_its_folder_with_its_env_files(tmp_path):
    import subprocess

    project = tmp_path / "p"
    (project / "app").mkdir(parents=True)
    (project / "app" / "helper.py").write_text("NAME = 'x'\n", encoding="utf-8")
    (project / "app" / "graph.py").write_text("import app.helper\n" + PLAIN.replace(
        'lambda state: {"text": state["text"].upper()}',
        'lambda state: print("SEEN", __import__("os").getenv("A"), __import__("os").getenv("B"), '
        '__import__("pathlib").Path.cwd().name) or {"text": "x"}'), encoding="utf-8")
    (project / "app" / ".env").write_text("A=from-app\nB=from-app\n", encoding="utf-8")
    (project / ".env").write_text("B=from-root\n", encoding="utf-8")
    entry = {"project": str(project), "graph": "app/graph.py:graph", "cwd": "app", "env_file": [".env", "app/.env"],
             "paths": ["app"], "inputs": ["hi"]}   # the project folder not listed: their `import app...` still works
    (tmp_path / "e.json").write_text(json.dumps(entry), encoding="utf-8")
    out = subprocess.run([sys.executable, driver.__file__, str(tmp_path / "e.json")], capture_output=True, text=True).stdout
    assert "SEEN from-app from-root app" in out          # in order, a name already set is kept; started in its folder


def test_an_entry_says_where_it_starts_and_which_env_files(project, tmp_path):
    _trying(project, tmp_path)
    try:
        (project / "app").mkdir()
        entry = tools._entry({**ENTRY, "cwd": "app", "env_file": ".env"})
        assert (entry["cwd"], entry["env_file"]) == ("app", [".env"])
        with pytest.raises(ValueError, match="cwd must be a folder in the project"):
            tools._entry({**ENTRY, "cwd": ".."})
    finally:
        tools.CTX.clear()


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
