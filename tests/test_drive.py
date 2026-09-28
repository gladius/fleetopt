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


def test_what_the_project_declares_is_the_list_otherwise_what_its_source_shows(project):
    assert [name for name, _ in entry.candidates(project)] == ["agent"]  # declared: that is the list
    (project / "langgraph.json").unlink()
    (project / "patterns.py").write_text("import re  # used by langgraph code\nWORD = re.compile('a+')\n", encoding="utf-8")
    (project / "factory.py").write_text(
        "from langgraph.graph import StateGraph\n"
        "def build_graph(checkpointer=None):\n    return StateGraph(dict).compile()\n"
        "def build_graph_for(tenant):\n    return None\n", encoding="utf-8")
    assert [spec for _, spec in entry.candidates(project)] == ["agent.py:graph", "supervisor.py:graph", "factory.py:build_graph()"]


def test_asking_for_an_agent_that_is_not_there_lists_the_ones_that_are(project):
    found = [("agent", "agent.py:graph"), ("supervisor", "supervisor.py:graph")]
    assert entry.choose(found) == ("agent", "agent.py:graph")
    assert entry.choose(found, "supervisor")[1] == "supervisor.py:graph"
    assert entry.choose(found, "supervisor.py:graph")[0] == "supervisor"
    with pytest.raises(entry.Unstartable, match="agent, supervisor"):
        entry.choose(found, "billing")
    declared = entry.candidates(project)  # only `agent`; a file named outright is still accepted
    assert entry.choose(declared, "supervisor.py:graph", project) == ("supervisor.py:graph", "supervisor.py:graph")
    with pytest.raises(entry.Unstartable):
        entry.choose(declared, "missing.py:graph", project)
    with pytest.raises(entry.Unstartable, match="no graph found"):
        entry.choose([])
    (project / "langgraph.json").unlink()
    for name in ("agent.py", "supervisor.py"):
        (project / name).unlink()
    (project / "pipeline.py").write_text("from langgraph.graph import StateGraph\n"
                                         "def build_pipeline(settings):\n    return StateGraph(dict).compile()\n", encoding="utf-8")
    with pytest.raises(entry.NotReady, match="pipeline.py.*functions that need arguments.*langgraph.json"):
        entry.choose(entry.candidates(project), None, project)


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
    assert any("it runs" in line for line in said)

    def never(*a, **k):
        raise AssertionError("a proven entry is not proven again")

    monkeypatch.setattr(entry, "prove", never)
    _, again = entry.ensure(project, tmp_path / "out", say=said.append)
    assert again == first


def _no_repair(monkeypatch):
    def never(*a, **k):
        raise AssertionError("what the team must set up is not something to repair around")

    monkeypatch.setattr(setup, "repair", never)


def test_a_project_that_is_not_ready_gets_the_whole_list_at_once(project, tmp_path, monkeypatch):
    _no_repair(monkeypatch)
    declared = json.loads((project / "langgraph.json").read_text(encoding="utf-8"))
    declared["env"] = ".env"
    (project / "langgraph.json").write_text(json.dumps(declared), encoding="utf-8")
    (project / ".env.example").write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    (project / "uv.lock").write_text("", encoding="utf-8")
    (project / "agent.py").write_text("import a_package_nobody_installed\n" + (project / "agent.py").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(entry.NotReady) as caught:
        entry.ensure(project, tmp_path / "out", say=lambda line: None)
    text = str(caught.value)
    assert "2 things to set up, then run the same command again" in text
    assert ".env does not exist" in text and "Copy .env.example to it" in text
    assert "`a_package_nobody_installed`" in text and "uv sync" in text


def test_a_missing_provider_key_is_named_and_a_present_one_is_enough(project, tmp_path, monkeypatch):
    _no_repair(monkeypatch)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for package in ("langchain_anthropic", "langchain_openai"):  # stand-ins: the agent pulls in two providers
        (project / package).mkdir()
        (project / package / "__init__.py").write_text("", encoding="utf-8")
    (project / "agent.py").write_text("import langchain_anthropic, langchain_openai\n" + (project / "agent.py").read_text(encoding="utf-8"),
                                      encoding="utf-8")
    with pytest.raises(entry.NotReady, match="No key for a model provider.*ANTHROPIC_API_KEY.*OPENAI_API_KEY.*in the environment"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)

    (project / ".env").write_text("OPENAI_API_KEY=placeholder\n", encoding="utf-8")  # one of them is enough to try
    _, found = entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert found["proven"] and found["env_file"] == ".env"
    assert "placeholder" not in json.dumps(found)  # the entry names the file, never what is in it


def test_an_agent_that_fails_while_loading_is_the_teams_to_fix(project, tmp_path, monkeypatch):
    _no_repair(monkeypatch)
    (project / "agent.py").write_text("raise RuntimeError('the vector store is not built')\n" + (project / "agent.py").read_text(encoding="utf-8"),
                                      encoding="utf-8")
    with pytest.raises(entry.NotReady, match="failed while loading: RuntimeError: the vector store is not built"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)


def test_a_call_the_provider_refuses_is_the_teams_to_fix(project, tmp_path, monkeypatch):
    _no_repair(monkeypatch)
    monkeypatch.setattr(entry, "prove", lambda path, found: (False, "[driver] FAILED 'x'\n   AuthenticationError: invalid x-api-key"))
    with pytest.raises(entry.NotReady, match="The model provider refused the call: AuthenticationError: invalid x-api-key"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert entry.refused("ValueError: boom") is None
    assert entry.refused("httpx.ConnectError: [Errno 111] Connection refused").startswith("A service the agent depends on")


def test_an_agent_that_cannot_be_started_says_what_was_tried(project, tmp_path, monkeypatch):
    source = (project / "agent.py").read_text(encoding="utf-8")
    (project / "agent.py").write_text(source.replace("def plan(state: State) -> dict:",
                                                     "def plan(state: State) -> dict:\n    raise ValueError('this node is broken')", 1),
                                      encoding="utf-8")
    asked = []
    monkeypatch.setattr(setup, "repair", lambda project, found, failure: asked.append(failure) or None)
    with pytest.raises(entry.Unstartable, match="this node is broken") as caught:
        entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert not isinstance(caught.value, entry.NotReady)  # nothing for the team to set up: this one is fleetopt's to work out
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


def _two_agents(project):
    declared = json.loads((project / "langgraph.json").read_text(encoding="utf-8"))
    declared["graphs"] = {"agent": "./agent.py:graph", "supervisor": "./supervisor.py:graph"}
    (project / "langgraph.json").write_text(json.dumps(declared), encoding="utf-8")


def test_with_several_agents_it_picks_the_one_the_team_ships_says_why_and_remembers(project, tmp_path, monkeypatch):
    _two_agents(project)
    asked = []
    monkeypatch.setattr(setup, "choose_agent", lambda project, found: asked.append(found) or
                        ("supervisor", "the team's eval runner builds it and the other is one of its parts"))
    said = []
    _, first = entry.ensure(project, tmp_path / "out", say=said.append)
    assert first["name"] == "supervisor" and first["proven"]
    assert any("2 agents" in line and "the team's eval runner builds it" in line for line in said)
    assert any("the others: agent" in line and "--graph picks another" in line for line in said)

    _, again = entry.ensure(project, tmp_path / "out", say=said.append)
    assert again["name"] == "supervisor" and len(asked) == 1     # asked once, remembered
    _, named = entry.ensure(project, tmp_path / "out", "agent", say=said.append)
    assert named["name"] == "agent" and len(asked) == 1          # naming one is never second-guessed
    assert any("because you named it" in line for line in said)


def test_when_nothing_tells_them_apart_it_takes_the_first_and_says_so(project, tmp_path, monkeypatch):
    _two_agents(project)
    monkeypatch.setattr(setup, "choose_agent", lambda project, found: None)
    said = []
    _, found = entry.ensure(project, tmp_path / "out", say=said.append)
    assert found["name"] == "agent" and any("nothing told them apart" in line for line in said)


def test_the_choice_must_be_one_of_the_projects_agents(monkeypatch):
    async def answer(*a, **k):
        return json.dumps({"name": "an_agent_that_is_not_there", "why": "it sounded right"})

    monkeypatch.setattr(setup, "_ask", answer)
    assert setup.choose_agent(pathlib.Path("."), [("agent", "agent.py:graph"), ("supervisor", "supervisor.py:graph")]) is None


def test_the_reason_for_a_choice_is_one_short_line(monkeypatch):
    async def answer(*a, **k):
        return json.dumps({"name": "supervisor", "why": "The CI gate builds exactly this agent. " + "and more words " * 30})

    monkeypatch.setattr(setup, "_ask", answer)
    name, why = setup.choose_agent(pathlib.Path("."), [("agent", "agent.py:graph"), ("supervisor", "supervisor.py:graph")])
    assert name == "supervisor" and why.startswith("the CI gate builds exactly this agent") and why.endswith("...")
    assert len(why.split()) <= 31


def test_a_relative_output_folder_still_finds_its_entry(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path, found = entry.ensure(project, "out", say=lambda line: None)
    assert path.is_absolute() and found["proven"]
