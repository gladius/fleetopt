"""Starting an agent: what fleetopt finds, what it runs, what it remembers. No LLM."""

import json
import pathlib
import shutil
import sys

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
    (message,) = driver.build_input(chat, "hi")["messages"]
    assert (message.type, message.content) == ("human", "hi")  # a real message: a graph without a reducer reads .content
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
    assert any("it answered" in line for line in said) and any(line.startswith("    - battery") for line in said)

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
    assert "Every attempt is in" in str(caught.value) and len(asked) == 1


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


def test_a_project_with_two_providers_is_switched_to_the_one_it_has_a_key_for(project, tmp_path, monkeypatch):
    missing = "[driver] FAILED 'x'\n   OpenAIError: Missing credentials. Set the OPENAI_API_KEY environment variable."
    report = {"keys_present": ["ANTHROPIC_API_KEY"]}
    assert entry.other_provider(missing, report) == ["ANTHROPIC_API_KEY"]
    assert entry.other_provider(missing, {"keys_present": ["OPENAI_API_KEY"]}) == []   # it has that key: the key is wrong
    assert entry.other_provider("AuthenticationError: invalid x-api-key", report) == []  # names no key: the team's to fix
    assert entry.other_provider(missing, {"keys_present": []}) == []

    tried, asked, said = [], [], []

    def prove(path, found):
        tried.append(dict(found["env"]))
        return (True, "") if found["env"].get("MODEL_PROVIDER") == "anthropic" else (False, missing)

    def repair(project, found, failure):
        asked.append(failure)
        return {"env": {"MODEL_PROVIDER": "anthropic"}}

    monkeypatch.setattr(entry, "preflight", lambda path, found: ([], report))
    monkeypatch.setattr(entry, "prove", prove)
    monkeypatch.setattr(setup, "repair", repair)
    _, found = entry.ensure(project, tmp_path / "out", say=said.append)
    assert found["proven"] and tried == [{}, {"MODEL_PROVIDER": "anthropic"}]
    assert "ANTHROPIC_API_KEY" in asked[0] and "names only" in asked[0]
    assert any("MODEL_PROVIDER=anthropic" in line for line in said)

    # one switch, not a loop: refused again, it is the team's
    monkeypatch.setattr(entry, "prove", lambda path, found: (False, missing))
    with pytest.raises(entry.NotReady, match="The model provider refused the call"):
        entry.ensure(project, tmp_path / "out2", say=lambda line: None)


def test_an_agent_that_starts_and_never_finishes_is_carried_on_with_as_broken(project, tmp_path, monkeypatch):
    own_bug = ("[driver] FAILED 'Build a briefing'\n         BadRequestError: tool_result without a tool_use before it\n"
               "[driver] 0 finished, 0 paused, 1 failed, of 1 inputs\n" + entry.ANSWERED.format(n=5))
    assert entry.starts_but_fails(own_bug) == "BadRequestError: tool_result without a tool_use before it"
    assert entry.starts_but_fails("[driver] FAILED 'x'\n   KeyError: 'tenant'\n[driver] 0 finished") is None  # no model call: it never started

    said = []
    monkeypatch.setattr(entry, "preflight", lambda path, found: ([], {}))
    monkeypatch.setattr(entry, "prove", lambda path, found: (False, own_bug))
    monkeypatch.setattr(setup, "repair", lambda project, found, failure: None)  # nothing about the entry helps
    _, found = entry.ensure(project, tmp_path / "out", say=said.append)
    assert found["proven"] and "tool_result" in found["broken"]
    assert any("no request finishes" in line for line in said) and any("fixing it comes first" in line for line in said)

    # the same failure with no model call behind it is still a start that did not work
    monkeypatch.setattr(entry, "prove", lambda path, found: (False, "[driver] FAILED 'x'\n   KeyError: 'tenant'"))
    with pytest.raises(entry.Unstartable):
        entry.ensure(project, tmp_path / "out2", say=lambda line: None)


def test_a_trial_counts_the_calls_the_model_answered(project, tmp_path):
    out = tmp_path / "out"
    path = entry.path_for(out, project, "agent")
    found = {"project": str(project), "interpreter": sys.executable, "graph": "agent.py:graph", "paths": ["."],
             "env_file": None, "env": {}, "config": {}, "input_template": None, "inputs": ["battery degradation"]}
    entry.save(path, found)
    worked, tail = entry.prove(path, found)
    assert worked and "1 finished" in tail and "the model answered" not in tail  # said only when the request failed
    assert not list(out.glob("fleetopt-*"))  # a trial leaves nothing behind


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
    assert {"context", "store"} <= set(setup.ALLOWED)


def test_eval_cases_that_are_supplied_are_what_the_agent_is_run_on(project, tmp_path, monkeypatch):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"input": "what is 17% of 2,340?", "expected": "397.8"}\n'
                     '{"input": "draft an outreach email", "expected": "a short email"}\n', encoding="utf-8")
    inputs, source = entry.inputs_for(project, cases)
    assert inputs == ["what is 17% of 2,340?", "draft an outreach email"] and source == "the eval cases you supplied, cases.jsonl"
    with pytest.raises(entry.Unstartable, match="no eval cases could be read"):
        entry.inputs_for(project, tmp_path / "empty-folder-or-missing")

    monkeypatch.setattr(entry, "preflight", lambda path, found: ([], {}))
    monkeypatch.setattr(entry, "prove", lambda path, found: (True, ""))
    out = tmp_path / "out"
    _, first = entry.ensure(project, out, say=lambda line: None)                       # settled on the project's own inputs
    assert not first.get("asked_anew")
    _, found = entry.ensure(project, out, say=lambda line: None, supplied=cases)       # then cases arrive
    assert found["inputs"] == inputs and found["asked_anew"] and found["proven"] == first["proven"]
    _, again = entry.ensure(project, out, say=lambda line: None, supplied=cases)
    assert again["inputs"] == inputs and not again.get("asked_anew")                   # the same cases: nothing is new
    assert "asked_anew" not in json.loads(entry.path_for(out, project, first["name"]).read_text(encoding="utf-8"))
