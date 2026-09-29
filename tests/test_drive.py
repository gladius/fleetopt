"""Starting an agent: what fleetopt is told, what it checks, what it remembers.

The session that reads the project (setup.settle) is replaced by a stand-in answer, so
no model is called. Everything code does with that answer runs for real on the fixture.
"""

import json
import pathlib
import shutil
import sys

import pytest

from fleetopt.drive import driver, entry, setup

ROOT = pathlib.Path(__file__).resolve().parents[1]
ANSWER = {"graph": "agent.py:graph", "agent": "agent", "why": "", "others": [], "paths": ["."], "env_file": None,
          "env": {}, "config": {}, "context": {}, "store": None, "input_template": None,
          "inputs": ["battery degradation", "route optimization"], "inputs_from": "the project's inputs.jsonl",
          "missing": [], "agent_fault": None}


@pytest.fixture
def project(tmp_path):
    target = tmp_path / "fixture"
    shutil.copytree(ROOT / "fixture", target)
    return target


def _answers(monkeypatch, *answers):
    """The session's answers, in order, checked the way real ones are. Records what it was asked."""
    asked, queue = [], list(answers)

    def settle(project, **kw):
        asked.append(kw)
        return setup.checked(queue.pop(0) if len(queue) > 1 else queue[0], project)

    monkeypatch.setattr(setup, "settle", settle)
    return asked


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


def test_an_entry_is_worked_out_once_proven_by_running_it_and_remembered(project, tmp_path, monkeypatch):
    asked = _answers(monkeypatch, ANSWER)
    said = []
    path, first = entry.ensure(project, tmp_path / "out", say=said.append)
    assert first["proven"] and first["graph"] == "agent.py:graph" and first["inputs"] == ANSWER["inputs"]
    assert path.is_relative_to(tmp_path / "out") and not (project / "entries").exists()  # fleetopt's folder, not the repo
    assert any("it answered" in line for line in said) and any(line.startswith("    - battery") for line in said)
    assert asked[0]["have_inputs"] is False  # the fixture keeps inputs but no expected answers: not eval cases

    def never(*a, **k):
        raise AssertionError("a proven entry is neither worked out nor proven again")

    monkeypatch.setattr(entry, "prove", never)
    monkeypatch.setattr(setup, "settle", never)
    _, again = entry.ensure(project, tmp_path / "out", say=said.append)
    assert again == first


def test_what_only_the_team_can_provide_is_listed_and_nothing_runs(project, tmp_path, monkeypatch):
    _answers(monkeypatch, {**ANSWER, "missing": ["A Postgres database at DATABASE_URL; start it with docker compose up db"]})

    def never(*a, **k):
        raise AssertionError("nothing is run for a project that is not ready")

    monkeypatch.setattr(entry, "prove", never)
    with pytest.raises(entry.NotReady, match="(?s)1 thing to set up.*Postgres database"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)


def test_a_project_that_is_not_ready_gets_the_whole_list_at_once(project, tmp_path, monkeypatch):
    _answers(monkeypatch, {**ANSWER, "env_file": ".env"})
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
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for package in ("langchain_anthropic", "langchain_openai"):  # stand-ins: the agent pulls in two providers
        (project / package).mkdir()
        (project / package / "__init__.py").write_text("", encoding="utf-8")
    (project / "agent.py").write_text("import langchain_anthropic, langchain_openai\n" + (project / "agent.py").read_text(encoding="utf-8"),
                                      encoding="utf-8")
    _answers(monkeypatch, ANSWER)
    with pytest.raises(entry.NotReady, match="(?s)No key for a model provider.*ANTHROPIC_API_KEY.*OPENAI_API_KEY"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)

    (project / ".env").write_text("OPENAI_API_KEY=placeholder\n", encoding="utf-8")  # one of them is enough to try
    _answers(monkeypatch, {**ANSWER, "env_file": ".env"})
    _, found = entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert found["proven"] and found["env_file"] == ".env"
    assert "placeholder" not in json.dumps(found)  # the entry names the file, never what is in it


def test_a_wrong_answer_gets_one_more_try_with_what_happened(project, tmp_path, monkeypatch):
    asked = _answers(monkeypatch, {**ANSWER, "graph": "agent.py:no_such_graph"}, ANSWER)
    _, found = entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert found["proven"] and len(asked) == 2
    assert "could not be loaded" in asked[1]["failure"] and asked[1]["earlier"]["graph"] == "agent.py:no_such_graph"


def test_an_answer_that_names_a_file_not_there_is_not_used(project):
    with pytest.raises(ValueError, match="not in the project"):
        setup.checked({**ANSWER, "graph": "made_up.py:graph"}, project)
    with pytest.raises(ValueError, match="no graph named"):
        setup.checked({**ANSWER, "graph": "graph"}, project)
    with pytest.raises(ValueError, match="not the JSON object"):
        setup.checked(["a list"], project)


def test_a_credential_never_travels_through_an_entry(project):
    plan = setup.checked({**ANSWER, "env": {"MODEL_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk-x",
                                            "db_password": "x", "AUTH_TOKEN": "x"}}, project)
    assert plan["env"] == {"MODEL_PROVIDER": "anthropic"}


def test_an_agent_that_still_does_not_load_is_the_teams_to_fix(project, tmp_path, monkeypatch):
    _answers(monkeypatch, ANSWER)
    (project / "agent.py").write_text("raise RuntimeError('the vector store is not built')\n" + (project / "agent.py").read_text(encoding="utf-8"),
                                      encoding="utf-8")
    with pytest.raises(entry.NotReady, match="failed while loading: RuntimeError: the vector store is not built"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)


def test_a_call_the_provider_refuses_is_the_teams_to_fix(project, tmp_path, monkeypatch):
    asked = _answers(monkeypatch, ANSWER)
    monkeypatch.setattr(entry, "prove", lambda path, found: (False, "[driver] FAILED 'x'\n   AuthenticationError: invalid x-api-key"))
    with pytest.raises(entry.NotReady, match="The model provider refused the call: AuthenticationError: invalid x-api-key"):
        entry.ensure(project, tmp_path / "out", say=lambda line: None)
    assert "AuthenticationError" in asked[1]["failure"]  # the session saw it too: another provider may be the answer
    assert entry.refused("ValueError: boom") is None
    assert entry.refused("httpx.ConnectError: [Errno 111] Connection refused").startswith("A service the agent depends on")


def test_an_agent_that_starts_and_never_finishes_is_carried_on_with_as_broken(project, tmp_path, monkeypatch):
    own_bug = ("[driver] FAILED 'Build a briefing'\n         BadRequestError: tool_result without a tool_use before it\n"
               "[driver] 0 finished, 0 paused, 1 failed, of 1 inputs\n" + entry.ANSWERED.format(n=5))
    assert entry.starts_but_fails(own_bug) == "BadRequestError: tool_result without a tool_use before it"
    _answers(monkeypatch, ANSWER, {**ANSWER, "agent_fault": "act builds a prompt that orphans a tool result"})
    monkeypatch.setattr(entry, "prove", lambda path, found: (False, own_bug))
    said = []
    _, found = entry.ensure(project, tmp_path / "out", say=said.append)
    assert found["proven"] and "tool_result" in found["broken"] and any("no request finishes" in line for line in said)

    _answers(monkeypatch, ANSWER)  # no model call behind the failure: it never started
    monkeypatch.setattr(entry, "prove", lambda path, found: (False, "[driver] FAILED 'x'\n   KeyError: 'tenant'"))
    with pytest.raises(entry.Unstartable, match="after 2 tries"):
        entry.ensure(project, tmp_path / "out2", say=lambda line: None)


def test_a_relative_output_folder_still_finds_its_entry(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _answers(monkeypatch, ANSWER)
    path, found = entry.ensure(project, "out", say=lambda line: None)
    assert path.is_absolute() and found["proven"]


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


def test_eval_cases_are_the_inputs_and_the_session_is_told_so(project, tmp_path, monkeypatch):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"input": "what is 17% of 2,340?", "expected": "397.8"}\n'
                     '{"input": "draft an outreach email", "expected": "a short email"}\n', encoding="utf-8")
    inputs, source = entry.team_inputs(project, cases)
    assert inputs == ["what is 17% of 2,340?", "draft an outreach email"] and source == "the eval cases you supplied, cases.jsonl"
    with pytest.raises(entry.Unstartable, match="no eval cases could be read"):
        entry.team_inputs(project, tmp_path / "empty-folder-or-missing")

    asked = _answers(monkeypatch, ANSWER)
    monkeypatch.setattr(entry, "preflight", lambda path, found: ([], {"loaded": True, "input": "ok"}))
    monkeypatch.setattr(entry, "prove", lambda path, found: (True, ""))
    out = tmp_path / "out"
    _, first = entry.ensure(project, out, say=lambda line: None)                       # settled on the session's inputs
    assert not first.get("asked_anew") and asked[0]["have_inputs"] is False
    _, found = entry.ensure(project, out, say=lambda line: None, supplied=cases)       # then cases arrive
    assert found["inputs"] == inputs and found["asked_anew"] and found["proven"] == first["proven"]
    _, again = entry.ensure(project, out, say=lambda line: None, supplied=cases)
    assert again["inputs"] == inputs and not again.get("asked_anew")                   # the same cases: nothing is new
    assert "asked_anew" not in json.loads(entry.path_for(out, project, "agent").read_text(encoding="utf-8"))
    _answers(monkeypatch, ANSWER)
    fresh = []
    monkeypatch.setattr(setup, "settle", lambda project, **kw: fresh.append(kw) or setup.checked(ANSWER, project))
    entry.ensure(project, tmp_path / "out2", say=lambda line: None, supplied=cases)
    assert fresh[0]["have_inputs"] is True                                              # it is not asked to write any
