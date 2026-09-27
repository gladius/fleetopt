"""What must stay true whatever else changes: one test per promise the product makes.

    pytest -q tests/test_invariants.py

An experiment that breaks one of these broke the product, however good its numbers.
"""

import asyncio
import pathlib

import pytest

from fleetopt import cli, config
from fleetopt.evidence import judge, measure
from fleetopt.optimizer import review, session, tools
from fleetopt.probe import store

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = (ROOT / "fleetopt" / "probe" / "hooks" / "_fleetopt_hook.py").read_text(encoding="utf-8")


def options(**kw):
    return session.build_options(ROOT / "fixture", **kw)


def _session(conn, **row):
    cols, marks = ", ".join(row), ", ".join("?" * len(row))
    return conn.execute(f"INSERT INTO sessions ({cols}) VALUES ({marks})", tuple(row.values())).lastrowid


# --- the operator's machine and the target's repo stay out of a run ------------------

def test_a_session_inherits_nothing_from_the_operator():
    o = options()
    assert o.setting_sources == [] and o.strict_mcp_config is True
    assert list(o.mcp_servers) == ["fleetopt"]
    assert [p["path"] for p in o.plugins] == [str(session.PLUGIN)]
    assert o.skills and all(s.startswith("fleetopt:") for s in o.skills)


def test_only_credential_keys_leave_the_operators_settings():
    assert set(config.AUTH_KEYS) == {"env", "apiKeyHelper", "awsAuthRefresh", "awsCredentialExport"}


def test_a_session_has_no_web_no_scheduler_no_subagents():
    assert set(options().tools) == {"Read", "Grep", "Glob", "Bash", "Edit", "Write", "Skill"}


def test_a_session_cannot_read_secrets_or_stop_to_ask():
    denied = options().disallowed_tools
    assert {"AskUserQuestion", "Read(**/.env)", "Read(**/*.pem)", "Read(**/*secret*)"} <= set(denied)


# --- what touches the target is gated or forbidden -----------------------------------

def test_running_and_editing_are_never_approved_by_the_allowlist():
    allowed = set(options().allowed_tools)
    assert not {"Bash", "Edit", "Write"} & allowed
    assert not any(t.endswith(("measure", "set_run_command")) for t in allowed)


def test_the_bash_guard_is_always_attached_and_refuses_installs_and_hand_runs():
    assert [h.matcher for h in options(run_cmd="python agent.py").hooks["PreToolUse"]] == ["Bash"]
    guard = session.guard_bash("python agent.py")
    for cmd in ("pip install x", "uv run --with x python a.py", "python agent.py", "pytest tests/", "deepeval test run t/"):
        verdict = asyncio.run(guard({"tool_input": {"command": cmd}}, "id", None))
        assert verdict["hookSpecificOutput"]["permissionDecision"] == "deny", cmd


def test_the_optimizers_own_spend_is_capped_by_default():
    assert cli._parser().parse_args(["optimize", "repo"]).max_usd == 5.0


def test_the_probe_only_observes():
    assert "set_llm_cache" not in HOOK and "_generate" not in HOOK and "bind_tools" not in HOOK
    assert HOOK.count("StateGraph.compile = ") == 1  # the one patch: snapshot the graph as compiled


def test_the_target_never_inherits_fleetopts_own_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLEETOPT_INVARIANT_KEY", raising=False)
    (tmp_path / ".env").write_text("FLEETOPT_INVARIANT_KEY=abc\n", encoding="utf-8")
    config.load_env()
    try:
        assert "FLEETOPT_INVARIANT_KEY" not in config.child_env()
    finally:
        config._injected.discard("FLEETOPT_INVARIANT_KEY")
        monkeypatch.delenv("FLEETOPT_INVARIANT_KEY", raising=False)


# --- a number is only a claim when it has earned it ------------------------------------

def _measured(conn, tokens, exit_code=0):
    sid = _session(conn, project="p", label="x", code_state="v1", exit_code=exit_code)
    conn.execute("INSERT INTO runs (session_id, run_type, model, input_tokens, output_tokens, duration_ms)"
                 " VALUES (?, 'llm', 'claude-haiku-4-5', ?, 10, 100)", (sid, tokens))
    return sid


def test_a_change_inside_the_noise_is_not_a_saving(tmp_path):
    conn = store.connect(tmp_path / "n.db")
    steady = [_measured(conn, 1000) for _ in range(3)]  # reruns that agree exactly: spread 0
    verdict = lambda tokens: measure.compare(conn, steady, [_measured(conn, tokens)])["input_tokens"]["verdict"]
    assert verdict(990) == "within noise"   # 1%: under the minimum margin
    assert verdict(1015) == "within noise"
    assert verdict(900) == "improved"
    assert verdict(1100) == "regressed"


def test_a_crashed_run_never_reaches_a_median(tmp_path):
    tools.CTX.clear()
    tools.CTX.update(out=tmp_path, project=tmp_path / "a")
    with store.connect(tmp_path / "fleetopt.db") as conn:
        _session(conn, project=str(tmp_path / "a"), label="baseline", code_state="v1", exit_code=1)
        good = _session(conn, project=str(tmp_path / "a"), label="baseline", code_state="v1", exit_code=0)
    assert tools._ids("baseline") == [good]


def test_versions_of_the_source_are_never_pooled(tmp_path):
    conn = store.connect(tmp_path / "v.db")
    a = _session(conn, project="p", label="x", code_state="v1", exit_code=0)
    b = _session(conn, project="p", label="x", code_state="v2", exit_code=0)
    with pytest.raises(RuntimeError, match="different versions"):
        measure.aggregate(conn, [a, b])


def test_the_judge_fails_closed(monkeypatch):
    async def garbage(prompt, model=None):
        return {"_unparseable": "well, maybe"}

    monkeypatch.setattr(judge, "_ask", garbage)
    assert asyncio.run(judge.judge("task", "in", "before", "after"))["equivalent"] is False
    assert asyncio.run(judge.judge_expected("task", "in", "expected", "out"))["pass"] is False


# --- the review looks, and a structural change waits for the team's cases --------------

def test_the_reviewer_can_only_look():
    assert not {"Bash", "Edit", "Write", "Skill"} & set(review.READ_ONLY)
    served = {t.name for t in tools._TOOLS if f"mcp__fleetopt__{t.name}" in review.READ_ONLY}
    assert served == {"graph_shape", "graph_topology", "query_traces"}


def test_the_reviewer_gets_no_network_only_local_references():
    assert "create_agent" in review.REFERENCES and "Checked 20" in review.REFERENCES
    assert "WebFetch" not in review.READ_ONLY and "WebSearch" not in review.READ_ONLY


def test_structural_patches_wait_for_eval_cases():
    guide = " ".join((session.PLUGIN / "skills" / "patterns" / "SKILL.md").read_text(encoding="utf-8").split())
    assert "only if eval cases are loaded" in " ".join(session.REVIEW_MISSION.split())
    assert "only when eval cases are loaded" in guide
    assert "what must survive" in guide


# --- the verdict belongs to the measurements, not to the agent that wants it --------------

def _judged(candidate, passed, base_state="v1", cand_state="v2", before=4, after=4):
    return {"event": "judge", "baseline": "baseline", "candidate": candidate, "passed": passed,
            "baseline_state": base_state, "candidate_state": cand_state,
            "equivalence": [{"equivalent": True}, {"equivalent": passed}],
            "correctness": {"cases": 12, "matched": 4, "baseline_pass": before, "candidate_pass": after}}


def test_a_failed_gate_cannot_be_argued_away():
    events = [_judged("patched", False, after=3),
              _judged("patched-again", True),                                       # one pass does not erase a failure
              _judged("baseline-retest", False, base_state="v2", cand_state="v2")]  # same code: not a before/after
    text = session.verdict(events)
    assert text.startswith("NOT PROVEN SAFE") and "4/4 before and 3/4 after" in text
    assert "patched-again" in text and "baseline-retest" not in text


def test_nothing_is_proven_without_a_judged_change():
    assert session.verdict([]).startswith("NOTHING PROVEN")
    assert session.verdict([_judged("again", True, cand_state="v1")]).startswith("NOTHING PROVEN")
    assert session.verdict([_judged("patched", True)]).startswith("PROVEN ON THIS EVIDENCE")


def test_a_comparison_says_which_code_each_side_ran():
    same = tools.sides("baseline", "baseline-retest", "abc+1", "abc+1")
    assert "SAME code" in same and "says nothing about the effect of a change" in same
    assert "SAME code" not in tools.sides("baseline", "patched", "abc+1", "abc+2")
