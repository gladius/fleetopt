"""Run a project n times and compare two sets of runs.

The whole point is variance. A single before/after reports noise as improvement,
so everything here works in medians over n sessions and refuses to call a
difference a saving when it sits inside the spread of the baseline itself.
"""

import os
import pathlib
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fleetopt.evidence import pricing
from fleetopt.probe import runner, store


RUN_MINUTES = 15  # one run of the agent, all its inputs; FLEETOPT_RUN_MINUTES overrides


def spent(out_dir, project, after=0):
    """(runs of the agent, what they cost on the team's key) since capture `after`.
    The cost is None when a model it used has no price here."""
    db = pathlib.Path(out_dir).resolve() / "fleetopt.db"
    if not db.exists():
        return 0, 0.0
    with store.connect(db) as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM sessions WHERE id > ? AND project = ?",
                                             (after, str(project)))]
        costs = [session_stats(conn, i)["cost_usd"] for i in ids]
    return len(ids), (None if None in costs else sum(costs))


def collect(project, run_cmd, out_dir, n, label, with_io=True, say=print, max_steps=None, probe=False):
    """Run the target n times under one label. Returns the session ids.

    The runs execute concurrently (FLEETOPT_PARALLEL, default n, max 5). This is
    not for speed: the optimizer calling us is an LLM session whose prompt cache
    expires after 5 minutes of silence. Three serial 70-second runs blow through
    that and force a full re-write of its context - measured at a third of the
    optimizer's bill. Concurrent runs keep the whole measurement under the TTL.
    Token and dollar figures are unaffected; wall_ms picks up contention noise,
    so set FLEETOPT_PARALLEL=1 when latency is the thing being measured.
    """
    workers = min(n, int(os.environ.get("FLEETOPT_PARALLEL", n) or 1), 5)
    minutes = float(os.environ.get("FLEETOPT_RUN_MINUTES") or RUN_MINUTES)
    executed, lock = [], threading.Lock()

    def run_one(_=None):
        began = time.time()
        result = runner.execute(project, run_cmd, out_dir, with_io, timeout=60 * minutes, max_steps=max_steps)
        code = result[3]
        ended = ("stopped: it had not ended" if code == runner.TIMED_OUT else
                 "stopped: far more steps than the original" if code == runner.RAN_AWAY else
                 "finished" if code == 0 else f"failed (exit {code})")
        with lock:  # said as each run ends, not after all of them
            executed.append(result)
            say(f"    run {len(executed)} of {n}: {ended} in {time.time() - began:.0f} s")
        return result

    # A command that has never succeeded on this project gets one probe run before
    # the rest start: a wrong interpreter then costs one crash, not n, and the
    # agent gets the traceback at once instead of after 15 failed runs (observed).
    # `probe`: changed code gets one run first too. A change that breaks the agent, or
    # sends it into a loop, then costs one run and not n.
    if n > 1 and (probe or not _ever_succeeded(out_dir, project, run_cmd)):
        n_left = n - 1 if run_one()[3] == 0 else 0
    else:
        n_left = n
    if n_left:
        with ThreadPoolExecutor(max_workers=min(workers, n_left)) as pool:
            list(pool.map(run_one, range(n_left)))

    # Ingest every executed run before judging any of them: stopping at the first
    # failure used to leave the other runs' temp dirs behind forever.
    ids, failure = [], None
    for i, (raw, traces, graphs, code) in enumerate(executed):  # ingest serially
        tail = runner.output_tail(raw)  # before ingest: it removes the log once runs are stored
        session_id, n_runs, _ = runner.ingest(
            project, run_cmd, out_dir, label, raw, traces, graphs, code
        )
        # A run that crashed produced a truncated trace. Averaging it in drags the
        # median toward "cheaper" for the worst possible reason - the work didn't
        # happen. Refuse the whole measurement rather than quietly discount it.
        if code == runner.RAN_AWAY:
            failure = failure or (
                f"{label} run {i + 1} took more than {max_steps} steps, far more than the original, and was "
                "stopped. The change most likely removed what ended the agent's loop."
            )
        elif code == runner.TIMED_OUT:
            failure = failure or (
                f"{label} run {i + 1} had not ended after {minutes:g} minutes and was stopped. An agent that "
                f"waits for a keyboard or for a service does this. Its last output lines:\n{tail}"
            )
        elif not n_runs:
            failure = failure or (
                f"{label} run {i + 1} captured nothing (exit {code}). The target's last "
                f"output lines:\n{tail}\n(full output: {raw}/target.log)"
            )
        elif code != 0:
            failure = failure or (
                f"{label} run {i + 1} exited {code} after {n_runs} runs - a failed "
                "invocation cannot be measured. The target's last output lines:\n"
                f"{tail}"
            )
        ids.append(session_id)
    if failure:
        raise RuntimeError(failure)
    return ids


def _ever_succeeded(out_dir, project, run_cmd):
    conn = store.connect(pathlib.Path(out_dir).resolve() / "fleetopt.db")
    try:
        return conn.execute(
            "SELECT 1 FROM sessions WHERE project = ? AND run_cmd = ? AND exit_code = 0 LIMIT 1",
            (str(project), run_cmd),
        ).fetchone() is not None
    finally:
        conn.close()


def session_stats(conn, session_id):
    """Per-session totals. Cost is None if any model used is unpriced."""
    rows = conn.execute(
        "SELECT model, provider, input_tokens, output_tokens,"
        "       cache_read_tokens, cache_write_tokens"
        "  FROM runs WHERE session_id = ? AND run_type = 'llm'",
        (session_id,),
    ).fetchall()

    total_cost, priced = 0.0, True
    for r in rows:
        # Not every integration sets ls_model_name; fall back to the provider so
        # an incomplete metadata payload doesn't silently price the run at zero.
        c = pricing.cost(
            r["model"] or r["provider"],
            r["input_tokens"],
            r["output_tokens"],
            r["cache_read_tokens"],
            r["cache_write_tokens"],
        )
        if c is None and not r["model"]:
            c = 0.0  # no model name at all: a fake or a local model, with nothing to bill (observed: a test's
            #          fake model made a whole run "not priced", and the cap on the team's key went blind)
        if c is None:
            priced = False
        else:
            total_cost += c

    roots = conn.execute(
        "SELECT duration_ms, outputs, error FROM runs WHERE session_id = ? AND parent_run_id IS NULL",
        (session_id,),
    ).fetchall()
    wall = [r["duration_ms"] for r in roots if r["duration_ms"]]
    # A request that raised, or returned nothing, did not do the work. Money spent on it
    # is not comparable with money spent on one that did: a design that crashes after
    # five calls is "cheaper" than one that finishes.
    completed = sum(r["error"] is None and r["outputs"] is not None for r in roots)
    cost = total_cost if priced else None
    return {
        "llm_calls": len(rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_usd": cost,
        "wall_ms": sum(wall),
        "completed": completed if roots else None,
        "cost_per_completed": (cost / completed) if cost is not None and completed else None,
    }


# Billed cost first: tokens track it loosely. Then whether the work got done at all.
METRICS = ("cost_usd", "wall_ms", "completed", "cost_per_completed", "llm_calls", "input_tokens", "output_tokens")
HIGHER_IS_BETTER = {"completed"}
MIN_RELATIVE_NOISE = 0.02


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def assert_one_code_state(conn, session_ids):
    """Refuse to average sessions that ran against different source.

    Reusing a label after an edit would otherwise median the before and the
    after together, producing a contaminated number indistinguishable from a
    real one. Fail loudly instead.
    """
    states = {
        r["code_state"]
        for r in conn.execute(
            "SELECT DISTINCT code_state FROM sessions WHERE id IN "
            f"({','.join('?' * len(session_ids))})",
            session_ids,
        )
    }
    if len(states) > 1:
        raise RuntimeError(
            f"these sessions ran against {len(states)} different versions of the "
            "source - a label must name one state of the code. Use a fresh label "
            "for the changed version."
        )


def aggregate(conn, session_ids):
    assert_one_code_state(conn, session_ids)
    per = [session_stats(conn, sid) for sid in session_ids]
    return {
        key: _median([p[key] for p in per])
        for key in METRICS
    }, per


def compare(conn, baseline_ids, candidate_ids):
    """Before/after on medians, with the baseline's own spread as the noise floor."""
    base, base_per = aggregate(conn, baseline_ids)
    cand, _ = aggregate(conn, candidate_ids)

    out = {}
    for key, before in base.items():
        after = cand.get(key)
        if before is None or after is None:
            if key == "cost_per_completed" and base.get("completed") == 0 and after is not None:
                verdict = "baseline finished nothing"
            elif key in ("completed", "cost_per_completed"):
                verdict = "n/a"
            else:
                verdict = "unpriced"
            out[key] = {"before": before, "after": after, "delta_pct": None, "verdict": verdict}
            continue

        spread = [p[key] for p in base_per if p[key] is not None]
        noise = (max(spread) - min(spread)) if len(spread) > 1 else 0
        # Three reruns of a steady agent spread by almost nothing, and then a 4 ms
        # change on a 455 ms run reads as a regression (observed). Reruns alone
        # understate variance, so nothing under 2% of the baseline is a verdict.
        noise = max(noise, MIN_RELATIVE_NOISE * abs(before))
        delta = after - before
        pct = (100 * delta / before) if before else None

        if abs(delta) <= noise:
            verdict = "within noise"
        elif (delta < 0) != (key in HIGHER_IS_BETTER):
            verdict = "improved"
        else:
            verdict = "regressed"

        out[key] = {"before": before, "after": after, "delta_pct": pct, "verdict": verdict}
    return out


def render(comparison):
    lines = [f"{'metric':<19} {'before':>12} {'after':>12} {'change':>9}  verdict", "-" * 67]
    for key, v in comparison.items():
        money = key in ("cost_usd", "cost_per_completed")
        before = f"{v['before']:,.4f}" if money and v["before"] else (
            f"{v['before']:,.0f}" if v["before"] is not None else "-"
        )
        after = f"{v['after']:,.4f}" if money and v["after"] else (
            f"{v['after']:,.0f}" if v["after"] is not None else "-"
        )
        pct = f"{v['delta_pct']:+.1f}%" if v["delta_pct"] is not None else "-"
        lines.append(f"{key:<19} {before:>12} {after:>12} {pct:>9}  {v['verdict']}")
    return "\n".join(lines)
