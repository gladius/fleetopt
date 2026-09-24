"""Run a project n times and compare two sets of runs.

The whole point is variance. A single before/after reports noise as improvement,
so everything here works in medians over n sessions and refuses to call a
difference a saving when it sits inside the spread of the baseline itself.
"""

import os
import statistics
from concurrent.futures import ThreadPoolExecutor

from fleetopt.evidence import pricing
from fleetopt.probe import runner, store


def collect(project, run_cmd, out_dir, n, label, with_io=True):
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
    with ThreadPoolExecutor(max_workers=workers) as pool:
        executed = list(pool.map(
            lambda _: runner.execute(project, run_cmd, out_dir, with_io), range(n)
        ))

    ids = []
    for i, (raw, traces, graphs, code) in enumerate(executed):  # ingest serially
        session_id, n_runs, _ = runner.ingest(
            project, run_cmd, out_dir, label, raw, traces, graphs, code
        )
        print(f"[fleetopt] {label} {i + 1}/{n}: session {session_id}, {n_runs} runs, exit {code}")
        # A run that crashed produced a truncated trace. Averaging it in drags the
        # median toward "cheaper" for the worst possible reason - the work didn't
        # happen. Refuse the whole measurement rather than quietly discount it.
        if not n_runs:
            raise RuntimeError(f"{label} run {i + 1} captured nothing - aborting")
        if code != 0:
            raise RuntimeError(
                f"{label} run {i + 1} exited {code} after {n_runs} runs - a failed "
                "invocation cannot be measured. Fix the run command or the target first."
            )
        ids.append(session_id)
    return ids


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
        if c is None:
            priced = False
        else:
            total_cost += c

    wall = [
        r["duration_ms"]
        for r in conn.execute(
            "SELECT duration_ms FROM runs WHERE session_id = ? AND parent_run_id IS NULL",
            (session_id,),
        )
        if r["duration_ms"]
    ]

    return {
        "llm_calls": len(rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "cost_usd": total_cost if priced else None,
        "wall_ms": sum(wall),
    }


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
        for key in ("llm_calls", "input_tokens", "output_tokens", "cost_usd", "wall_ms")
    }, per


def compare(conn, baseline_ids, candidate_ids):
    """Before/after on medians, with the baseline's own spread as the noise floor."""
    base, base_per = aggregate(conn, baseline_ids)
    cand, _ = aggregate(conn, candidate_ids)

    out = {}
    for key, before in base.items():
        after = cand.get(key)
        if before is None or after is None:
            out[key] = {"before": before, "after": after, "delta_pct": None, "verdict": "unpriced"}
            continue

        spread = [p[key] for p in base_per if p[key] is not None]
        noise = (max(spread) - min(spread)) if len(spread) > 1 else 0
        delta = after - before
        pct = (100 * delta / before) if before else None

        if abs(delta) <= noise:
            verdict = "within noise"
        elif delta < 0:
            verdict = "improved"
        else:
            verdict = "regressed"

        out[key] = {"before": before, "after": after, "delta_pct": pct, "verdict": verdict}
    return out


def render(comparison):
    lines = [f"{'metric':<16} {'before':>12} {'after':>12} {'change':>9}  verdict", "-" * 64]
    for key, v in comparison.items():
        before = f"{v['before']:,.4f}" if key == "cost_usd" and v["before"] else (
            f"{v['before']:,.0f}" if v["before"] is not None else "-"
        )
        after = f"{v['after']:,.4f}" if key == "cost_usd" and v["after"] else (
            f"{v['after']:,.0f}" if v["after"] is not None else "-"
        )
        pct = f"{v['delta_pct']:+.1f}%" if v["delta_pct"] is not None else "-"
        lines.append(f"{key:<16} {before:>12} {after:>12} {pct:>9}  {v['verdict']}")
    return "\n".join(lines)
