"""Run a project under instrumentation and ingest what came back.

The hook writes JSONL from inside the target process - append-only, lock-free,
and survives a crash mid-graph. We ingest into SQLite here, after the process
has exited, so nothing in the target's address space can hit a database lock.
"""

import hashlib
import os
import pathlib
import subprocess
import tempfile

from fleetopt import config
from fleetopt.probe import store

HOOKS_DIR = pathlib.Path(__file__).parent / "hooks"


def code_state(project):
    """Fingerprint the target's source at capture time.

    A label names a state of the code. Without this, measuring twice under one
    label after an edit silently medians the before and the after together - a
    contaminated number that looks exactly like a real one.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=project, capture_output=True, text=True
        ).stdout
    except OSError:
        return None
    if not head:
        return None
    return f"{head[:12]}+{hashlib.sha1(diff.encode()).hexdigest()[:8]}"


def execute(project, run_cmd, out_dir, with_io=False):
    """Run run_cmd under instrumentation. Touches no database, so several can run
    at once. Returns (raw_dir, traces_path, graphs_path, returncode)."""
    out = pathlib.Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = pathlib.Path(tempfile.mkdtemp(prefix="fleetopt-", dir=out))
    traces, graphs = raw / "traces.jsonl", raw / "graphs.jsonl"

    env = config.child_env()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HOOKS_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["FLEETOPT_CAPTURE"] = "1"
    env["FLEETOPT_TRACE_FILE"] = str(traces)
    env["FLEETOPT_GRAPH_FILE"] = str(graphs)
    if with_io:
        env["FLEETOPT_CAPTURE_IO"] = "1"

    result = subprocess.run(run_cmd, shell=True, cwd=project, env=env)
    return raw, traces, graphs, result.returncode


def ingest(project, run_cmd, out_dir, label, raw, traces, graphs, returncode):
    """Store one executed run. Serial by design - one SQLite writer at a time."""
    out = pathlib.Path(out_dir).resolve()
    conn = store.connect(out / "fleetopt.db")
    session_id = store.create_session(conn, project, run_cmd, label, code_state(project))
    n_runs = store.ingest_runs(conn, session_id, traces) if traces.exists() else 0
    n_graphs = store.ingest_graphs(conn, session_id, graphs) if graphs.exists() else 0
    store.finish_session(conn, session_id, returncode)
    conn.close()

    # Raw files are a debug artifact; the database is the store. Keep them only
    # if nothing was ingested, so a silent capture is diagnosable.
    if n_runs:
        for path in (traces, graphs):
            path.unlink(missing_ok=True)
        raw.rmdir()
    elif any(raw.iterdir()):
        print(f"[fleetopt] nothing ingested - raw output left in {raw}")
    else:
        raw.rmdir()  # the process died before the hook wrote anything; nothing to keep

    return session_id, n_runs, n_graphs


def run(project, run_cmd, out_dir, with_io=False, label=None):
    """Execute run_cmd under instrumentation and ingest it; return
    (session_id, exit_code, n_runs, n_graphs)."""
    raw, traces, graphs, code = execute(project, run_cmd, out_dir, with_io)
    session_id, n_runs, n_graphs = ingest(project, run_cmd, out_dir, label, raw, traces, graphs, code)
    return session_id, code, n_runs, n_graphs
