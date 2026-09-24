"""SQLite store for captures.

One row per run, one session per capture. Sessions accumulate - comparing a
before and an after is a join, not a filesystem convention, and collecting n
runs to see past variance is just running capture n times.

Usage numbers are flattened into columns at ingest. That's what makes the DB
useful as an agent tool surface: `SELECT node, SUM(input_tokens) ... GROUP BY
node` returns five rows, where reading the raw JSONL would spend the context
window on payloads.
"""

import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project     TEXT,
    run_cmd     TEXT,
    label       TEXT,
    code_state  TEXT,
    started_at  TEXT,
    exit_code   INTEGER
);

CREATE TABLE IF NOT EXISTS runs (
    session_id        INTEGER NOT NULL REFERENCES sessions(id),
    run_id            TEXT,
    parent_run_id     TEXT,
    trace_id          TEXT,
    name              TEXT,
    run_type          TEXT,
    node              TEXT,
    step              INTEGER,
    model             TEXT,
    provider          TEXT,
    start_time        TEXT,
    end_time          TEXT,
    duration_ms       REAL,
    input_tokens      INTEGER DEFAULT 0,
    output_tokens     INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    prompt_chars      INTEGER,
    prompt            TEXT,
    completion        TEXT,
    inputs            TEXT,
    outputs           TEXT,
    error             TEXT
);

CREATE INDEX IF NOT EXISTS runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS runs_node    ON runs(session_id, node);

CREATE TABLE IF NOT EXISTS graphs (
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    name        TEXT,
    nodes       TEXT,
    edges       TEXT,
    mermaid     TEXT
);
"""


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def create_session(conn, project, run_cmd, label=None, code_state=None):
    cur = conn.execute(
        "INSERT INTO sessions (project, run_cmd, label, code_state, started_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (str(project), run_cmd, label, code_state),
    )
    conn.commit()
    return cur.lastrowid


def finish_session(conn, session_id, exit_code):
    conn.execute("UPDATE sessions SET exit_code = ? WHERE id = ?", (exit_code, session_id))
    conn.commit()


def _tokens(usage):
    """Pull the four numbers we care about out of whatever shape usage arrived in.

    Providers disagree on spelling (`input_tokens` vs `prompt_tokens`) and bury
    cache counts one level down, so probe for each rather than assume a layout.
    """
    if not isinstance(usage, dict):
        return 0, 0, 0, 0

    def pick(*names):
        for name in names:
            value = usage.get(name)
            if isinstance(value, int):
                return value
        return 0

    details = usage.get("input_token_details")
    if not isinstance(details, dict):
        details = {}

    cache_read = details.get("cache_read") or usage.get("cache_read_input_tokens") or 0
    cache_write = details.get("cache_creation") or usage.get("cache_creation_input_tokens") or 0

    return (
        pick("input_tokens", "prompt_tokens"),
        pick("output_tokens", "completion_tokens"),
        int(cache_read),
        int(cache_write),
    )


def ingest_runs(conn, session_id, path):
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        tin, tout, cread, cwrite = _tokens(r.get("usage"))
        rows.append(
            (
                session_id,
                r.get("run_id"),
                r.get("parent_run_id"),
                r.get("trace_id"),
                r.get("name"),
                r.get("run_type"),
                r.get("node"),
                r.get("step"),
                r.get("model"),
                r.get("provider"),
                r.get("start_time"),
                r.get("end_time"),
                r.get("duration_ms"),
                tin,
                tout,
                cread,
                cwrite,
                r.get("prompt_chars"),
                r.get("prompt"),
                r.get("completion"),
                json.dumps(r["inputs"], default=str) if "inputs" in r else None,
                json.dumps(r["outputs"], default=str) if "outputs" in r else None,
                r.get("error"),
            )
        )

    conn.executemany(
        "INSERT INTO runs VALUES (" + ",".join("?" * 23) + ")",
        rows,
    )
    conn.commit()
    return len(rows)


def ingest_graphs(conn, session_id, path):
    rows = []
    for line in path.open():
        if not line.strip():
            continue
        g = json.loads(line)
        rows.append(
            (
                session_id,
                g.get("name"),
                json.dumps(g.get("nodes")),
                json.dumps(g.get("edges")),
                g.get("mermaid"),
            )
        )
    conn.executemany("INSERT INTO graphs VALUES (?, ?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def latest_session(conn):
    row = conn.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None
