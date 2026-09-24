"""fleetopt - find cost savings in a LangGraph project without editing it.

    fleetopt optimize <project>

That is the product. `capture` and `report` below are diagnostics for when the
harness comes back empty on an unfamiliar repo - they are not part of the flow.
Everything else the optimizer needs is a tool it calls itself, not a step someone
has to run.
"""

import argparse
import asyncio
import io
import os
import pathlib
import statistics
import sys

from fleetopt import config
from fleetopt.probe import runner, store

# Fix Windows Unicode console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def optimize(args):
    from fleetopt.optimizer import session

    # Same credential Claude Code uses on this machine; nothing fleetopt-specific.
    print(f"[fleetopt] auth: {config.auth_summary() or 'unknown (could not run auth status)'}")
    # Model from env, then .env, then default
    model = os.environ.get("FLEETOPT_MODEL")
    if not model:
        for path in (pathlib.Path(".env"), config.GLOBAL_ENV):
            if path.exists():
                for line in path.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("FLEETOPT_MODEL="):
                        model = line.split("=", 1)[1].strip()
                        break
            if model:
                break
    model = model or "claude-sonnet-5"

    return asyncio.run(
        session.run(
            args.project,
            args.out,
            run_cmd=args.run,
            auto=args.auto,
            model=model,
            max_usd=args.max_usd,
            effort=os.environ.get("FLEETOPT_EFFORT") or None,
            evals=args.evals,
        )
    )


def capture(args):
    print(f"[fleetopt] running: {args.run}\n[fleetopt] cwd:     {args.project}")
    session_id, code, n_runs, n_graphs = runner.run(
        pathlib.Path(args.project).resolve(), args.run, args.out, with_io=True, label=args.label
    )
    print(f"\n[fleetopt] exit {code} | session {session_id} | {n_runs} runs, {n_graphs} graphs")
    if not n_runs:
        print("[fleetopt] no runs captured - did the command actually invoke the graph?")
        print("[fleetopt] the target's last output lines are in the path printed above")
    return code


def report(args):
    path = pathlib.Path(args.out).resolve() / "fleetopt.db"
    if not path.exists():
        sys.exit(f"no capture found at {path}")
    conn = store.connect(path)
    session_id = args.session or store.latest_session(conn)
    if session_id is None:
        print("no sessions captured")
        return 1

    totals = conn.execute(
        "SELECT COUNT(*) n, SUM(run_type = 'llm') llm,"
        "       SUM(input_tokens) tin, SUM(output_tokens) tout,"
        "       SUM(cache_read_tokens) cread"
        "  FROM runs WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    roots = [
        r["duration_ms"]
        for r in conn.execute(
            "SELECT duration_ms FROM runs WHERE session_id = ? AND parent_run_id IS NULL",
            (session_id,),
        )
        if r["duration_ms"]
    ]

    print(f"session       {session_id}")
    print(f"invocations   {len(roots)}")
    print(f"runs          {totals['n']}")
    print(f"llm calls     {totals['llm'] or 0}")
    print(f"tokens        {totals['tin'] or 0:,} in / {totals['tout'] or 0:,} out")
    if totals["cread"]:
        print(f"cache reads   {totals['cread']:,} tokens")
    if roots:
        print(f"wall clock    {statistics.median(roots):,.0f} ms (median per invocation)")

    nodes = conn.execute(
        "SELECT node, SUM(run_type = 'llm') llm, SUM(input_tokens) tin,"
        "       SUM(output_tokens) tout, MAX(prompt_chars) maxprompt"
        "  FROM runs WHERE session_id = ? AND node IS NOT NULL"
        " GROUP BY node ORDER BY (SUM(input_tokens) + SUM(output_tokens)) DESC",
        (session_id,),
    ).fetchall()
    if not nodes:
        print("\nno node attribution found (is this a LangGraph project?)")
        return 0

    grand = sum((n["tin"] or 0) + (n["tout"] or 0) for n in nodes) or 1
    print(f"\n{'node':<20} {'llm':>5} {'tok in':>10} {'tok out':>10} {'% tok':>7} {'max prompt':>11}")
    print("-" * 68)
    for n in nodes:
        share = 100 * ((n["tin"] or 0) + (n["tout"] or 0)) / grand
        prompt = f"{n['maxprompt']:,}" if n["maxprompt"] else "-"
        print(
            f"{n['node']:<20} {n['llm'] or 0:>5} {n['tin'] or 0:>10,} "
            f"{n['tout'] or 0:>10,} {share:>6.1f}% {prompt:>11}"
        )
    return 0


def _console_never_crashes():
    """The optimizer's text and the target's output can contain any character; a
    Windows console defaults to a legacy code page and raises on the first one it
    cannot encode. Keep the console's encoding, replace what it cannot show."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a real console stream
            pass


def main(argv=None):
    _console_never_crashes()
    config.load_env()
    parser = argparse.ArgumentParser(prog="fleetopt", description=__doc__)
    parser.add_argument("--out", default=".fleetopt", help="where captures are stored")
    sub = parser.add_subparsers(dest="cmd", required=True)

    opt = sub.add_parser("optimize", help="find and prove cost savings in a project")
    opt.add_argument("project")
    opt.add_argument("--run", help="how to invoke the agent (the optimizer finds it otherwise)")
    opt.add_argument("--auto", action="store_true", help="no permission prompts")
    opt.add_argument("--evals", help="file or folder of eval cases (input + expected answer): "
                                     "JSONL/JSON or deepeval tests. Found automatically otherwise.")
    opt.add_argument("--max-usd", type=float, default=5.0,
                     help="stop the optimizer once its own spend reaches this (default 5). "
                          "Does not cover the target's API calls.")
    opt.set_defaults(fn=optimize)

    cap = sub.add_parser("capture", help="[debug] run a project under instrumentation")
    cap.add_argument("project")
    cap.add_argument("--run", required=True)
    cap.add_argument("--label", default="manual")
    cap.set_defaults(fn=capture)

    rep = sub.add_parser("report", help="[debug] summarize a capture")
    rep.add_argument("--session", type=int)
    rep.set_defaults(fn=report)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
