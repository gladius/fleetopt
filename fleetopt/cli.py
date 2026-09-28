"""fleetopt - find cost savings in a LangGraph project without editing it.

    fleetopt optimize <project>            find waste, patch on a branch, prove it
    fleetopt optimize <project> --review   the same, plus an architecture review
    fleetopt review <project>              the architecture review alone

fleetopt works out how to start the agent itself, once per project, and remembers it.

`capture` and `report` are diagnostics for when a run comes back empty on an
unfamiliar repo. They are not part of the flow.
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


def _start(args):
    """The command that starts this project's agent: fleetopt's own driver, run by the
    project's interpreter, against the entry settled for it. None when it cannot be started."""
    from fleetopt.drive import entry

    try:
        path, found = entry.ensure(pathlib.Path(args.project).resolve(), pathlib.Path(args.out).resolve(), args.graph)
    except (entry.Unstartable, RuntimeError) as exc:
        print(f"[fleetopt] {exc}")
        return None
    return entry.command(path, found)


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

    run_cmd = _start(args)
    if run_cmd is None:
        return 1
    return asyncio.run(
        session.run(
            args.project,
            args.out,
            run_cmd,
            model=model,
            max_usd=args.max_usd,
            effort=os.environ.get("FLEETOPT_EFFORT") or None,
            evals=args.evals,
            review=args.review,
        )
    )


def capture(args):
    run_cmd = _start(args)
    if run_cmd is None:
        return 1
    session_id, code, n_runs, n_graphs = runner.run(
        pathlib.Path(args.project).resolve(), run_cmd, args.out, with_io=True, label=args.label
    )
    print(f"\n[fleetopt] exit {code} | session {session_id} | {n_runs} runs, {n_graphs} graphs")
    if not n_runs:
        print("[fleetopt] nothing was captured: the agent ran without going through LangChain's callbacks")
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


def review(args):
    """Review only: capture the agent, print the structural numbers, run the reviewer.
    No optimizer session, so it costs the target's own run plus one reviewer session."""
    import datetime
    import json

    from fleetopt.evidence import measure as measure_mod
    from fleetopt.evidence import shape
    from fleetopt.optimizer import review as review_mod
    from fleetopt.optimizer import tools
    from fleetopt.probe import store

    project = pathlib.Path(args.project).resolve()
    out = pathlib.Path(args.out).resolve()
    print(f"[fleetopt] auth: {config.auth_summary() or 'unknown (could not run auth status)'}")
    run_cmd = None
    if args.label:  # reuse a capture: a second opinion costs no second run of the target
        label = args.label
    else:
        run_cmd = _start(args)
        if run_cmd is None:
            return 1
        label = f"review-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        try:
            measure_mod.collect(project, run_cmd, out, 1, label)
        except RuntimeError as exc:
            # Unusable for a measurement, not for a review: what ran is evidence and the
            # crash is the first finding.
            print(f"[fleetopt] the run failed; reviewing what was captured.\n{exc}")

    tools.CTX.update({"project": project, "out": out, "run_cmd": run_cmd, "events": [], "review": True,
                      "include_failed": True})
    ids = tools._ids(label)
    if not ids:
        print(f"[fleetopt] nothing captured under {label!r} for this project - nothing to review")
        return 1
    with store.connect(out / "fleetopt.db") as conn:
        facts = shape.analyze(conn, ids)
        failed = conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE exit_code != 0 AND id IN ({','.join('?' * len(ids))})", ids).fetchone()[0]
    if not facts["traces"]:
        # Observed: a run command that failed at import, and a reviewer session spent
        # describing a graph that never ran.
        print("[fleetopt] the agent never ran, so there is nothing to review. Fix the run command first.")
        return 1
    print(f"\n--- structure, from the traces (label {label}) ---\n" + shape.render(facts))

    model = os.environ.get("FLEETOPT_REVIEW_MODEL") or os.environ.get("FLEETOPT_MODEL") or "claude-sonnet-5"
    purpose = "not stated - derive it from the README and the prompts"
    if failed:
        purpose += (f". NOTE: {failed} of {len(ids)} captured runs exited with an error, so the traces are partial;"
                    " say so, and treat the failure as the first finding")
    try:
        text, cost = asyncio.run(review_mod.run(project, label, purpose, model=model, max_usd=args.max_usd))
    except RuntimeError as exc:
        print(f"[fleetopt] {exc}")
        return 1

    run_dir = out / "runs" / f"{label}-{project.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "review.md").write_text(text + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "kind": "review", "project": str(project), "run_cmd": run_cmd, "label": label,
        "model": model, "reviewer_cost_usd": cost, "shape": facts, "events": tools.CTX["events"],
    }, indent=1, default=str), encoding="utf-8")
    print("\n--- architecture review ---\n" + text)
    print(f"\n--- reviewer ${cost or 0:.4f} ---\n[fleetopt] run record: {run_dir}")
    return 0


def _parser():
    parser = argparse.ArgumentParser(prog="fleetopt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=".fleetopt", help=argparse.SUPPRESS)  # old position, still accepted
    sub = parser.add_subparsers(dest="cmd", required=True)

    opt = sub.add_parser("optimize", help="find and prove cost savings in a project")
    opt.add_argument("project")
    opt.add_argument("--evals", help="file or folder of eval cases (input + expected answer): "
                                     "JSONL/JSON or deepeval tests. Found automatically otherwise.")
    opt.add_argument("--max-usd", type=float, default=5.0,
                     help="stop the optimizer once its own spend reaches this (default 5). "
                          "Does not cover the target's API calls.")
    opt.add_argument("--review", action="store_true",
                     help="also review the architecture (does the design fit the job?). Findings are "
                          "recommendations with evidence, under their own heading; nothing is patched on them")
    opt.set_defaults(fn=optimize)

    cap = sub.add_parser("capture", help="[debug] run a project under instrumentation")
    cap.add_argument("project")
    cap.add_argument("--label", default="manual", help="name for this capture (default: manual)")
    cap.set_defaults(fn=capture)

    rep = sub.add_parser("report", help="[debug] token and cost breakdown of a capture (not the optimizer's report)")
    rep.add_argument("--session", type=int, help="which capture to summarize (default: the newest)")
    rep.set_defaults(fn=report)

    rev = sub.add_parser("review", help="review the architecture only, without optimizing")
    rev.add_argument("project")
    rev.add_argument("--label", help="review an earlier capture again instead of running the agent (the label is "
                                     "printed by every review)")
    rev.add_argument("--max-usd", type=float, default=1.0,
                     help="stop the reviewer once its own spend reaches this (default 1). Does not cover the "
                          "target's API calls.")
    rev.set_defaults(fn=review)

    for p in (opt, cap, rev):
        p.add_argument("--graph", help="rarely needed: with several agents in a project fleetopt picks the one the "
                                       "team ships and says why. This overrides it: a name from its "
                                       "langgraph.json, or file.py:variable")
    for p in (opt, cap, rep, rev):  # after the subcommand, where people put it
        p.add_argument("--out", default=argparse.SUPPRESS,
                       help="where captures and run records go (default ./.fleetopt)")

    return parser


def main(argv=None):
    _console_never_crashes()
    config.load_env()
    args = _parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
