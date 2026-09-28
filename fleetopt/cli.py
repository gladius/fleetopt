"""fleetopt - make a LangGraph agent cheaper and simpler, and prove it still works.

    fleetopt apply <project>     review it, then change it on a new branch and prove each change
    fleetopt review <project>    look only: where it wastes money, whether its design fits

`apply` is the whole loop. `review` is its first half, for looking before anything
changes; `apply` then starts from that review instead of paying for another.
fleetopt works out how to start the agent itself, once per project, and remembers it.

`capture` and `report` are diagnostics for when a run comes back empty on an
unfamiliar repo. They are not part of the flow.
"""

import argparse
import asyncio
import datetime
import io
import json
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


def _model(*names):
    """The model for a session: the first of `names` set in the environment, then
    FLEETOPT_MODEL from a .env, then the default."""
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    for path in (pathlib.Path(".env"), config.GLOBAL_ENV):
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip().startswith("FLEETOPT_MODEL="):
                    return line.split("=", 1)[1].strip()
    return "claude-sonnet-5"


def _captured(out, project, run_cmd, state):
    """The label of an earlier review capture of this exact code that never got its
    review, so the agent is not run on the team's key twice for one answer."""
    if not state or not (out / "fleetopt.db").exists():
        return None
    with store.connect(out / "fleetopt.db") as conn:
        row = conn.execute(
            "SELECT label FROM sessions WHERE project = ? AND run_cmd = ? AND code_state = ?"
            "   AND exit_code = 0 AND label LIKE 'review-%' ORDER BY id DESC LIMIT 1",
            (str(project), run_cmd, state)).fetchone()
    return row["label"] if row else None


def _reviewed(project, out, run_cmd, max_usd, fresh=False):
    """The review of the code as it stands now: the saved one when the code has not
    changed since, a new one otherwise. Returns (record, new) or (None, False)."""
    from fleetopt.evidence import evals
    from fleetopt.evidence import measure as measure_mod
    from fleetopt.evidence import shape
    from fleetopt.optimizer import review as review_mod
    from fleetopt.optimizer import tools
    from fleetopt.probe import runner

    state = runner.code_state(project)
    if not fresh:
        kept = review_mod.saved(out, project, run_cmd, state)
        if kept:
            return kept, False

    label = None if fresh else _captured(out, project, run_cmd, state)
    if label:
        print(f"[fleetopt] this code was captured before ({label}); the agent is not run again")
    else:
        label = f"review-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        try:
            measure_mod.collect(project, run_cmd, out, 1, label)
        except RuntimeError as exc:
            # Unusable for a measurement, not for a review: what ran is evidence and the
            # crash is the first finding.
            print(f"[fleetopt] the run failed; reviewing what was captured.\n{exc}")

    tools.CTX.update({"project": project, "out": out, "run_cmd": run_cmd, "events": [], "include_failed": True})
    try:
        ids = tools._ids(label)
        if not ids:
            print(f"[fleetopt] nothing captured under {label!r} for this project - nothing to review")
            return None, False
        with store.connect(out / "fleetopt.db") as conn:
            facts = shape.analyze(conn, ids)
            failed = conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE exit_code != 0 AND id IN ({','.join('?' * len(ids))})",
                ids).fetchone()[0]
        if not facts["traces"]:
            # Observed: an agent that failed at import, and a reviewer session spent
            # describing a graph that never ran.
            print("[fleetopt] the agent never ran, so there is nothing to review.")
            return None, False
        print(f"\n--- structure, from the traces (label {label}) ---\n" + shape.render(facts))

        cases, _ = evals.load(project)
        purpose = ("What it is for: not stated - derive it from the README and the prompts. Eval cases: "
                   + (f"{len(cases)} found in the repository" if cases else "none found in the repository"))
        if failed:
            purpose += (f". NOTE: {failed} of {len(ids)} captured runs exited with an error, so the traces are "
                        "partial; say so, and treat the failure as the first finding")
        model = _model("FLEETOPT_REVIEW_MODEL", "FLEETOPT_MODEL")
        print(f"[fleetopt] auth: {config.auth_summary() or 'unknown (could not run auth status)'}")
        try:
            text, cost = asyncio.run(review_mod.run(project, label, purpose, model=model, max_usd=max_usd))
        except RuntimeError as exc:
            print(f"[fleetopt] {exc}")
            return None, False
    finally:
        tools.CTX.pop("include_failed", None)  # a crashed run must never reach a median afterwards

    found = review_mod.findings(text)
    run_dir = out / "runs" / f"{label}-{project.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "review.md").write_text(text + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps({
        "kind": "review", "project": str(project), "run_cmd": run_cmd, "label": label, "code_state": state,
        "model": model, "reviewer_cost_usd": cost, "findings": found, "shape": facts,
        "events": tools.CTX["events"],
    }, indent=1, default=str), encoding="utf-8")
    record = review_mod.remember(out, project, run_cmd, state, label, run_dir, found)
    print("\n--- review ---\n" + text)
    print(f"\n--- reviewer ${cost or 0:.4f} ---\n[fleetopt] run record: {run_dir}")
    return {**record, "text": text}, True


def _named(found, kinds):
    return ", ".join(f"{f['id']} ({f['title']})" for f in found if f["apply"] in kinds)


def chosen(found, only, has_cases):
    """(the findings apply will try, why the others are left). Decided here, in code:
    a change to the design is never tried without the team's eval cases, and a
    redesign never unless a person named it."""
    left = []
    if only:
        ids = [i.strip().upper() for i in only.split(",") if i.strip()]
        unknown = [i for i in ids if i not in {f["id"] for f in found}]
        if unknown:
            raise ValueError(f"the review has no finding {', '.join(unknown)}. It has: "
                             f"{', '.join(f['id'] for f in found) or 'none'}")
        picked = [f for f in found if f["id"] in ids]
    else:
        picked = [f for f in found if f["apply"] != "human decides"]
        if len(picked) < len(found):
            left.append(f"for a person to decide, tried only when named with --only: {_named(found, ('human decides',))}")
    if not has_cases:
        design = [f for f in picked if f["apply"] != "yes"]
        if design:
            left.append("they change the design and this project has no eval cases to check the answers against "
                        f"(--evals supplies them): {_named(design, ('needs cases', 'human decides'))}")
        picked = [f for f in picked if f["apply"] == "yes"]
    return picked, left


def review(args):
    """Look only: capture the agent once, print the structural numbers, run the reviewer.
    Costs the target's own run plus one reviewer session, and nothing when the code has
    not changed since the last review."""
    from fleetopt.evidence import evals

    project = pathlib.Path(args.project).resolve()
    out = pathlib.Path(args.out).resolve()
    run_cmd = _start(args)
    if run_cmd is None:
        return 1
    record, new = _reviewed(project, out, run_cmd, args.max_usd, fresh=args.fresh)
    if record is None:
        return 1
    if not new:
        print(f"[fleetopt] the code has not changed since the review of {record['when']}, so this is that review. "
              "Nothing was run, nothing was spent (--fresh runs it again)")
        print("\n--- review ---\n" + record["text"])

    picked, left = chosen(record["findings"], None, bool(evals.load(project)[0]))
    print("\n--- what you can do next ---")
    if picked:
        print(f"fleetopt apply {args.project}")
        print(f"  tries, on a new branch, one commit each: {_named(picked, ('yes', 'needs cases'))}")
        print("  then looks again for what the first fixes uncover. --only C1,D2 tries just those and stops")
    else:
        print("The review found nothing for `fleetopt apply` to try on its own.")
    for why in left:
        print(f"  left alone, {why}")
    print("Nothing is merged or pushed: the branch is yours to read, keep or drop.")
    return 0


def apply(args):
    """The whole loop: review the code as it stands (or reuse the review of it), then
    try the findings on a new branch and prove each one."""
    from fleetopt.evidence import evals
    from fleetopt.optimizer import session

    project = pathlib.Path(args.project).resolve()
    out = pathlib.Path(args.out).resolve()
    run_cmd = _start(args)
    if run_cmd is None:
        return 1
    record, new = _reviewed(project, out, run_cmd, min(1.0, args.max_usd))
    if record is None:
        return 1
    if not new:
        print(f"[fleetopt] starting from the review of {record['when']}: the code has not changed since "
              f"({pathlib.Path(record['run_dir']) / 'review.md'})")

    cases, _ = evals.load(pathlib.Path(args.evals).resolve() if args.evals else project)
    try:
        picked, left = chosen(record["findings"], args.only, bool(cases))
    except ValueError as exc:
        print(f"[fleetopt] {exc}")
        return 1
    for why in left:
        print(f"[fleetopt] left alone, {why}")
    if not picked:
        print("[fleetopt] nothing to try, so the agent was not run again and nothing was changed.")
        return 1 if args.only else 0
    print(f"[fleetopt] trying: {_named(picked, ('yes', 'needs cases', 'human decides'))}"
          + ("; only these" if args.only else "; then looking again"))

    print(f"[fleetopt] auth: {config.auth_summary() or 'unknown (could not run auth status)'}")
    return asyncio.run(
        session.run(
            project, out, run_cmd, record["text"], picked,
            model=_model("FLEETOPT_MODEL"),
            max_usd=args.max_usd,
            effort=os.environ.get("FLEETOPT_EFFORT") or None,
            evals=args.evals,
            fenced=bool(args.only),
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


def _parser():
    parser = argparse.ArgumentParser(prog="fleetopt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=".fleetopt", help=argparse.SUPPRESS)  # old position, still accepted
    sub = parser.add_subparsers(dest="cmd", required=True)

    app = sub.add_parser("apply", help="review, then change the agent on a new branch and prove each change")
    app.add_argument("project")
    app.add_argument("--only", metavar="IDS",
                     help="try just these findings of the review and nothing else, e.g. C1,D2. Without it, "
                          "everything the review marked safe to try, then whatever those fixes uncover")
    app.add_argument("--evals", help="file or folder of eval cases (input + expected answer): "
                                     "JSONL/JSON or deepeval tests. Found automatically otherwise.")
    app.add_argument("--max-usd", type=float, default=5.0,
                     help="stop fleetopt's own session once its spend reaches this (default 5). "
                          "Does not cover the target's API calls.")
    app.set_defaults(fn=apply)

    cap = sub.add_parser("capture", help="[debug] run a project under instrumentation")
    cap.add_argument("project")
    cap.add_argument("--label", default="manual", help="name for this capture (default: manual)")
    cap.set_defaults(fn=capture)

    rep = sub.add_parser("report", help="[debug] token and cost breakdown of a capture (not the report of a run)")
    rep.add_argument("--session", type=int, help="which capture to summarize (default: the newest)")
    rep.set_defaults(fn=report)

    rev = sub.add_parser("review", help="look only: where the agent wastes money and whether its design fits")
    rev.add_argument("project")
    rev.add_argument("--fresh", action="store_true",
                     help="run the agent and review it again even though the code has not changed")
    rev.add_argument("--max-usd", type=float, default=1.0,
                     help="stop the reviewer once its own spend reaches this (default 1). Does not cover the "
                          "target's API calls.")
    rev.set_defaults(fn=review)

    for p in (app, cap, rev):
        p.add_argument("--graph", help="rarely needed: with several agents in a project fleetopt picks the one the "
                                       "team ships and says why. This overrides it: a name from its "
                                       "langgraph.json, or file.py:variable")
    for p in (app, cap, rep, rev):  # after the subcommand, where people put it
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
