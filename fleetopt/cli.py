"""fleetopt - make a LangGraph agent cheaper and simpler, and prove it still works.

    fleetopt apply <project>     review it, then change it on a new branch and prove each change
    fleetopt review <project>    look only: where it wastes money, whether its design fits

`apply` is the whole loop. `review` is its first half, for looking before anything
changes; `apply` then starts from that review instead of paying for another.
fleetopt works out how to start the agent itself, once per project, and remembers it.

`capture` is a diagnostic for when a run comes back empty on an unfamiliar repo: it
starts the agent under instrumentation and says how much it saw. Not part of the flow.
"""

import argparse
import asyncio
import datetime
import io
import json
import os
import pathlib
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
        path, found = entry.ensure(pathlib.Path(args.project).resolve(), pathlib.Path(args.out).resolve(), args.graph,
                                   supplied=getattr(args, "evals", None))
    except (entry.Unstartable, RuntimeError) as exc:
        print(f"[fleetopt] {exc}")
        return None
    args.agent = found.get("name") or found["graph"]
    args.asked_anew = bool(found.get("asked_anew"))
    return entry.command(path, found)


def _newest(out):
    """The id of the newest capture, to tell afterwards which ones a run made."""
    if not (out / "fleetopt.db").exists():
        return 0
    with store.connect(out / "fleetopt.db") as conn:
        return conn.execute("SELECT COALESCE(MAX(id), 0) FROM sessions").fetchone()[0]


def spent(out, project, after):
    """(runs of the agent, what they cost on the team's key) since capture `after`."""
    from fleetopt.evidence import measure as measure_mod

    return measure_mod.spent(out, project, after)


def summary(agent, record, facts, runs, team_cost, own_cost):
    """The run in a few lines, for someone who will not read the report. Every line is
    computed from what was recorded; none of it is written by a session."""
    from fleetopt.optimizer import review as review_mod

    kinds = [f["kind"] for f in record["findings"]]
    found = ", ".join(f"{kinds.count(k)} {k}" for k in review_mod.KINDS if k in kinds) or "nothing"
    money = lambda usd: "not priced" if usd is None else f"${usd:.2f}"
    undone = max(facts["tried"] - facts["kept"], 0)
    lines = [
        ("Agent", agent),
        ("Level", f"{record['level']} of 4, {review_mod.LEVELS[record['level']]}"),
        ("Found", found),
        ("Tried", f"{facts['tried']} changed version(s): {facts['kept']} kept, {undone} undone"),
        ("Gained", ", ".join(facts["gained"]) if facts["gained"] else "nothing proven"),
        ("Verdict", facts["verdict"].split(":")[0].lower()),
        ("Spent", f"{money(team_cost)} on the team's key in {runs} run(s) of the agent, {money(own_cost)} by fleetopt"),
        ("Branch", facts["branch"] + ("" if facts["kept"] else ", the same code it started from")),
        ("Read", facts["run_dir"] + "/report.md"),
    ]
    return "\n".join(f"{name:<8} {text}" for name, text in lines)


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


def _reviewed(project, out, run_cmd, max_usd, fresh=False, supplied=None, design=False):
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
        kept = review_mod.saved(out, project, run_cmd, state, design)
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
        marks = ",".join("?" * len(ids))
        with store.connect(out / "fleetopt.db") as conn:
            facts = shape.analyze(conn, ids)
            failed = conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE exit_code != 0 AND id IN ({marks})", ids).fetchone()[0]
            unfinished = conn.execute(
                "SELECT COUNT(*) FROM runs WHERE parent_run_id IS NULL AND (error IS NOT NULL OR outputs IS NULL)"
                f" AND session_id IN ({marks})", ids).fetchone()[0]
        if not facts["traces"]:
            # Observed: an agent that failed at import, and a reviewer session spent
            # describing a graph that never ran.
            print("[fleetopt] the agent never ran, so there is nothing to review.")
            return None, False
        with store.connect(out / "fleetopt.db") as conn:
            calls = conn.execute(f"SELECT COUNT(*) FROM runs WHERE run_type = 'llm' AND session_id IN ({marks})",
                                 ids).fetchone()[0]
        if not calls and not design:
            # Observed: an agent that calls Claude through the command-line program, not
            # through LangChain. 52 steps recorded, no model call, and a reviewer that spent
            # five minutes looking for costs in a run that had none to show.
            print("[fleetopt] fleetopt saw the agent run, but no model calls in it. It calls its models in a way\n"
                  "           fleetopt cannot see (not through LangChain), so there is nothing to review for cost.\n"
                  "           Nothing more was spent.")
            return None, False
        print(f"\n--- structure, from the traces (label {label}) ---\n" + shape.render(facts))

        cases, _ = evals.load(pathlib.Path(supplied).resolve() if supplied else project)
        purpose = ("What it is for: not stated - derive it from the README and the prompts. Eval cases: "
                   + (f"{len(cases)} supplied with --evals, and the agent was run on their inputs" if supplied and cases
                      else f"{len(cases)} found in the repository" if cases else "none found in the repository"))
        if failed:
            purpose += (f". NOTE: {failed} of {len(ids)} captured runs exited with an error, so the traces are "
                        "partial; say so, and treat the failure as the first finding")
        model = _model("FLEETOPT_REVIEW_MODEL", "FLEETOPT_MODEL")
        print(f"[fleetopt] auth: {config.auth_summary() or 'unknown (could not run auth status)'}")
        print("[fleetopt] reviewing: reading the code and the recorded run (about 5 minutes)", flush=True)
        try:
            text, cost = asyncio.run(review_mod.run(project, label, purpose, model=model, max_usd=max_usd,
                                                    design=design))
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
        "model": model, "reviewer_cost_usd": cost, "findings": found, "unfinished": unfinished,
        "level": review_mod.level(found, unfinished), "shape": facts, "events": tools.CTX["events"],
    }, indent=1, default=str), encoding="utf-8")
    record = review_mod.remember(out, project, run_cmd, state, label, run_dir, found, unfinished, design)
    print("\n--- review ---\n" + text)
    print(f"\n--- reviewer ${cost or 0:.4f} ---\n[fleetopt] run record: {run_dir}")
    return {**record, "text": text, "reviewer_cost_usd": cost}, True


def _named(found):
    return ", ".join(f"{f['id']} ({f['title']})" for f in found)


def chosen(found, only, has_cases):
    """(the findings apply will try, why the others are left). Decided here, in code:
    a change to the design, small or large, is tried only when the team has eval cases
    to judge it on. Without them the only evidence would be the old answers, and a
    different design does not give the old answers."""
    if only:
        ids = [i.strip().upper() for i in only.split(",") if i.strip()]
        unknown = [i for i in ids if i not in {f["id"] for f in found}]
        if unknown:
            raise ValueError(f"the review has no finding {', '.join(unknown)}. It has: "
                             f"{', '.join(f['id'] for f in found) or 'none'}")
        found = [f for f in found if f["id"] in ids]
    if has_cases:
        return found, []
    design = [f for f in found if f["kind"] != "cost"]
    left = [f"they change the design and this project has no eval cases to judge the answers on "
            f"(--evals supplies them): {_named(design)}"] if design else []
    return [f for f in found if f["kind"] == "cost"], left


def _level(record):
    from fleetopt.optimizer import review as review_mod

    kinds = [f["kind"] for f in record["findings"]]
    counts = ", ".join(f"{kinds.count(k)} {k}" for k in review_mod.KINDS if k in kinds) or "no findings"
    broken = f"; {record['unfinished']} request(s) did not finish" if record.get("unfinished") else ""
    return f"[fleetopt] level {record['level']} of 4, {review_mod.LEVELS[record['level']]}: {counts}{broken}"


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
    record, new = _reviewed(project, out, run_cmd, args.max_usd, fresh=args.fresh or args.asked_anew,
                            supplied=args.evals, design=args.design)
    if record is None:
        return 1
    if not new:
        print(f"[fleetopt] the code has not changed since the review of {record['when']}, so this is that review. "
              "Nothing was run, nothing was spent (--fresh runs it again)")
        print("\n--- review ---\n" + record["text"])

    picked, left = chosen(record["findings"], None,
                          bool(evals.load(pathlib.Path(args.evals).resolve() if args.evals else project)[0]))
    print("\n" + _level(record))
    print("\n--- what you can do next ---")
    if picked:
        print(f"fleetopt apply {args.project}")
        print(f"  tries, on a new branch, one commit each: {_named(picked)}")
        print("  then looks again for what the first fixes uncover. --only C1,D2 tries just those and stops")
    else:
        print("The review found nothing for `fleetopt apply` to try on its own.")
    for why in left:
        print(f"  left alone, {why}")
    print("Nothing is merged or pushed: the branch is yours to read, keep or drop.")
    return 0


def expect(out, project, label, findings):
    """What a run will take, from one run of the agent as the review saw it: said
    before anything is spent, so nobody takes a long run for a stuck one."""
    from fleetopt.evidence import measure as measure_mod
    from fleetopt.optimizer import loop

    with store.connect(out / "fleetopt.db") as conn:
        row = conn.execute("SELECT id FROM sessions WHERE label = ? AND project = ? ORDER BY id DESC LIMIT 1",
                           (label, str(project))).fetchone()
        stats = measure_mod.session_stats(conn, row["id"]) if row else {}
    one = (stats.get("wall_ms") or 60_000) / 60_000
    k = len(findings)
    runs = loop.RUNS * (1 + k)
    minutes = one * (1 + 2 * k) + 2 * k  # the baseline; per finding a trial run, the rest at once, the edit
    cost = stats.get("cost_usd")
    money = f", about ${cost * runs:.2f} on the team's key" if cost is not None else ""
    return (f"[fleetopt] {k} finding(s) to try. Expect about {max(5, round(minutes / 5) * 5):.0f} minutes and "
            f"{runs} runs of the agent{money}. It stops at ${float(os.environ.get('FLEETOPT_TEAM_USD') or loop.TEAM_USD):.2f} "
            "on the team's key whatever happens")


def apply(args):
    """The whole loop: review the code as it stands (or reuse the review of it), then
    try the findings on a new branch and prove each one."""
    from fleetopt.evidence import evals
    from fleetopt.optimizer import loop

    project = pathlib.Path(args.project).resolve()
    out = pathlib.Path(args.out).resolve()
    before = _newest(out)
    run_cmd = _start(args)
    if run_cmd is None:
        return 1
    record, new = _reviewed(project, out, run_cmd, min(1.0, args.max_usd), fresh=args.asked_anew,
                            supplied=args.evals, design=args.design)
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
    print(_level(record))
    print(f"[fleetopt] trying, in this order: {_named(picked)}")
    print(expect(out, project, record["label"], picked))
    print(f"[fleetopt] auth: {config.auth_summary() or 'unknown (could not run auth status)'}")
    job = next((line.split(":", 1)[1].strip() for line in record["text"].splitlines() if line.startswith("Job:")),
               "answer the user's request")
    facts = asyncio.run(loop.run(project, out, run_cmd, record["text"], picked, task=job,
                                 model=_model("FLEETOPT_MODEL"), max_usd=args.max_usd, evals=args.evals,
                                 first_session=before))
    runs, team_cost = spent(out, project, before)
    own = (facts["own_cost_usd"] or 0) + ((record.get("reviewer_cost_usd") or 0) if new else 0)
    text = summary(args.agent, record, facts, runs, team_cost, own)
    print("\n--- summary (computed) ---\n" + text)
    try:
        (pathlib.Path(facts["run_dir"]) / "summary.txt").write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"[fleetopt] could not write the summary: {exc}")
    return 0


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
                     help="try just these findings of the review, e.g. C1,D2. Without it, every finding "
                          "the rules allow")
    app.add_argument("--max-usd", type=float, default=5.0,
                     help="the most fleetopt's own sessions may spend in this run (default 5). The agent's "
                          "calls on the team's key stop at $2 (FLEETOPT_TEAM_USD)")
    app.set_defaults(fn=apply)

    cap = sub.add_parser("capture", help="[debug] run a project under instrumentation")
    cap.add_argument("project")
    cap.add_argument("--label", default="manual", help="name for this capture (default: manual)")
    cap.set_defaults(fn=capture)

    rev = sub.add_parser("review", help="look only: where the agent wastes money and whether its design fits")
    rev.add_argument("project")
    rev.add_argument("--fresh", action="store_true",
                     help="run the agent and review it again even though the code has not changed")
    rev.add_argument("--max-usd", type=float, default=1.0,
                     help="stop the reviewer once its own spend reaches this (default 1). Does not cover the "
                          "target's API calls.")
    rev.set_defaults(fn=review)

    for p in (app, rev):
        p.add_argument("--design", action="store_true",
                       help="also review the design (does it fit the job, could it be simpler) and, with apply, "
                            "try design changes when the team has eval cases. Off by default: cost only")
    for p in (app, rev):  # the same cases to look and to change, or the review saw other requests
        p.add_argument("--evals", help="file or folder of eval cases (input + expected answer): JSONL/JSON or "
                                       "deepeval tests. Their inputs are what the agent is run on. Found in "
                                       "the project automatically otherwise.")
    for p in (app, cap, rev):
        p.add_argument("--graph", help="rarely needed: with several agents in a project fleetopt picks the one the "
                                       "team ships and says why. This overrides it: a name from its "
                                       "langgraph.json, or file.py:variable")
    for p in (app, cap, rev):  # after the subcommand, where people put it
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
