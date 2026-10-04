"""fleetopt: sends an expert to a team's LangGraph agent, and proves what it changes.

    fleetopt review <project>    look only: what is worth changing, with the numbers
    fleetopt apply  <project>    look, change it on a new branch, and prove each change

One Claude session does the work as the expert asked for (--expert, cost by default): it
works out how to start the project's agent, measures it, finds what its expertise looks
for and, with apply, changes it and proves each change. How to start it is remembered per
project.
"""

import argparse
import asyncio
import os
import pathlib
import signal
import sys

from fleetopt import config


def _parser():
    parser = argparse.ArgumentParser(prog="fleetopt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=".fleetopt", help=argparse.SUPPRESS)  # also accepted before the command
    sub = parser.add_subparsers(dest="cmd", required=True)
    from fleetopt.expert import EXPERTS

    for name, usd, text in (("review", 2.0, "look only: what is worth changing, with the numbers"),
                            ("apply", 5.0, "look, change the agent on a new branch, and prove each change")):
        p = sub.add_parser(name, help=text)
        p.add_argument("project")
        p.add_argument("--expert", choices=sorted(EXPERTS), default="cost",
                       help="which expert looks at it: " + "; ".join(f"{e.name}: {e.does}" for e in EXPERTS.values())
                            + " (default cost)")
        p.add_argument("--no-ask", action="store_true",
                       help="never ask a question at the terminal. Otherwise, while getting the agent started, it may "
                            "ask up to 3 things only the team knows; with no terminal it never asks")
        p.add_argument("--evals", metavar="WHAT", help="what to check the agent with: the eval command you run "
                                                       "(e.g. 'pytest tests/evals'), or a file of test cases, expected "
                                                       "answers or example requests. Found in the project otherwise")
        p.add_argument("--graph", help="which agent, when the project has several: a name from langgraph.json, "
                                       "or file.py:variable")
        p.add_argument("--max-usd", type=float, default=usd,
                       help=f"the most fleetopt's own work may spend on your Claude login (default {usd:g}). "
                            "The agent's runs stop at $2 on its own API key (FLEETOPT_TEAM_USD)")
        p.add_argument("--out", default=argparse.SUPPRESS, help="where records go (default ./.fleetopt)")
    return parser


def _console_never_crashes():
    """A Windows console raises on the first character it cannot encode: replace instead."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a real console stream
            pass


def _stop(signum, frame):
    raise KeyboardInterrupt  # a kill ends a run the way Ctrl-C does: the team's copy is put back first


def main(argv=None):
    _console_never_crashes()
    config.load_env()
    args = _parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _stop)
    from fleetopt import session, tools

    project = pathlib.Path(args.project).resolve()
    team = float(os.environ.get("FLEETOPT_TEAM_USD") or tools.TEAM_USD)
    minutes = float(os.environ.get("FLEETOPT_MAX_MINUTES") or tools.MAX_MINUTES)
    print(f"fleetopt {args.cmd} · {args.expert} · {project.name}\n  stops at: {minutes:g} min · ${team:.2f} spent by the agent on its "
          f"own API key · ${args.max_usd:.2f} spent by fleetopt on your Claude login", flush=True)
    try:
        facts = asyncio.run(session.run(project, args.out, look_only=args.cmd == "review", evals=args.evals,
                                        graph=args.graph, model=os.environ.get("FLEETOPT_MODEL") or None,
                                        max_usd=args.max_usd, expert=args.expert,
                                        ask=not args.no_ask and sys.stdin.isatty() and sys.stdout.isatty()))
    except (ValueError, RuntimeError) as exc:
        print(f"  can't run: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
    if facts["account"] and (facts["mode"] == "review" or not facts["kept"]):  # why nothing was kept, in its words
        print("\n" + facts["account"] + "\n")
    print("\n".join(session.summary(facts)))
    return 0 if facts["measured"] else 1


if __name__ == "__main__":
    sys.exit(main())
