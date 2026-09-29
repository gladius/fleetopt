"""fleetopt: makes a LangGraph agent cheaper, and proves its answers still hold.

    fleetopt review <project>    look only: where it wastes tokens and money
    fleetopt apply  <project>    look, change it on a new branch, and prove each change

One agent does the work: it works out how to start the project's agent, measures it,
finds the waste and, with apply, changes it and proves each change. How to start it is
remembered per project.
"""

import argparse
import asyncio
import os
import pathlib
import sys

from fleetopt import config


def _parser():
    parser = argparse.ArgumentParser(prog="fleetopt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=".fleetopt", help=argparse.SUPPRESS)  # also accepted before the command
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, usd, text in (("review", 2.0, "look only: where the agent wastes tokens and money"),
                            ("apply", 5.0, "look, change the agent on a new branch, and prove each change")):
        p = sub.add_parser(name, help=text)
        p.add_argument("project")
        p.add_argument("--evals", help="eval cases (input and expected answer), JSONL/JSON or deepeval tests; their "
                                       "inputs are what the agent is run on. Found in the project otherwise")
        p.add_argument("--graph", help="which agent, when the project has several: a name from langgraph.json, "
                                       "or file.py:variable")
        p.add_argument("--max-usd", type=float, default=usd,
                       help=f"the most fleetopt itself may spend (default {usd:g}). Runs of the agent on the "
                            "team's key stop at $2 (FLEETOPT_TEAM_USD)")
        p.add_argument("--out", default=argparse.SUPPRESS, help="where records go (default ./.fleetopt)")
    return parser


def _console_never_crashes():
    """A Windows console raises on the first character it cannot encode: replace instead."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a real console stream
            pass


def main(argv=None):
    _console_never_crashes()
    config.load_env()
    args = _parser().parse_args(argv)
    from fleetopt.optimizer import agent, tools

    project = pathlib.Path(args.project).resolve()
    team = float(os.environ.get("FLEETOPT_TEAM_USD") or tools.TEAM_USD)
    minutes = float(os.environ.get("FLEETOPT_MAX_MINUTES") or tools.MAX_MINUTES)
    print(f"fleetopt {args.cmd} · {project.name} · limits: ${team:.2f} on the team's key · {minutes:g} min · "
          f"${args.max_usd:.2f} by fleetopt", flush=True)
    try:
        facts = asyncio.run(agent.run(project, args.out, look_only=args.cmd == "review", evals=args.evals,
                                      graph=args.graph, model=os.environ.get("FLEETOPT_MODEL") or "claude-sonnet-5",
                                      max_usd=args.max_usd))
    except (ValueError, RuntimeError) as exc:
        print(f"  can't run: {exc}")
        return 1
    if facts["account"] and (facts["mode"] == "review" or not facts["measured"]):
        print("\n" + facts["account"] + "\n")
    print("\n".join(agent.summary(facts)))
    return 0 if facts["measured"] else 1


if __name__ == "__main__":
    sys.exit(main())
