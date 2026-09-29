# fleetopt

Finds where a LangGraph agent wastes money or is more complicated than its job needs,
fixes what it can prove on a separate branch, and leaves the merge to the team. It
observes the agent without editing it.

```bash
fleetopt apply <project>      # the whole loop: review, change, prove, report
fleetopt review <project>     # look only: changes nothing
```

`apply` needs nothing run before it and asks no questions.

## Install

Needs Python 3.11+, git, and a machine where Claude Code already works (a claude.ai
login, or your company's key in `~/.claude/settings.json`). Linux, macOS, Windows.

```bash
git clone <this repo> fleetopt && cd fleetopt
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Try it on a copy of the bundled fixture. About 10 minutes, a fake model, no API spend:

```bash
T=$(mktemp -d) && cp -r fixture/. "$T" && git -C "$T" init -q && git -C "$T" add -A && git -C "$T" commit -qm base
fleetopt apply "$T"
```

Use a copy: the branch is made in whatever git repository the project is in.

## What a team provides

| | |
|---|---|
| A project that runs | Its own environment and its own keys. fleetopt installs nothing and never reads the keys |
| A git repository | Changes land on a new branch. The branch you were on is never touched |
| Eval cases, for design changes | Inputs with the answers expected. Found in the project, or passed with `--evals` |

If something is missing, fleetopt lists all of it at once and stops before spending:

```
fleetopt cannot start this agent yet. 2 things to set up, then run the same command again:

  1. .env does not exist. The project reads its keys and settings from it. Copy .env.example to it and fill it in
  2. The project's environment has no module named `langgraph`. It has a uv.lock, so: uv sync
```

## What it does

| Step | |
|---|---|
| Start | Finds the agent and works out how to start it. Once per project, then remembered |
| Watch | Runs it on the team's own inputs and records every step |
| Review | Numbered findings, each with the number it rests on and the change it points to: cost (`C1`, ...), and design (`D1`, ...) with `--design` |
| Change | Code runs the same steps for every finding, in order: a session makes the one change; code commits it, tries it once, measures it, compares it with the code before, judges the answers, keeps it or undoes it. At most two attempts, the second told why the first failed |
| Report | What changed, what it gained, and a verdict computed from the measurements |

`review` is the first three steps. While the code has not changed, a review is reused:
neither command runs the agent or the reviewer twice for the same code.

## What is tried, and what is kept

| A finding that is | is tried | and kept when |
|---|---|---|
| cost: the graph keeps its nodes and edges | always | the graph is unchanged, the gain clears the noise and every request passes the judge |
| design: a node or edge removed, merged or rewired | only with `--design`, and only when the team has eval cases | something got better and every request passes the judge |
| redesign | the same | the same |

**Cost only by default.** Design is reviewed and changed only with `--design`. A change
to the design alters how an agent loops and stops, and that is where the one runaway so
far came from.

**The judge, per request.** Where an eval case covers the request, the new answer must
be correct by that case. Where none does, it must be equivalent to the original answer.
One failed request fails the change.

These rules are applied in code, before any session starts. Nothing is merged or
pushed. One commit per finding means a team can keep some and drop others.

## The level

Every review gives the agent a level: the largest kind of change it found.

| Level | Name | Meaning |
|---|---|---|
| 0 | fit | Nothing worth changing |
| 1 | wasteful | Works, and costs more than it should |
| 2 | over-built | Parts of the design do nothing |
| 3 | wrong shape | A simpler design would do the whole job |
| 4 | broken | Requests do not finish. Fixed before anything is made cheaper |

## What you get

The last thing printed is a summary, computed from what was recorded:

```
Agent    supervisor_hitl_sql_agent
Level    3 of 4, wrong shape
Found    2 cost, 1 redesign
Tried    3 changed version(s): 0 kept, 3 undone
Gained   nothing proven
Verdict  nothing left standing
Spent    $1.22 on the team's key in 27 run(s) of the agent, $2.63 by fleetopt
Branch   fleetopt/c1-c2-d1, the same code it started from
```

| Where | What |
|---|---|
| The project's repository | A new branch, one commit per finding left standing |
| `.fleetopt/runs/<time>-<project>/` | `report.md`, `summary.txt`, `patch.diff`, and `run.json`: every measurement and judgment |
| `.fleetopt/runs/review-<time>-<project>/` | `review.md` and the findings |
| `.fleetopt/fleetopt.db` | Every recorded run of the agent |

A run folder holds no prompts and no outputs of the agent.

## Options

| Flag | Command | Meaning |
|---|---|---|
| `--design` | both | Also review the design, and with `apply` try design changes when the team has eval cases |
| `--only C1,D2` | apply | Try just these findings, then stop |
| `--evals FILE` | both | Eval cases to use. Their inputs are what the agent is run on |
| `--fresh` | review | Review again although the code has not changed |
| `--graph NAME` | both | Rarely needed. With several agents, fleetopt picks the one the team ships and says why. This overrides it |
| `--max-usd N` | both | Cap on fleetopt's own spend (5 for apply, 1 for review). The agent's calls stop at $2, below |
| `--out DIR` | both | Where records go (default `./.fleetopt`) |

`fleetopt capture <project>` is a diagnostic: it runs the agent under observation and
says how much it saw.

Settings, all optional, in the environment or `~/.config/fleetopt/env`: `FLEETOPT_MODEL`,
`FLEETOPT_REVIEW_MODEL`, `FLEETOPT_JUDGE_MODEL`, `FLEETOPT_PARALLEL`, `FLEETOPT_PRICES`,
`FLEETOPT_RUN_MINUTES`, `FLEETOPT_MAX_MINUTES`, `FLEETOPT_TEAM_USD`.

## What ends a run

Before it starts, `apply` says how many findings it will try, about how long it will
take and how often it will run the agent. Then it says which finding it is on.

| Limit | Default | Then |
|---|---|---|
| A changed agent takes far more steps than the original | 3 times as many (6 when the original finishes nothing) | That run is stopped within seconds, and the change undone |
| One run of the agent | 15 minutes | That run is stopped, with everything it started |
| Attempts per finding | 2 | The finding is left undone |
| Spend on the team's key | $2 (`FLEETOPT_TEAM_USD`) | No further run of the agent. It reports what it has |
| fleetopt's own spend | `--max-usd`, $5 | No further change is made |
| The whole run | 2 hours (`FLEETOPT_MAX_MINUTES`) | No further run of the agent |

## What keeps it safe

Enforced, not asked. Each is a test in `tests/test_invariants.py`.

| | Rule |
|---|---|
| The team's code | Edited only inside the project, and only on a new branch |
| Publishing | No `git push`, no merge |
| The team's machine | Nothing installed, nothing downloaded |
| Secrets | `.env`, keys and certificates are never read |
| The verdict | Computed from the measurements, not written by the session that made the change |
| The session | No web access. Loads nothing from the operator's Claude Code or the project's `.claude/` |

## Known limits

| | |
|---|---|
| No change kept on a real agent yet | Three real agents were run unattended on 2026-09-28. All started without help and every verdict was true. None left a change standing |
| Cost of a run | Not yet measured with the new loop. About $3 to $4 per agent with the old one |
| Frameworks | LangGraph, in Python. Nothing else |
| Providers | Only Anthropic has been run. OpenAI and Gemini are priced and untested |
| Eval cases | Four are run, spread over the file. Not every case |
| The judge | Reads one run per side, of the three measured |
| One pass | `apply` tries the review's findings once each. Run it again on its branch to review the changed code |
| Agents with outside tools | One request lost to a flaky web search fails the change |
| Agents inside agents | Structural numbers cover the outer agent only |
| New files | A file git does not track yet does not count as a change to the code |

## Testing

| | Command | Cost |
|---|---|---|
| The code and the product's promises | `pytest -q` | None. About 30 seconds |
| The seven skills | `claude plugin eval fleetopt/optimizer/plugin --trust-plugin` | A few dollars on your login |
| Real agents | `tests/corpus/ledger.md` records every run | The team's key, and fleetopt's sessions |

## More

| | |
|---|---|
| `docs/design.md` | Why it is built this way, how it observes without editing, how it starts an agent, every guardrail and the incident behind it |
| `docs/fleetopt-overview.html` | A one-screen briefing |
| `tests/corpus/ledger.md` | Every run on a real agent: what was found, what broke, what was fixed |
