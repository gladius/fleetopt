# fleetopt

Makes another team's LangGraph agent cheaper, and proves each change still works. It
watches the agent without editing it, changes it on a new branch, and leaves the merge to
the team.

```bash
fleetopt review <project>     # look only: where it wastes money. Changes nothing
fleetopt apply <project>      # review, then change it on a new branch and prove each change
```

## Install

Python 3.11+, git, and a machine where Claude Code already works (a claude.ai login, or
your company's key in `~/.claude/settings.json`).

```bash
git clone <this repo> fleetopt && cd fleetopt
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Try it on a copy of the bundled fixture (a fake model, no API spend):

```bash
T=$(mktemp -d) && cp -r fixture/. "$T" && git -C "$T" init -q && git -C "$T" add -A && git -C "$T" commit -qm base
fleetopt apply "$T"
```

## How it works

One agent does the work, the way a developer would. Code holds the numbers and the limits.

| Step | The agent | Code |
|---|---|---|
| Start | reads the project and works out how to run its agent; tries it | runs each try once under watch; keeps only an entry a try proved |
| Review | reads the code and one recorded run; writes numbered findings | records every model call, token and step |
| Apply | makes one change at a time, decides what to try next and when to stop | measures (3 runs), judges the answers, keeps or undoes, owns git |

A change is kept only if something got better past the noise and the judge passed every
request: correct by the team's eval case where one covers it, the same answer as before
where none does. A change to the graph's structure is kept only with `--design`, and only
on the team's cases. One commit per kept change; nothing is merged or pushed. The verdict
is computed from the measurements, never written by the agent.

When fleetopt cannot help, it says so and stops: something only the team can provide (a
key, a service, a dependency), or an agent whose model calls it cannot see (they do not
go through LangChain).

## What ends a run

| Limit | Default |
|---|---|
| Spend on the team's key | $2 (`FLEETOPT_TEAM_USD`) |
| fleetopt's own spend | `--max-usd`, $5 for apply, $1 for review |
| The whole run | 2 hours (`FLEETOPT_MAX_MINUTES`) |
| One run of the agent | 15 minutes (`FLEETOPT_RUN_MINUTES`), stopped with everything it started |
| A changed agent's steps | 3 times the original's (6 if the original finishes nothing): stopped, and it runs once before three times |
| Attempts per finding | 2 |
| Tries to start an agent | 4 |

## What keeps it safe

Enforced, not asked; each is a test in `tests/test_invariants.py`.

- The team's code is edited only inside the project, on fleetopt's branch. No push.
- Nothing installed, nothing downloaded. `.env`, keys and certificates are never read.
- The agent runs only through fleetopt's tools, never by hand.
- The sessions have no web access and load nothing from your Claude Code or the project's `.claude/`.
- Nothing about you goes into what is sent to the agent.

## Options

| Flag | Command | Meaning |
|---|---|---|
| `--design` | both | Also review the design; with apply, try design changes (kept only on the team's cases) |
| `--only C1,D2` | apply | Try just these findings |
| `--evals FILE` | both | Eval cases (input and expected answer). Their inputs are what the agent is run on |
| `--fresh` | review | Review again although the code has not changed |
| `--graph NAME` | both | Which agent, when a project has several |
| `--out DIR` | both | Where records go (default `./.fleetopt`) |

Records: `.fleetopt/runs/<time>-<project>/` holds `report.md`, `summary.txt`, `patch.diff`
and `run.json`. No prompts or outputs of the agent are stored there.

## Known limits

- No change kept on a real agent yet.
- LangGraph in Python only. Model calls must go through LangChain to be seen.
- Only Anthropic has been run; OpenAI and Gemini are priced, untested.
- The judge reads one run per side of the three measured.

## Tests

`pytest -q`: no model is called (a test that tries fails), no spend.
