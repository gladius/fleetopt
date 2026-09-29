# fleetopt

Makes a LangGraph agent cheaper, and proves it on the team's own evals. Point it at a
project: it works out how to run the agent, finds where it wastes tokens and money,
changes it on a new branch, and keeps only what is cheaper and still passes the team's evals.

## What the project needs

Your normal development setup, ready: the project under git, the agent running in its own
environment with its keys, and **an eval suite you run** (any tool: pytest, deepeval,
LangSmith, Galileo, your own script). fleetopt checks first; if something is missing it
says what, and stops before spending anything. Without evals: "No evals found: create
them, then run this again."

```bash
fleetopt review <project>     # look only: where it wastes tokens and money. Changes nothing
fleetopt apply <project>      # look, change it on a new branch, and prove each change
```

## Install

Python 3.11+, git, and a machine where Claude Code already works (a claude.ai login, or
your company's key in `~/.claude/settings.json`).

```bash
git clone <this repo> fleetopt && cd fleetopt
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Try it on a copy of the bundled fixture (a fake model, nothing spent on any key):

```bash
T=$(mktemp -d) && cp -r fixture/. "$T" && git -C "$T" init -q && git -C "$T" add -A && git -C "$T" commit -qm base
fleetopt apply "$T"
```

## How it works

One Claude agent does the work, the way an expert would; code holds the numbers and the limits.

1. **Start it.** The agent reads the project and works out how to run its agent, then
   tries it on one input. It is remembered per project.
2. **Run your evals** on the code as it is: that is what "still works" means.
3. **Measure it** as it is: 3 runs, every model call, token and step recorded.
4. **Find the waste**: prompts that grow, caching not used, a bigger model than a step
   needs, output with no limit, loops that do not stop early, repeated calls, oversized
   tool lists.
5. **Change it** (apply only): every change the evidence supports, each its own commit,
   named in plain words, measured together.
6. **Prove it**: kept only if it is cheaper past the noise and your evals, run again,
   still pass everything they passed before. A separate model call reads the two eval
   results; the agent that made the change never grades it. Your tests and evals are never
   changed, and changes to the graph's structure are out of scope. Nothing is merged or
   pushed.

When it cannot help it says so and stops: something only the team can provide (a key, a
service, a dependency, evals), or an agent whose model calls it cannot see (they do not go
through LangChain).

## What ends a run

| Limit | Default |
|---|---|
| Spend on the team's key | $2 (`FLEETOPT_TEAM_USD`) |
| fleetopt's own spend | `--max-usd`: $5 for apply, $2 for review |
| The whole run | 2 hours (`FLEETOPT_MAX_MINUTES`) |
| One run of the agent | 15 minutes (`FLEETOPT_RUN_MINUTES`), stopped with everything it started |
| A changed agent's steps | 3 times the original's: stopped, and changed code runs once before three times |
| Tries to start an agent | 4 |
| One run of your evals | 30 minutes (`FLEETOPT_EVAL_MINUTES`) |

## What keeps it safe

Enforced, not asked; each is a test in `tests/test_invariants.py`.

- The team's code is edited only inside the project, on fleetopt's branch. No push.
- Nothing installed, nothing downloaded. `.env`, keys and certificates are never read.
- The agent runs only through fleetopt's tools, never by hand.
- No web access, and nothing loaded from your Claude Code or the project's `.claude/`.
- Nothing about you goes into what is sent to the agent.

## Options

| Flag | Meaning |
|---|---|
| `--evals COMMAND` | How you run your evals, e.g. `pytest tests/evals`. Found in the project otherwise |
| `--graph NAME` | Which agent, when a project has several |
| `--max-usd N` | Cap on fleetopt's own spend |
| `--out DIR` | Where records go (default `./.fleetopt`) |

Each run leaves `.fleetopt/runs/<time>-<project>/`: `report.md`, `run.json`, `log.txt`,
and `patch.diff` when something was kept. No prompts or outputs of the agent are stored.

## Known limits

- LangGraph in Python only; model calls must go through LangChain to be seen.
- Only Anthropic has been run; OpenAI and Gemini are priced, untested.
- The judge reads one run per side of the three measured.

## Tests

`pytest -q`: no model is called (a test that tries fails), no spend.
