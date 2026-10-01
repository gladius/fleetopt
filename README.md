# fleetopt

Makes a LangGraph agent cheaper, and proves it still works by the team's own measure. Point
it at a project: it works out how to run the agent, finds where it wastes tokens and money,
changes it on a new branch, and keeps only what is cheaper and proven not to break it.

## What the project needs

Your normal development setup, ready: the project under git with nothing uncommitted, the
agent running in its own environment with its keys, and **something to check it against**.
fleetopt uses the strongest you have and says which:

| You have | How a change is proven |
|---|---|
| An eval suite (any tool: pytest, deepeval, LangSmith, Galileo, your own script) | Run before and after: nothing that passed may fail |
| A golden dataset: requests with expected answers, one or many (CSV, JSON, any text file) | Each answer checked against the expected one, before and after |
| Examples of what your agent is sent, one or many (a test case, a sample ticket, records in a data or metrics file, an example in the README) | Each answer after must be as good as the one before |

It looks anywhere in the project, whatever the files are called. With none of these it says
what it looked at, why each fell short, and stops before spending anything. To point it at
the right thing yourself: `--evals "pytest tests/evals"` or `--evals metrics/cases.csv`.

```bash
fleetopt review <project>     # look only: where it wastes tokens and money. Changes nothing
fleetopt apply <project>      # look, change it on a new branch, prove the changes on your checks
```

## Install

Python 3.11+, git, and a machine where Claude Code already works (a claude.ai login, or
your company's key in `~/.claude/settings.json`).

```bash
git clone <this repo> fleetopt && cd fleetopt
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Try `review` on a copy of the bundled fixture (a fake model, nothing spent on any key):

```bash
T=$(mktemp -d) && cp -r fixture/. "$T" && git -C "$T" init -q && git -C "$T" add -A && git -C "$T" commit -qm base
fleetopt review "$T"
```

## How it works

One Claude session does the work, the way an expert would; code holds the numbers and the limits.
Which expert is `--expert` (cost by default): an expert is a guide and skills in
`fleetopt/experts/<name>/`, working with the same tools and under the same rules as every other.

1. **Start it.** The agent reads the project and works out how its agent runs here: which
   graph, which env files, which folder. Then it tries it on one input. It is remembered per
   project. While getting started it may ask you up to three things only you know (which env
   file, which graph you ship); with no terminal, or `--no-ask`, it never asks. Run fleetopt from the terminal where your agent works for you: that terminal's
   environment and cloud logins are what the agent runs with.
2. **Check it as it is** (apply): your eval suite, or its answers to your golden dataset or examples.
3. **Measure it** as it is: 3 runs, every model call, token and step recorded.
4. **Find the waste**: prompts that grow, caching not used, a bigger model than a step
   needs, output with no limit, loops that do not stop early, repeated calls, oversized
   tool lists.
5. **Change it** (apply only): every change the evidence supports, each its own commit,
   named in plain words, measured together.
6. **Prove it**: kept only if it is cheaper past the noise and nothing broke on your
   check, run again. A separate model call reads the results before and after; the agent
   that made the change never grades it. With no eval suite it reads each request's
   answers whole, beside two runs of the original, so the agent's own variation is not
   taken for a break. Your tests and evals are never changed, and changes to the graph's
   structure are out of scope. Nothing is merged or pushed.

The report says which nodes of the graph the requests reached and which never ran, by their
full path when graphs are nested (`research:model` is not `math:model`). A change to the code
of a node that never ran is proven by nothing: fleetopt reports it, and refuses to keep it.

When it cannot help it says so and stops: something only the team can provide (a key, a
service, a dependency, something to check against), or an agent whose model calls it cannot
see (they do not go through LangChain).

## What ends a run

| Limit | Default |
|---|---|
| What the agent spends on its own API key, in fleetopt's runs of it | $2 (`FLEETOPT_TEAM_USD`) |
| What fleetopt's own work spends on your Claude login | `--max-usd`: $5 for apply, $2 for review |
| The whole run | 2 hours (`FLEETOPT_MAX_MINUTES`) |
| One run of the agent | 15 minutes (`FLEETOPT_RUN_MINUTES`), stopped with everything it started |
| A changed agent's steps | 3 times the original's: stopped, and changed code runs once before three times |
| Tries to start an agent | 4 |
| Questions to you, before the first measurement | 3, five minutes each to answer |
| One run of your evals | 30 minutes (`FLEETOPT_EVAL_MINUTES`) |
| Reading your eval results, or one request's answers | 5 minutes; no answer means the change is not kept |

## What keeps it safe

Enforced, not asked; each is a test in `tests/test_invariants.py`.

- The team's code is edited only inside the project, on fleetopt's branch. No push.
- Nothing installed, nothing downloaded. fleetopt's AI is shown the names your env files set,
  never their values: its file reader is blocked from `.env`, keys and certificates (its
  shell is not blocked from them). The values are loaded into your agent's and your evals'
  own processes, and fleetopt never logs them.
- The agent runs only through fleetopt's tools, never by hand.
- No web access, and nothing loaded from your Claude Code or the project's `.claude/`.
- Nothing about you goes into what is sent to the agent.

## Options

| Flag | Meaning |
|---|---|
| `--evals WHAT` | What to check the agent with: your eval command (`pytest tests/evals`) or a file of test cases, expected answers or example requests. Found in the project otherwise |
| `--graph NAME` | Which agent, when a project has several |
| `--expert NAME` | Which expert looks at it (default `cost`) |
| `--no-ask` | Never ask a question at the terminal |
| `--max-usd N` | Cap on fleetopt's own spend |
| `--out DIR` | Where records go (default `./.fleetopt`) |

Models: the agent runs on `sonnet` (falling back to `opus`) and the reader on `haiku` (falling
back to `sonnet`): whatever those names give in your Claude setup, login, gateway, Bedrock or
Vertex. The model used is printed at the start. `FLEETOPT_MODEL` and `FLEETOPT_JUDGE_MODEL` override.

Each run leaves `.fleetopt/runs/<time>-<project>/`: `report.md`, `run.json`, `log.txt` (the
whole session, live), `evals-N.log` (each eval run's output), and `patch.diff` when something
was kept. The log, the eval output and `.fleetopt/fleetopt.db` contain the agent's prompts and
answers: treat them like the project's own logs.

## Known limits

- LangGraph in Python only; model calls must go through LangChain to be seen.
- Only Anthropic has been run; OpenAI and Gemini are priced, untested.
- Reading eval results has been tried on pytest output only; checking by a golden dataset or
  examples is tested in code, not yet on a real run.
- A change is tied to a node by the function the graph runs for it. A change to a helper or a
  prompt constant used only by a node the requests never ran is not caught.
- A golden dataset has to be a text file in the project. A spreadsheet, or a dataset kept only
  in an eval service, counts through your eval script, or not at all.

## More

`overview.html` is a one-page briefing on what fleetopt does and how it works.

## Tests

`pytest -q`: no model is called (a test that tries fails), no spend.
