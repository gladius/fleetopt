# fleetopt

Makes a LangGraph agent cheaper and simpler, **without editing it to observe it**,
and proves each change before claiming it.

```bash
fleetopt apply <project>      # review it, change it on a new branch, prove each change
fleetopt review <project>     # look only: the first half of the above, changes nothing
```

`apply` is the whole loop and needs nothing run before it. Everything else —
capturing, querying, measuring, comparing, judging — is a tool it calls itself.

## Quick start

**Needs:** Python 3.11+, git, and a machine where Claude Code already works (a
claude.ai login, or your company's key in `~/.claude/settings.json`). No other
secret. There is no compile step; the editable install below is the whole build.
Linux, macOS and Windows (activate with `.venv\Scripts\activate` there; console
output never trips on a legacy code page).

```bash
git clone <this repo> fleetopt && cd fleetopt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # fleetopt + the fixture's langgraph; ~250 MB (bundled Claude Code binary)
# smoke test on a copy of the bundled fixture: 3-5 min, no API spend. Keep the venv
# active - the fixture runs on this venv's langgraph.
T=$(mktemp -d) && cp -r fixture/. "$T" && git -C "$T" init -q && git -C "$T" add -A && git -C "$T" commit -qm base
fleetopt apply "$T"
```

It prints the review (numbered findings), then the changes it tries, and ends with a
`fleetopt verdict` line computed from the measurements. It leaves a new branch in
that temp repo with one commit per finding left standing. The copy matters: the
branch is made in whatever git repo the target is in, and `fixture/` inside this
checkout would mean branching fleetopt itself.

**On a real project** (must be a git repo; the patch lands on a new branch, the branch
you were on is never touched):

```bash
fleetopt apply ~/work/their-agent                # the whole loop
fleetopt review ~/work/their-agent               # look first; apply then starts from this review
fleetopt apply ~/work/their-agent --only C1,D2   # just the findings a person chose, nothing else
```

**Look, then change.** `review` runs the agent once and reports two things, as
numbered findings with the trace numbers they rest on: where the agent wastes money
(`C1`, `C2`, ...) and whether its design fits its job (`D1`, ...). It changes nothing
and claims no saving: it saw one run. `apply` tries the findings on a new branch, one
commit each, measures and judges each one, undoes what fails, and then looks again,
because a fix often uncovers the next cost. The review is where it starts, not a fence.

| A finding that is | `apply` does |
|---|---|
| cost (`C`): the graph keeps its nodes and edges | tries it, keeps it only if the saving clears the noise and the judge passes |
| design, tier one (`D`): a node or edge removed, merged or rewired mechanically | tries it only when the team's eval cases exist and cover that path |
| design, tier two (`D`): a redesign | nothing, unless a person names it with `--only`, and then only with eval cases |

These rules are applied in code before the session starts. The reviewer says what
kind of change a finding is and what could go wrong; it does not get to decide what is
tried (left to choose, one marked every cost finding "needs cases" and nothing was).
**The person decides at the merge:** nothing is ever merged or pushed, and one commit
per finding means a team can keep some and drop others.

**The same code is never reviewed twice.** Every capture records a fingerprint of the
code (commit plus uncommitted changes). While it has not changed, `review` shows the
saved review and `apply` starts from it; neither runs the agent or a reviewer again.
(A new file git does not track yet does not change the fingerprint.)

There is no run command to pass and nothing to approve. **How an agent is started is a
fact about the project, so fleetopt works it out once per project and remembers it**:

| What it needs | Where it looks |
|---|---|
| The agent | the project's `langgraph.json`; otherwise a compiled graph, or a graph factory that takes no arguments, in its source. With several, the one the team ships and tests |
| The interpreter | the project's own `.venv` / `venv`. fleetopt never installs anything |
| Keys and settings | the env file the project names, loaded inside the agent's own process. fleetopt never reads it |
| Inputs | the team's eval cases; otherwise a file of inputs the project keeps; otherwise four written from its README |

**The team provides a project that runs; fleetopt adds nothing to a developer's
machine.** Before anything is spent it checks, in a few seconds and without calling a
model, that the agent loads in the project's environment and that a key for one of
the providers it uses is set. If not, it lists everything that is missing at once,
each with the fix in the project's own terms, and stops:

```
fleetopt cannot start this agent yet. 2 things to set up, then run the same command again:

  1. .env does not exist. The project reads its keys and settings from it. Copy .env.example to it and fill it in
  2. The project's environment has no module named `langgraph`. It has a uv.lock, so: uv sync
```

It then starts the agent once with one input to prove the answer, and saves it as a
small JSON file under `.fleetopt/entries/` (never in the team's repo). A call the
provider refuses, or a service that cannot be reached, is reported the same way. What
is fleetopt's to work out is worked out: if the agent loads and still does not run,
a read-only session reads the source and the error, proposes what to change (which
graph, a setting, a tenant id, the shape of the input), and fleetopt tries that by
running it, twice at most. From then on fleetopt runs its own driver
(`fleetopt/drive/driver.py`) against that entry and nothing else: never a command
somebody guessed. Another framework is another way of filling in the same entry.

| Flag | Meaning |
|---|---|
| `--graph NAME` | Rarely needed. When a project has several agents, fleetopt reads its README, code and the team's own tests, picks the one the team ships, says why in one line and remembers the choice. `--graph` overrides that: a name from its `langgraph.json`, or `file.py:variable`. |
| `--only IDS` | `apply` only. Try just these findings of the review, for example `C1,D2`, and stop. Without it: everything the review marked safe to try, then whatever those fixes uncover. |
| `--fresh` | `review` only. Run the agent and review again although the code has not changed. |
| `--evals FILE` | Eval cases (input + expected answer) as JSONL/JSON, a LangSmith dataset export, or deepeval tests. Optional: they are looked for in the repo otherwise. With cases, the judge reports correctness pass rates before and after, not just "unchanged". |
| `--max-usd N` | Stop fleetopt's own session once its spend reaches N (default 5 for apply, 1 for review). The target's API calls are its own bill. |
| `--out DIR` | Where captures, entries and run records go (default `./.fleetopt`, relative to where you run it). |

**What you get:** the report in the terminal (one row per finding: what happened,
measured before/after in dollars first, judge verdict), the changes committed on a branch in the target repo,
every measurement in `.fleetopt/fleetopt.db`, and a run folder at
`.fleetopt/runs/<timestamp>-<project>/` with `report.md`, `run.json` (what every
tool established: medians, compare, judge, skills used, turns, cost, and the computed verdict),
`patch.diff` and `log.txt`; a review leaves its own folder with `review.md` and the findings in `run.json`. The run folder is what feeds the ledger and what you
would send back to the central team; it holds no prompts or outputs of the target.
The run also prints which credential it is using as its first line.

**One debug command**, for when a run comes back empty on a repo: `fleetopt capture <project>`
starts the agent under instrumentation and nothing more. If it shows `0 runs`, the agent ran
without going through LangChain's callbacks.

**When it stops early:** a declined prompt is final for that session and it
reports from read-only evidence. `within noise` means the change did not clear
the baseline's own spread and is not a saving. Hitting `--max-usd` ends the run
with whatever was measured so far.

## Why it is built this way

Cheaper is trivially achievable by making an agent dumber: drop a model tier,
truncate the context, skip a step. So the hard part was never finding savings, it
was being able to hand another team a number they can trust.

That splits the system in two:

| | Who does it | Why |
|---|---|---|
| Discovery, diagnosis, the fix | **The agent** | Can't be enumerated. There are too many patterns and too many project shapes to encode as a workflow |
| Measurement and the equivalence gate | **Deterministic code** | The output is a claim handed to another team. It has to be reproducible, and an LLM eyeballing medians will report noise as savings |

```
fleetopt/
  cli.py          one product command, two debug commands
  config.py       the one place for knobs, auth policy, and what the target's process may inherit
  probe/          observes an unmodified target — hooks, runner, sqlite store
  evidence/       turns observations into defensible claims — measure, pricing, judge
  optimizer/      the agent — session, tools, SKILL.md (router) + plugin/skills/ (one skill per decision)
                  plugin/evals/  eval cases for those skills (`claude plugin eval`)
tests/            pytest for the deterministic code; corpus/ = real-repo ledger and candidate list
```

`judge` lives in `evidence/`, not `optimizer/`, on purpose: the optimizer
proposed the patch and wants it to pass. It does not get to grade its own work.
It runs through the same Claude Code binary as the optimizer (same credential, no
second setup) but in its own process with a clean context, no tools and one turn;
with `tools=[]` that is ~400 input tokens, the price of a direct API call.

## Zero-edit capture

Python's `site` module auto-imports `sitecustomize` at interpreter startup. We
put ours on `PYTHONPATH` and run the target's own command, so instrumentation
lands before any of its code executes. Two things get installed:

- **A global tracer** registered via `register_configure_hook`. LangChain
  *appends* it to every callback manager, so it runs alongside whatever the
  project already configured — existing LangSmith traces keep flowing untouched.
  This is the same `Run` stream LangSmith itself consumes, so prompts,
  completions, token counts, cache reads and node attribution are all available.
- **A patch on `StateGraph.compile`**, so every compiled graph snapshots its own
  topology (`get_graph(xray=True)` + mermaid). No need to locate the graph object
  in the target's source.

The hook writes JSONL — append-only and lock-free, so nothing it does can
deadlock someone else's program. The CLI ingests into SQLite after the process
exits.

## Guardrails

These exist because each one is a way to produce a confident wrong number.

- **Built-in tools by allowlist.** Read, Grep, Glob, Bash, Edit, Write, Skill. Claude Code's default set also includes web
  fetch and search, cron, git worktrees, messaging and scheduling; none of it is loaded.
- **Noise floor.** `compare` uses the baseline's own run-to-run spread. A delta
  inside that spread reports as `within noise`, not as a saving.
- **Code-state fingerprint.** Every session records `git HEAD` + a hash of the
  working diff. Averaging sessions that ran against different source is refused
  outright — otherwise re-using a label after an edit silently medians the before
  and the after together.
- **Labels are scoped to the project.** The capture db is shared, so `baseline`
  from one repo must never pool with `baseline` from another. Enforced in
  `tools._ids`, on top of the code-state fingerprint.
- **Unpriced means `None`, never `$0`.** A savings figure built on a model with
  no price is worse than no figure. Rates live in `evidence/pricing.py` and
  **will rot**; override via `FLEETOPT_PRICES`.
- **Measurements run concurrently.** Not for speed: the optimizer is an LLM session
  whose prompt cache expires after 5 minutes of silence. Three serial 70-second
  target runs blew through that and forced a full re-write of its 67K-token context -
  three times in one run, a third of the optimizer's bill. `measure` now executes the
  n runs at once (`FLEETOPT_PARALLEL`, default n, max 5) and ingests serially. Token
  and dollar medians are identical to serial (verified on the fixture); `wall_ms`
  picks up contention, so set `FLEETOPT_PARALLEL=1` when latency is the subject.
- **The target runs only through `measure`.** A `PreToolUse` hook denies Bash commands
  that would run it by hand (`pytest`, `python x.py`, `langgraph dev`, the driver
  itself). Running it by hand spent the team's tokens twice, captured nothing, and fed
  12K tokens of tracebacks into the optimizer's context.
- **How the agent is started is not the optimizer's to decide.** fleetopt settles it
  before the session begins (see Quick start) and the optimizer has no tool to change
  it. It once spent 13 turns re-deriving a command that was already correct.
- **The verdict is computed, not written.** After the agent's report fleetopt prints
  what the recorded measurements support. Only a judged comparison of two different
  code states counts, and one failed judgment of the final code is a failure. Every
  comparison states which code each side ran. Observed 2026-09-28: an optimizer
  measured its patched code under a label called `baseline-retest`, cited that as
  proof the unmodified agent had the same defect, and reported the saving as real.
- **Equivalence gates everything.** A cost reduction with a failed judge is a
  regression nobody noticed yet.

- **Correct, not only unchanged, when the team has eval cases.** The judge always
  checks that the patch left the output equivalent to the previous run. When eval
  cases with expected answers exist - a deepeval suite, JSONL, JSON, or `--evals` -
  it also grades both baseline and candidate against them and reports pass rates;
  a candidate that passes fewer cases than the baseline fails. Only runs whose
  input matches a case are graded, and the rest are reported as unmatched. Without
  cases the report says "correctness not checked" in so many words.
- **Never changes the target's environment.** Installs and downloads (`pip install`,
  `uv add`/`--with`, `poetry add`, `npm install`, `curl`, `wget`, ...) are denied in the
  optimizer's shell. A missing dependency is a finding to report, not something to
  fix. Observed 2026-09-24: handed an interpreter without langgraph,
  the agent pulled the packages with `uv run --with`. Right for a sandbox, wrong on
  someone's machine.
- **A command that has never worked here is probed once** before the remaining runs
  start, and a failed measurement returns the target's own last output lines to the
  agent. Before this, a wrong interpreter cost 15 crashed runs and 20 turns of
  guessing; the traceback had gone to the operator's terminal, not to the agent.

## Permissions

Nothing is asked, so nothing depends on somebody being there to answer. What keeps
a run safe is enforced:

| | Rule |
|---|---|
| Start the target | only fleetopt's own driver, against the entry settled for the project |
| Shell | cannot install or download, cannot run the target by hand, cannot `git push` |
| Edit, Write | only inside the project, and only once a new branch exists |
| Read | never `.env*`, keys, certificates, `*secret*`, `*credential*`, `.netrc` and friends (`config.DENY_READS`; verified live 2026-09-24) |
| Built-in tools | Read, Grep, Glob, Bash, Edit, Write, Skill. No web access, no scheduler, no subagents |
| Spend | `--max-usd` caps fleetopt's own session; the target's API calls are not included |

A session loads nothing from the operator's Claude Code (plugins, skills, hooks, MCP
servers) and nothing from the target repo's `.claude/`, so another team's hooks,
permissions and CLAUDE.md never run inside it; only the credential-bearing keys of
`~/.claude/settings.json` are passed through. Subprocesses are spawned with telemetry
and auto-memory off. Each of these is a test in `tests/test_invariants.py`.

## Setup

```bash
pip install -e ".[dev]"                   # fleetopt + the fixture's langgraph
```

**Credentials: none to configure.** The optimizer and the judge run through the
Claude Code binary bundled with the Agent SDK, so they authenticate the way Claude
Code on the machine does, in Claude Code's own precedence: cloud-provider
switches, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, `apiKeyHelper`, then the
claude.ai login. Those variables can sit in the shell or in the `env` block of
`~/.claude/settings.json`, which is how a company gateway is usually distributed.
Only that `env` block and `apiKeyHelper` are read from the file: the operator's own
plugins, skills, hooks and MCP servers never enter a run, and neither does the
target repo's `.claude/`.
`fleetopt apply` prints which one it found. The target agent is separate: it
runs with the operator's real environment and its own keys, and never inherits
anything fleetopt read from its own config files.

Knobs are optional and live in `.env` (per project) or `~/.config/fleetopt/env`
(per machine); real environment wins over both. `FLEETOPT_MODEL` drives the
sessions (default `claude-sonnet-5`), `FLEETOPT_REVIEW_MODEL` the reviewer alone,
`FLEETOPT_JUDGE_MODEL` the equivalence check (default Haiku 4.5),
`FLEETOPT_PARALLEL` sets measurement concurrency.

**What a claude.ai login covers:** the optimizer and the judge, not the agent
under test. The fixture uses a fake model, so its runs are free of API spend.
A real target makes its own provider calls with its own key.

## Testing

Three layers, each the standard tool for what it checks. None of them touches a
target's API key.

| Layer | What it checks | Command | Cost |
|---|---|---|---|
| invariants | the product's promises, one test each: a session inherits nothing from the operator, has no web access, cannot read secrets; nothing is installed or hand-run; the probe only observes; a change inside the noise is not a saving; the judge fails closed; the reviewer can only look; structural patches wait for eval cases | `pytest -q tests/test_invariants.py` | none, under 1 s |
| pytest | the deterministic code: eval discovery, label scoping, the Bash guard, the noise floor, session isolation, and one capture of the fixture | `pytest -q` | none, about 2 s |
| plugin eval | the six skills: with the plugin loaded the agent reaches each skill's conclusion (the 4,096-token Haiku minimum, effort before tier, the ~10K schema threshold...); the default with/without arm shows whether the skill made the difference | `claude plugin eval fleetopt/optimizer/plugin --trust-plugin` | 12 short agent runs on your login, a few dollars |
| corpus ledger | the whole loop on real agents | `fleetopt apply targets/<repo>`, then a line in `tests/corpus/ledger.md` | the target's own tokens plus fleetopt's sessions |

The skill evals live next to the skills because the runner looks for them below the
plugin. `tests/corpus/corpus_build.py` regenerates the candidate list from GitHub
search; export `GITHUB_TOKEN` for the full pass. Every break found on a real repo
becomes a pytest case; every recurring pattern becomes a line in a skill.

Two fixtures, both on a fake model so nothing costs anything: `fixture/agent.py` has a
planted cost defect (a node that re-sends its whole history); `fixture/supervisor.py`
has planted structural smells for the architecture review (a router that only ever
takes one branch, a supervisor whose three workers always run in the same order, a
reflection loop that never changes the draft).

## Known gaps

- **Agents nested inside agents blur the structural numbers.** The probe records which
  node a call ran under, not which graph that node belongs to, and every tool-calling
  agent names its nodes `model` and `tools`. The numbers are computed for the one graph
  that accounts for most of what ran and leave nested agents out, so a supervisor whose
  workers are themselves agents gets numbers for the supervisor only. Recording the
  graph a node belongs to (LangGraph's checkpoint namespace) is the fix.

- **The fixture cannot exercise the judge.** Its fake model returns fixed strings
  regardless of input, so before/after outputs are byte-identical and the gate
  passes trivially. The real repo does exercise it, and it passes correctly there.
- **No optimization has yet been proven to save anything.** The fixture's win was
  real but synthetic; the real repo's candidate landed within noise. A confirmed
  saving on someone else's code is still outstanding.
- **Eval cases are graded, not driven.** Cases are matched to whatever the run
  command exercised; fleetopt does not yet invoke the agent per case, and it cannot
  fetch LangSmith, Galileo or promptfoo datasets - it reports their names for a
  human to export.
- **The decision skills are untested on a real run.** `SKILL.md` is a router; the
  mechanics live in five Agent Skills under `optimizer/plugin/skills/` (`caching`,
  `prompt-growth`, `model-tier`, `redundant-work`, `tool-surface`), loaded as a local
  plugin and triggered by their descriptions. They also work standalone: point Claude
  Code at the plugin and ask "is my caching set up right?" in any repo. Whether the
  optimizer invokes the right one at the right time has been observed once, on
  the fixture (2026-09-24): it loaded `fleetopt:prompt-growth` after seeing
  `prompt_chars` climb and before editing. Not yet observed on a real repo.
- Pricing covers Anthropic, OpenAI and Gemini list rates as of 2026-09-23; hosted
  variants (Bedrock/Vertex/Azure ids) and >200K-context tiers are not priced.
- Only LangGraph. ADK emits OpenTelemetry natively (1.17+), so its adapter should
  be an OTLP span processor installed the same way, not a second callback hook.
- No train/test split. Measuring and editing against the same handful of cases
  produces a beautiful number and a production regression.
- Async and streaming invocation paths are untested.
