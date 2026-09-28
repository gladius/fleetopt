# fleetopt: why it is built this way

For someone changing fleetopt, or deciding whether to trust it. How to use it is in
the README.

## The split

Cheaper is trivially achievable by making an agent dumber: drop a model tier,
truncate the context, skip a step. So the hard part was never finding savings, it
was being able to hand another team a number they can trust.

That splits the system in two:

| | Who does it | Why |
|---|---|---|
| Discovery, diagnosis, the fix | **A model** | Can't be enumerated. There are too many patterns and too many project shapes to encode as a workflow |
| Measurement, the judge's rule, the verdict, what may be tried | **Deterministic code** | The output is a claim handed to another team. It has to be reproducible, and a model eyeballing medians will report noise as savings |

```
fleetopt/
  cli.py          apply, review, and the capture diagnostic
  config.py       the one place for knobs, auth policy, and what the agent's process may inherit
  drive/          starting an agent: entry (what to start and how), driver (run inside the
                  project's interpreter), setup (the steps that need a model)
  probe/          observes an unmodified agent: hooks, runner, sqlite store
  evidence/       turns observations into defensible claims: measure, shape, pricing, evals, judge
  optimizer/      the sessions: review (looks), session (changes), tools, the guides
                  SKILL.md (how to work) and COST.md (what to look for)
                  plugin/skills/  one skill per decision; plugin/evals/ their eval cases
tests/            pytest for the deterministic code; corpus/ = the ledger of runs on real agents
```

Three sessions, each with its own context, none grading its own work:

| Session | May | May not |
|---|---|---|
| The reviewer | read source, query the recorded runs | edit, run anything, measure |
| The session that changes | edit on a new branch, measure, ask for a judgment | decide the verdict, decide what may be tried |
| The judge | see the task and the answers | see the change or the reasoning behind it |

The judge runs through the same Claude Code binary (same credential, no second setup)
but in its own process, no tools, one turn: about 400 input tokens a call.

## How an agent is started

How an agent is started is a fact about the project, so fleetopt works it out once per
project and saves it as a small JSON file under `.fleetopt/entries/`, never in the
team's repository. From then on it runs its own driver against that entry: never a
command somebody guessed. Another framework is another way of filling in the same entry.

| What it needs | Where it looks |
|---|---|
| The agent | the project's `langgraph.json`; otherwise a compiled graph, or a graph factory that takes no arguments, in its source. With several, the one the team ships and tests |
| The interpreter | the project's own `.venv` / `venv`. fleetopt never installs anything |
| Keys and settings | the env file the project names, loaded inside the agent's own process. fleetopt never reads it |
| Inputs | eval cases passed with `--evals`; otherwise the team's eval cases; otherwise a file of inputs the project keeps; otherwise four written from its README |

Before anything is spent it checks, in a few seconds and without calling a model, that
the agent loads and that a key for one of its providers is set. Then it starts the
agent once with one input.

| The first request | fleetopt |
|---|---|
| is refused by the provider (a wrong key, no credit), or a service cannot be reached | stops and says so: the team's to fix |
| is refused for want of a key the project does not have, while it has one for another provider it supports | finds the setting that selects that provider and uses it. Key names only, never values |
| fails on how the agent was called (which graph, a tenant id, the shape of the input) | changes the entry and tries again, twice at most |
| fails because the agent expects a hosting platform to hand it a store or a run context | attaches an empty in-memory store and passes the context, as `langgraph dev` does |
| fails on the agent's own bug, after the model had answered | carries on. That agent starts; it does not finish. It is reviewed as broken (level 4) |

## Observing without editing

Python's `site` module auto-imports `sitecustomize` at interpreter startup. We put ours
on `PYTHONPATH` and run the driver in the project's interpreter, so instrumentation
lands before any of the project's code executes. Two things get installed:

- **A global tracer** registered via `register_configure_hook`. LangChain *appends* it
  to every callback manager, so it runs alongside whatever the project already
  configured: existing LangSmith traces keep flowing untouched. This is the same `Run`
  stream LangSmith itself consumes, so prompts, completions, token counts, cache reads
  and node attribution are all available.
- **A patch on `StateGraph.compile`**, so every compiled graph snapshots its own
  topology. No need to locate the graph object in the project's source.

The hook writes JSONL, append-only and lock-free, so nothing it does can deadlock
someone else's program. fleetopt ingests into SQLite after the process exits. The
probe only observes: it never changes what the agent does.

## Guardrails, and the incident behind each

Each one is a way to produce a confident wrong number, and most were found by running.

**About the numbers**

- **Noise floor.** `compare` uses the baseline's own run-to-run spread, and never less
  than 2% of the baseline. A delta inside it reports as `within noise`, not as a saving.
- **Code fingerprint.** Every recorded run carries `git HEAD` plus a hash of the working
  diff. Averaging runs of different code is refused: re-using a label after an edit
  would median the before and the after together.
- **Labels are scoped to the project.** The database is shared, so `baseline` from one
  repository never pools with `baseline` from another.
- **Unpriced means `None`, never `$0`.** A saving built on a model with no price is
  worse than no figure. Rates live in `evidence/pricing.py` and will rot; override with
  `FLEETOPT_PRICES`.
- **Finished requests is a metric.** A design that crashes after five calls is
  "cheaper" than one that finishes. `compare` reports finished requests and cost per
  finished request. Observed: a working redesign was called a cost regression against
  a baseline that finished nothing.
- **Measurements run concurrently**, not for speed: the session calling `measure` has a
  prompt cache that expires after 5 minutes of silence. Three serial 70-second runs
  forced a full re-write of its context three times in one run. `FLEETOPT_PARALLEL=1`
  when latency is the subject.

**About the verdict**

- **Computed, not written.** After the session's report fleetopt prints what the
  recorded measurements support. Observed 2026-09-28: a session measured its patched
  code under a label called `baseline-retest`, cited that as proof the unmodified agent
  had the same defect, and reported the saving as real.
- **About the code left on the branch.** A change that failed and was undone does not
  condemn the ones standing, and code nobody judged is not proven by its neighbours.
- **What may be tried is a rule, not the reviewer's opinion.** Left to choose, a
  reviewer marked every cost finding "needs cases" and nothing was tried.
- **The judge's rule is per request.** Requiring equivalence everywhere made a design
  upgrade unprovable: a better design answers in other words. Observed: the team's
  cases 4 of 4 before and after, failed on 2 of 4 answers worded differently. So where
  a case covers a request the answer must be correct by it, and where none does it must
  be equivalent. Right before and wrong after fails whatever a judge thinks of the
  likeness. A request that never finished is judged on its case, there being no old answer.
- **Eval cases are the inputs, not only the answers.** Observed: cases were supplied and
  the inputs came from a file in the project. No request matched a case, and a redesign
  that took an agent from 0 of 6 finished requests to 4 of 6 had nothing to be judged on.

**About the sessions**

- **Built-in tools by allowlist.** Read, Grep, Glob, Bash, Edit, Write, Skill. Claude
  Code's default set also includes web fetch and search, scheduling and messaging; none
  of it is loaded. Observed: a session scheduled a wake-up to "wait" for a subagent.
- **The agent runs only through `measure`.** A hook denies shell commands that would run
  it by hand. Doing so spent the team's tokens twice, recorded nothing, and fed 12K
  tokens of tracebacks into the session.
- **How the agent is started is not the session's to decide.** It once spent 13 turns
  re-deriving a command that was already correct. It has no tool to change it.
- **A refusal is an answer.** Observed 2026-09-28: a session whose edits were refused
  got the same change in through `sed`, a glob and git plumbing. Its instructions now
  forbid looking for another way.
- **The environment is not ours to change.** Installs and downloads are denied.
  Observed 2026-09-24: handed an interpreter without langgraph, a session pulled the
  packages from the internet. Right for a sandbox, wrong on someone's machine.
- **A command that has never worked is tried once** before the remaining runs start. A
  wrong interpreter once cost 15 crashed runs and 20 turns of guessing.
- **Nothing about the operator goes to the agent.** Observed 2026-09-28: inputs fleetopt
  wrote opened with the operator's first name. They travel to the agent's provider.
- **Reviewer as a separate session, not a subagent.** A subagent ran in the background,
  the caller ended its turn, and the review was lost. Twice.

## Permissions

Nothing is asked, so nothing depends on somebody being there to answer.

| | Rule |
|---|---|
| Start the agent | only fleetopt's own driver, against the entry settled for the project |
| Shell | cannot install or download, cannot run the agent by hand, cannot `git push` |
| Edit, Write | only inside the project, and only once a new branch exists |
| Read | never `.env*`, keys, certificates, `*secret*`, `*credential*`, `.netrc` (`config.DENY_READS`) |
| Built-in tools | Read, Grep, Glob, Bash, Edit, Write, Skill. No web access, no scheduler, no subagents |
| Spend | `--max-usd` caps fleetopt's own session; the agent's API calls are not included |

A session loads nothing from the operator's Claude Code (plugins, skills, hooks, MCP
servers) and nothing from the project's `.claude/`, so another team's hooks, permissions
and CLAUDE.md never run inside it. Subprocesses are spawned with telemetry and
auto-memory off.

## Credentials

None to configure. The sessions and the judge run through the Claude Code binary
bundled with the Agent SDK, so they authenticate the way Claude Code on the machine
does, in its own precedence: cloud-provider switches, `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_API_KEY`, `apiKeyHelper`, then the claude.ai login. Those can sit in the
shell or in the `env` block of `~/.claude/settings.json`, which is how a company
gateway is usually distributed. Only that `env` block and `apiKeyHelper` are read from
the file. Every run prints which credential it found.

The agent under test is separate. It makes its own provider calls with its own key,
and never inherits anything fleetopt read from its own config files.

## Fixtures

Both run on a fake model, so nothing costs anything on a provider. `fixture/agent.py`
has a planted cost defect: a node that re-sends its whole history. `fixture/supervisor.py`
has planted design smells: a router that only ever takes one branch, a supervisor whose
three workers always run in the same order, a reflection loop that never changes the
draft. The fake model returns fixed strings, so the fixture cannot exercise the judge:
before and after are identical and it passes trivially.

Every break found on a real agent becomes a pytest case; every recurring pattern
becomes a line in a skill. `tests/corpus/corpus_build.py` regenerates the list of
candidate agents from GitHub search.

## Further limits

Beyond those in the README:

- Pricing covers Anthropic, OpenAI and Gemini list rates as of 2026-09-23; hosted
  variants (Bedrock, Vertex, Azure ids) and tiers above 200K context are not priced.
- Other frameworks: ADK emits OpenTelemetry natively (1.17+), so its adapter should be
  an OTLP span processor installed the same way, not a second callback hook.
- No train/test split. Measuring and editing against the same handful of cases
  produces a beautiful number and a production regression.
- Streaming invocation is untested. The driver calls `ainvoke`.
- An agent that writes to files git tracks changes the code fingerprint while it runs.
