# fleetopt

Finds cost savings in a LangGraph project, **without editing it to observe it**,
and proves the saving is real before claiming it.

```bash
fleetopt optimize <project>
```

That is the whole user surface. Everything else — capturing, querying, measuring,
comparing, judging — is a tool the optimizer calls itself.

## Quick start

**Needs:** Python 3.11+, git, and a machine where Claude Code already works (a
claude.ai login, or your company's key in `~/.claude/settings.json`). No other
secret. There is no compile step; the editable install below is the whole build.

```bash
git clone <this repo> fleetopt && cd fleetopt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # fleetopt + the fixture's langgraph; ~250 MB (bundled Claude Code binary)
# smoke test on a copy of the bundled fixture: 3-5 min, no API spend
cp -r fixture /tmp/fixture && git -C /tmp/fixture init -q && git -C /tmp/fixture add -A && git -C /tmp/fixture commit -qm base
fleetopt optimize /tmp/fixture --auto
```

The last line should end with an `equivalence: PASSED` block and a `─── done in N
turns, $x ───` line, and leave an `opt/...` branch in `/tmp/fixture`. The copy
matters: the optimizer branches whatever git repo the target is in, and
`fixture/` inside this checkout would mean branching fleetopt itself.

**On a real project** (must be a git repo; the patch lands on a branch, master is
never touched):

```bash
# first run on an unfamiliar repo: give it the command, keep the prompts, cap the spend
FLEETOPT_PARALLEL=1 fleetopt optimize ~/work/their-agent \
    --run "python -m pytest tests/integration -q" --max-usd 3

# once you trust it: no prompts
fleetopt optimize ~/work/their-agent --auto
```

| Flag | Meaning |
|---|---|
| `--run CMD` | How to invoke the agent once, end to end. Optional: the optimizer finds it otherwise, and sometimes picks the wrong interpreter first. Locked when given. |
| `--auto` | Answer yes to both permission prompts (run the target, edit on a branch). |
| `--max-usd N` | Stop the optimizer once its *own* spend reaches N (default 5). The target's API calls are its own bill. |
| `--out DIR` | Where captures go (default `./.fleetopt`, relative to where you run it). |

**What you get:** the report in the terminal (finding, measured before/after,
judge verdict), the patch committed on a branch in the target repo, and every
measurement in `.fleetopt/fleetopt.db`. The run also prints which credential it
is using as its first line.

**Two debug commands**, for when the optimizer comes back empty on a repo:
`fleetopt capture <project> --run CMD` runs the target under instrumentation
without any model involved (free), and `fleetopt report` prints what was
captured. If `capture` shows `0 runs`, the command did not invoke the graph, or
the interpreter has no langchain.

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
  that would run it by hand (`pytest`, `python x.py`, `langgraph dev`, the run command
  itself). Running it by hand spent the team's tokens twice, captured nothing, and fed
  12K tokens of tracebacks into the optimizer's context.
- **An operator-supplied `--run` is locked** until `measure` has actually failed with
  it. The agent once spent 13 turns re-deriving a command that was already correct.
- **Equivalence gates everything.** A cost reduction with a failed judge is a
  regression nobody noticed yet.

## Permissions

Gated by class of side effect, not per tool — an optimizer that asks seven times
is a wizard, not an agent.

| | Gate |
|---|---|
| Read, query, measure, judge | none |
| Run the target's own command | asked once |
| Modify the repository | asked once; git branch is the undo |

`--auto` drops to zero prompts. `--max-usd` (default 5) caps the optimizer's own
spend; the target's API calls are not included. The optimizer cannot read the
target's secrets: `.env*`, key and certificate files, anything named
`*secret*`/`*credential*`, `.netrc` and friends are denied to Read/Grep/Glob
(`config.DENY_READS`; verified live 2026-09-24). It loads the machine's
user-level Claude Code settings and nothing from the target repo's `.claude/`,
so another team's hooks, permissions and CLAUDE.md never run inside it, and it
spawns its subprocesses with telemetry and auto-memory off. A decline is final for the session: the agent is
told to stop and report from read-only evidence, not to look for another route.
Note for anyone touching `session.py`: a tool listed in `allowed_tools` is
auto-approved *before* `can_use_tool` is consulted (the SDK warns with
`CanUseToolShadowedWarning`), so gated tools must be left out of that list.
Verified 2026-09-23 by declining both prompts: no sessions written, no edits, no
fabricated numbers.

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
`fleetopt optimize` prints which one it found. The target agent is separate: it
runs with the operator's real environment and its own keys, and never inherits
anything fleetopt read from its own config files.

Knobs are optional and live in `.env` (per project) or `~/.config/fleetopt/env`
(per machine); real environment wins over both. `FLEETOPT_MODEL` drives the
optimizer (default `claude-sonnet-5`), `FLEETOPT_JUDGE_MODEL` the equivalence
check (default Haiku 4.5), `FLEETOPT_EFFORT` (low|medium|high) lowers the
optimizer's own reasoning spend (unverified for finding quality, off by default),
`FLEETOPT_PARALLEL` sets measurement concurrency.

**What a claude.ai login covers:** the optimizer and the judge, not the agent
under test. The fixture uses a fake model, so its runs are free of API spend.
A real target makes its own provider calls with its own key.

## Verified end to end

Against `fixture/` — a synthetic LangGraph agent with a planted defect: its
`research` node re-sends all prior notes every round.

```
input_tokens   3,029 → 2,029   -33.0%   improved
output_tokens  2,240 → 2,240    +0.0%   within noise
wall_ms          455 →   455    +0.0%   within noise
equivalence    PASSED (2/2)
branch         opt/bounded-research-context
51 turns, $0.81
```

The optimizer found the run command itself, established a baseline, identified
the pattern from `prompt_chars` rising 57 → 698 → 1339 → 1966 within a single
trace, branched, patched, re-measured and judged. It also found the same defect
in `summarize`, which was not planted.

Also verified:

- Coexists with a project's own callback handler (`fixture/with_callbacks.py`):
  their handler saw all 5 LLM calls while we independently recorded 14 runs.
- Chains a project's own `sitecustomize.py` instead of shadowing it.
- The judge discriminates: passes a reworded-but-equivalent output, fails one
  that drops the quantitative findings.
- The code-state guard catches two sessions sharing a label across an edit.

Re-run 2026-09-24 after the auth change, with **no API key anywhere**: the
optimizer and the judge both ran on the machine's claude.ai login (the CLI
printed `auth: claude.ai login (max, ...)`), the judge went through the Agent SDK
and returned PASSED 2/2, and the optimizer loaded `fleetopt:prompt-growth` before
editing. It picked the narrower fix this time (send only the latest note): 3,029 →
2,683 input tokens, -11.4%, outside noise, 32 turns, $0.66 nominal. It recovered
from its own first run command (`python agent.py`, wrong interpreter) and hit the
mixed-code-state guard on a reused label, which cost three extra runs; label
lookups now ignore stale code states, so that cannot recur.

## Verified against a real repository

[`JoshuaC215/agent-service-toolkit`](https://github.com/JoshuaC215/agent-service-toolkit)
— 81 Python files, 15 agents, FastAPI service, **async** invocation,
**langchain-core 1.4.x** (we developed against 1.6.3), real Anthropic calls and a
real web-search tool. Unmodified.

The optimizer found the entrypoint itself, recovered when its first run command
used the wrong interpreter, identified from the traces that an 891-token
system+tools prefix repeated verbatim on every call with `cache_read_tokens = 0`,
and applied prompt caching on a branch.

Then it measured a 7.6% cost reduction and **refused to report it as a saving** —
inside the baseline's own spread, so `compare` returned `within noise`. It
investigated further and found the actual reason: Haiku 4.5 needs a 4,096-token prefix
before Anthropic will write a cache entry (per the docs; the agent guessed ~2,048 at the
time), and this prefix is 891. The
pattern was real; it was not exploitable at this prompt size.

That outcome matters more than a win would have. An optimizer that reports 7.6%
of noise as a saving is worse than no optimizer.

The run also surfaced two defects, both since fixed: `measure` averaged in runs
that had crashed halfway, and the judge paired invocations by exact input match,
which never matches once an agent threads generated ids through its state.

## What a run costs, and why

Measured on the memory-agent run (38 API calls, Sonnet 5): $1.14 for the optimizer,
of which $0.40 was three full cache re-writes caused by `measure` outlasting the
5-minute cache TTL, $0.25 output, $0.36 cache reads, $0.13 tool results (mostly
tracebacks from hand-running the target). Reading source was under 2%. The fixes
above target the first and last items; expected optimizer cost on the same run is
roughly $0.55-0.65. The target's own calls are separate (memory-agent: $0.13/run).

## Known gaps

- **The fixture cannot exercise the judge.** Its fake model returns fixed strings
  regardless of input, so before/after outputs are byte-identical and the gate
  passes trivially. The real repo does exercise it, and it passes correctly there.
- **No optimization has yet been proven to save anything.** The fixture's win was
  real but synthetic; the real repo's candidate landed within noise. A confirmed
  saving on someone else's code is still outstanding.
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
