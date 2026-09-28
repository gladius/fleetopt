# Earlier runs, as they were reported at the time

Moved out of the README on 2026-09-28. These describe runs made with commands and flags that
no longer exist (`fleetopt optimize`, `--run`, `--auto`). Kept as a record, not as instructions.
Runs on real agents since then are in `ledger.md`.

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

Re-run 2026-09-26 with isolated sessions and the run record: 21 turns, $0.32,
`fleetopt:prompt-growth` fired, input tokens -11.4%, equivalence 2/2, and the run
folder written with `report.md`, `run.json`, `patch.diff`, `log.txt`. Cost shows
`unpriced` on the fixture because its model is a fake with no price; on a real
agent that line is dollars.

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

With an `evals/cases.jsonl` dropped into the fixture copy and **no flag**, the
optimizer found it on its own, loaded it before the baseline, and the report
gained a correctness section: 0/2 pass on both sides, with the judge's reason
(the fake model's fixed reply can never match a real expected answer) and the
conclusion that the patch did not make correctness worse. 30 turns, $0.43.

Then from a **fresh clone following the Quick start verbatim** (same day, after the
probe-once and environment-guard changes): run command right first time, no target
output in the terminal, `fleetopt:prompt-growth` loaded before the edit, same
-11.4%, judge PASSED 2/2, **20 turns, $0.22 nominal**. The run before those changes
on the same clone took 49 turns and $0.82, fifteen of them crashed baseline runs.

### Verified against a real repository

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

### What a run costs, and why

Measured on the memory-agent run (38 API calls, Sonnet 5): $1.14 for the optimizer,
of which $0.40 was three full cache re-writes caused by `measure` outlasting the
5-minute cache TTL, $0.25 output, $0.36 cache reads, $0.13 tool results (mostly
tracebacks from hand-running the target). Reading source was under 2%. The fixes
above target the first and last items; expected optimizer cost on the same run is
roughly $0.55-0.65. The target's own calls are separate (memory-agent: $0.13/run).

