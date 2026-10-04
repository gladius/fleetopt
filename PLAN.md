# Plan

Agreed with Gladius on 2026-10-04. Kept here so the plan is one page everybody works from.

## What this is

A developer runs one tool on their own machine, in their own setup, and names an expert. The
tool loads that expert, runs it against their LangGraph agent with shared tools and guards,
tracks what it checked, and reports. Experts are written centrally, as prose.

## What holds throughout

- **One session is one expert.** Only that expert's guide and skills are loaded.
- **An expert is a folder of prose.** Its header names what it is asked, which shared tools
  it uses, what a change must earn and what it checks. It can name a tool or a rule; it
  cannot bring its own.
- **Code owns the numbers and the verdict.** The tools, the gate and the probe are shared,
  and are what the central team pins. No expert grades its own work.
- **Experts reach each other through results, not shared context**: a review handed on as
  a file, or a call that runs the other expert in its own session and returns its result.
  No orchestrator agent, no free sub-agents.
- **Never an invented expected answer.** Generated requests are labelled as generated, and
  the original agent's own answers are the reference.

## Steps

| # | Step | Done when | State |
|---|---|---|---|
| 1 | Experts as folders of prose, outside the code (`experts/`), loaded from a named place | Committed, tested | done |
| 2 | Tracking: a matrix of node x check, opened by code, closed by the expert with its number | A real cost run prints the `Checks` line | done: 12 of 12 closed on the workshop agent |
| 3 | Pre-run the three demo agents and keep the outputs | Outputs saved | before the demo |
| 4 | An expert calls another through a tool; the tester is the first called expert | A cost run on an agent with no checks proceeds on generated requests, labelled so | after |
| 5 | Ask later in a run ("this cut changes one answer: keep it?") | Used once with a person at the terminal | after |
| 6 | Fetch an expert from the central catalogue over http, its fingerprint recorded with the run | `--expert name` works on a machine that never saw it | built and tried against a local web server; no real central server yet, and nothing decides when a pulled copy is refreshed except `fleetopt pull` |
| 7 | See model calls made outside LangChain | An agent with its own HTTP client measures | after |
| 8 | Runs report back to the centre | One table of runs across machines | after |

The run that matters most fits anywhere: the production agent, on the current `main`.

## Experts, in the order agreed

cost (built) · design (built, reviews only) · tester (step 4) · design changes, when a team
has expected answers · model migration.

## Later, and separate from each other

A central knowledge base. Experts that improve from what runs record, behind a benchmark.
Neither is started.

## Where it stands (2026-10-04)

Three real agents run. One proven saving: cost -42% on a supervisor-of-supervisors, kept after
a first fix was refused for changing an answer. One honest nothing: a call worth 29% of cost
found, its removal measured, worse, undone. One broken agent: the design review found why it
finishes nothing. On the production agent: not yet run with this version.
