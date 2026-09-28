---
name: patterns
description: Architecture review of a LangGraph agent - name the design patterns it uses, check each against the numbers from graph_shape, and say whether a simpler design would do the same job. Use when reviewing an agent, when asked whether its design fits its use case, or before applying a design finding from a review. Findings are recommendations with evidence; a tier-one change may be applied only when eval cases are loaded, a redesign only when a person names it.
---

# Does the design fit the job?

Optimization keeps the design and makes it cheaper. This review asks a different
question: is this the right design at all. The bar is the same. Every finding is a
number from the traces plus a line of source, or it is not a finding. A pattern that
fits is worth a line too - "supervisor justified: 4 distinct worker orders in 12 traces"
saves the team a debate.

## How to work

1. **State the job first.** From the README, the system prompts and the eval cases: what
   is this agent for, what does a good answer look like? Fitness is relative to that.
2. **Get the numbers.** `graph_topology` for the declared shape; `graph_shape` on the
   baseline label for what actually happened; `query_traces` for anything specific.
   Read source only to confirm what a number says.
3. **Name the patterns by what ran, not by what nodes are called.** A node named
   `supervisor` that never calls a model is a rule, not a supervisor; check the `model`
   column before naming anything. Then use the table below, then check each one's "warranted
   when" against the evidence. Say "on the inputs we ran" - a branch never taken in 12
   traces may be over-built, or may never have been asked. Evidence is as wide as the
   distinct inputs, not the trace count: `graph_shape` reports both, and 6 traces of 2
   inputs is 2 inputs.
4. **Report the effect on cost, latency and reliability**, separately, from the trace
   numbers (calls saved, hops removed, retries avoided). Simplicity is a note, never a
   headline. Correctness is the gate, not an axis.
5. **Step back once.** After the per-pattern findings, ask the question the team did not:
   would a standard construct - one model call, a fixed pipeline, a tool-calling agent -
   do this whole job? Answer in one finding, tier two, with the number that supports it
   (`calls_per_tool_round`, calls per trace, nodes that never vary), or say in one line
   why the structure earns its keep. Defects are not a reason to skip this: a design can
   be both broken and over-built, and fixing the defects inside a design that should not
   exist is wasted work.
6. **Never patch.** `fleetopt apply` owns the branch. Mark each recommendation tier one or
   tier two so it knows what to do with it.

## Tiers

- **Not design at all.** A change that keeps the graph's nodes and edges - an early
  exit from a loop that already exists, a smaller model on a node, a trimmed prompt -
  is a cost finding. It is tried on the evidence of the run and gated by the judge.
- **Tier one - mechanical, measurable.** Hardwire a branch that is always taken, drop a
  reflection round that never changes the output, merge two sequential model calls
  into one prompt, flatten a fixed-order supervisor into edges. These remove or rewire
  nodes or edges. `fleetopt apply` may try them - only when eval cases
  are loaded and cover the affected path, because "unchanged on 3 runs" is weak evidence
  for a structural change and removing a path removes capability for inputs the sample
  never contained. Without cases: report, and say which cases would unlock it.
- **Tier two - redesign.** "These five agents should be one", "this should not be an
  agent". Report with evidence. A human decides: it is attempted only when a person
  names that finding, and only with eval cases loaded, because being equivalent to a
  design that was wrong proves nothing.

## Before recommending a redesign: what must survive

A simpler design that loses something the old one did is a regression with better
numbers. List these from the code, and for each say how the simpler design keeps it:

- the entry point other code calls, and what it returns or streams;
- what happens when a tool fails - the old graph may have tolerated it;
- the cap on rounds or spend;
- anything written to disk, sent or stored while it runs;
- state fields that code outside the graph reads;
- the points where a human approves;
- provider rules the old prompt handling relied on, or broke.

## Patterns: warranted when, smell when, simpler is

| Pattern | Warranted when | Smell (evidence) | Simpler |
|---|---|---|---|
| **Single call / pipeline** | Fixed steps, no decisions between them | - | Already the floor |
| **Hand-built agent loop** (separate model calls to choose an action, to issue the tool call, to judge whether it is done) | Almost never: only when each step needs a different model or a human between them | `calls_per_tool_round` of 2 or more (it leaves out each trace's final answer); a `decide`/`plan`/`reflect` node whose reply is one token or a label | A tool-calling agent (`create_agent`): one call per round picks the tool, its arguments and when to stop, and keeps tool calls paired with their results as providers require (tier two) |
| **ReAct loop** (model picks tools until done) | Tool choice and count really vary per input | Same tool sequence in every trace (`fixed_dispatch`); loop always runs to cap (`constant_rounds`) | A pipeline of those tools; early exit on a done condition |
| **Router** (conditional edge on a model's label) | Several branches taken across inputs | `branch_never_taken` for one or more targets; one target in n/n traces | Hardwire the taken branch (tier one, needs cases for the untaken inputs); or a rule instead of a model call |
| **Supervisor / orchestrator** (a model decides which worker next) | Worker order or set varies with the input; workers are heterogeneous | `fixed_dispatch` with `calls_model`: a model is consulted at every step and the order never varies; one worker only | Edges in that order; the supervisor's calls disappear (cost and a hop of latency per step) |
| **Swarm / handoffs** | Control genuinely passes back and forth depending on content | Handoffs in the same order every trace; one agent does all the work | Supervisor or pipeline |
| **Plan-and-execute** | Plans differ across inputs and steps depend on the plan | Plan text nearly identical across traces; plan never consulted downstream (state field never reaches a prompt) | Fixed steps; drop the planner call |
| **Reflection / critic loop** | The critique changes the draft in some rounds | `repeated_identical_reply` on the critic; draft unchanged round to round; `constant_rounds` | One critique round with an exit condition, or none (tier one) |
| **Parallel fan-out** | 2+ independent branches per trace | Fan-out of one; branches always run in the same count | A plain edge |
| **RAG** | Retrieved passages appear in prompts and change answers | Retrieval on every call with identical queries; retrieved text never in the prompt | Cache the retrieval; retrieve once per trace |
| **Human-in-the-loop** | A human actually intervenes on some traces | Interrupt on every trace, always resumed unchanged | Remove the interrupt, log instead |
| **Long-term memory store** | Memories are written and later read on other traces | Writes with no reads; reads returning empty in n/n traces | Drop the store until something reads it |

Also worth a line when you see it: **orchestration code that never runs** - a module that
defines a supervisor, a router or a loop and is imported only by tests. The running graph
and the documented design have drifted apart; a human decides which one is right (tier two).

`graph_shape` kinds referenced above: `branch_never_taken`, `fixed_dispatch` (with
`calls_model`), `constant_rounds`, `repeated_identical_reply`, `calls_per_tool_round`, `interrupt` (a node paused
for a human), and `node_error`. A `node_error` comes before any pattern: a node that
raises in every trace is broken, and one marked as caught inside the node means the
graph carried on without that step's result - report it first, as a reliability finding. Anything else in the
table needs a `query_traces` number of your own; write the query into the finding.

## What a design finding says

One numbered block per pattern that is over-built, under-built or broken:

```
### D1 - <short title>
pattern: <pattern> on <node(s)> - over-built | under-built | broken
evidence: <the number, e.g. "route -> technical in 12/12 traces; billing, other never taken">
source: <file:line>
change: <what>; for tier one, the path the team's eval cases must cover
effect: cost <calls or tokens saved per trace>, latency <hops removed>, reliability <retries/errors>
risk: <what an input outside the sample could do>
tier: one | two
```

A pattern that fits gets one line with its number under "Checked and fine". No
number, no finding.
