---
name: patterns
description: Architecture review of a LangGraph agent - name the design patterns it uses, check each against the numbers from graph_shape, and say whether a simpler design would do the same job. Use when the run has --review, or when asked whether an agent's design fits its use case. Findings are recommendations with evidence, never patches; tier-one changes may become patches only when eval cases are loaded.
---

# Does the design fit the job?

Optimization keeps the design and makes it cheaper. This review asks a different
question: is this the right design at all. The bar is the same. Every finding is a
number from the traces plus a line of source, or it is not a finding. A pattern that
fits is a finding too - "supervisor justified: 4 distinct worker orders in 12 traces"
saves the team a debate.

## How to work

1. **State the job first.** From the README, the system prompts and the eval cases: what
   is this agent for, what does a good answer look like? Fitness is relative to that.
2. **Get the numbers.** `graph_topology` for the declared shape; `graph_shape` on the
   baseline label for what actually happened; `query_traces` for anything specific.
   Read source only to confirm what a number says.
3. **Name the patterns you see** with the table below, then check each one's "warranted
   when" against the evidence. Say "on the inputs we ran" - a branch never taken in 12
   traces may be over-built, or may never have been asked.
4. **Report the effect on cost, latency and reliability**, separately, from the trace
   numbers (calls saved, hops removed, retries avoided). Simplicity is a note, never a
   headline. Correctness is the gate, not an axis.
5. **Never patch.** The optimizer owns the branch. Mark each recommendation tier one or
   tier two so it knows what to do with it.

## Tiers

- **Tier one - mechanical, measurable.** Hardwire a branch that is always taken, drop a
  reflection round that never changes the output, add an early exit, merge two
  sequential model calls into one prompt, flatten a fixed-order supervisor into edges,
  demote a node's model. The optimizer may try these as patches - only when eval cases
  are loaded and cover the affected path, because "unchanged on 3 runs" is weak evidence
  for a structural change and removing a path removes capability for inputs the sample
  never contained. Without cases: report, and say which cases would unlock it.
- **Tier two - redesign.** "These five agents should be one", "this should not be an
  agent". Report with evidence. A human decides.

## Patterns: warranted when, smell when, simpler is

| Pattern | Warranted when | Smell (evidence) | Simpler |
|---|---|---|---|
| **Single call / pipeline** | Fixed steps, no decisions between them | - | Already the floor |
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

`graph_shape` kinds referenced above: `branch_never_taken`, `fixed_dispatch` (with
`calls_model`), `constant_rounds`, `repeated_identical_reply`. Anything else in the
table needs a `query_traces` number of your own; write the query into the finding.

## Report format

For each pattern found:

```
<pattern> on <node(s)> - fits | over-built | under-built
evidence: <the number, e.g. "route -> technical in 12/12 traces; billing, other never taken">
source: <file:line>
change: <what> (tier one|two; needs eval cases covering <path> | no cases needed)
effect: cost <calls or tokens saved per trace>, latency <hops removed>, reliability <retries/errors>
risk: <what an input outside the sample could do>
```

Close with one line: how many traces and inputs the review rests on, and whether eval
cases were loaded. No number, no finding.
