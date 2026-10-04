---
name: patterns
description: The design patterns a LangGraph agent is built from, when each is warranted, the evidence that it is not, and the simpler design.
---

### Patterns: warranted when, the evidence that it is not, and what is simpler

| Pattern | Warranted when | Smell (evidence) | Simpler |
|---|---|---|---|
| **Single call / pipeline** | Fixed steps, no decisions between them | - | Already the floor |
| **Hand-built agent loop** (separate model calls to choose an action, to issue the tool call, to judge whether it is done) | Almost never: only when each step needs a different model, or a human between them | `calls_per_tool_round` of 2 or more (it leaves out each request's final answer); a `decide`, `plan` or `reflect` node whose reply is one token or a label | A tool-calling agent: one call a round picks the tool, its arguments and when to stop, and keeps tool calls paired with their results as providers require (redesign) |
| **Tool-calling loop** (the model picks tools until done) | Tool choice and count really vary with the request | Same tool sequence on every request (`fixed_dispatch`); the loop always runs to its cap (`constant_rounds`) | A pipeline of those tools; an exit on a done condition |
| **Router** (a conditional edge on a model's label) | Several branches taken across requests | `branch_never_taken` for one or more targets; one target in n/n runs | Hardwire the branch taken (small; needs cases for the requests that would take the others); or a rule in place of the model call |
| **Supervisor / orchestrator** (a model decides which worker next) | The order or the set of workers varies with the request; the workers differ | `fixed_dispatch` with a model called at every step and the order never varying; one worker only | Edges in that order: the supervisor's calls disappear, a model call and a hop a step (small) |
| **Workers that are alike** | Each has its own tools, prompt or model | The same model, the same tools and near-identical system prompts across the worker nodes (compare `prompt` per node with `query`) | One worker run with a parameter, or one agent |
| **Swarm / handoffs** | Control really passes back and forth depending on content | Handoffs in the same order on every request; one agent does all the work | A supervisor, or a pipeline |
| **Plan and execute** | Plans differ across requests and the steps depend on the plan | Plan text nearly identical across requests; the plan never reaches a later prompt | Fixed steps; drop the planner's call |
| **Review / critic loop** | The critique changes the draft in some rounds | `repeated_identical_reply` on the critic; the draft unchanged round to round; `constant_rounds` | One round with an exit condition, or none (small) |
| **Parallel fan-out** | Two or more independent branches a request | A fan-out of one; the same count every time | A plain edge |
| **Retrieval** | Retrieved passages appear in prompts and change answers | Retrieval on every call with identical queries; retrieved text never in a prompt | Retrieve once a request; cache it |
| **Human approval** | A person really intervenes on some requests | An interrupt on every request, always resumed unchanged | Log it, and remove the interrupt |
| **Long-term memory** | Memories are written and later read on other requests | Writes with no reads; reads returning empty in n/n requests | Drop the store until something reads it |
| **Agents nested in agents** | The inner agent's loop really varies: it is more than one call in a wrapper | The nested graph runs one node once every time (`shape` reports each nested graph on its own) | A node, or a tool of the outer agent |

Also worth a line when you see it: **orchestration code that never runs**, a module that
defines a supervisor, a router or a loop and is imported only by tests. The running graph
and the documented design have drifted apart; a person decides which one is right
(redesign).

### What `shape` reports

One line a finding, for the graph that ran and for each graph nested in it (its nodes are
named `outer:inner`):

- `branch_never_taken`: a branch point with targets it never went to, on these runs;
- `fixed_dispatch`: a branch point that sent the work to the same targets in the same
  order every time, and whether it called a model to decide;
- `constant_rounds`: a node that ran the same number of times, more than once, every time;
- `repeated_identical_reply`: a model whose reply was the same in every round;
- `calls_per_tool_round`: two or more model calls spent per round of tool use;
- a node that raised, with whether the error was caught inside the node (the graph then
  carried on without that step's result), and a node that paused for a human;
- `targets_not_declared`: a branch point the graph does not list targets for, with where
  it went on these runs. What it never took is in its function, not in the numbers.

Anything else in the table needs a number of your own from `query`: put what you counted
in the finding. No number, no finding.
