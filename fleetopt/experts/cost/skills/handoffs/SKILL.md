---
name: handoffs
description: Where the money goes between agents - calls that only reword, transcripts re-sent at every layer, rounds that repeat, loops with no cap across layers.
---

# Is money spent between the agents rather than on the work?

A graph of agents (a supervisor and workers, agents used as tools, a hierarchy) pays for
every hand-off twice: the call that decides to hand off, and the call that reads what came
back. None of that shows up as a large prompt or a big model, so the five other checks miss
it. Count calls per request, by agent, and ask what each one is for.

## A call that only rewords (observed: 29% of one agent's cost)

A worker used as a tool reads its tool result and writes prose about it; the supervisor
then reads that prose and writes the answer. The second of those calls adds nothing the
first did not have. Evidence: in the recordings, the worker's last completion and the
supervisor's final answer say the same thing; the worker's last call follows its tool
result with no further tool call. Mechanics: return the tool's raw result from the worker
(`messages[-1].content` of a nested agent is the reworded one; the tool message before it is
the data), or let the worker's answer be the final answer instead of restating it.

## The transcript re-sent at every layer

`output_mode="full_history"` on `create_supervisor`, hand-off-back messages, or a worker
that returns its whole message list: every layer above then carries every layer below.
Evidence: `prompt_chars` of the supervisor's calls grows by the size of the worker's
transcript after each hand-off. Mechanics: `output_mode="last_message"`; return a summary
field, not the messages; keep hand-off-back messages only if something reads them (a UI
that expects them is a reason to keep them: say so).

## Rounds that repeat

A tool called again with the same arguments, a search re-run with the same query, a worker
re-invoked for the same task: the model did not get what it wanted and tried again.
Evidence: identical `inputs` on consecutive tool runs in one request; a worker invoked
more than once with the same request. Mechanics: a tool that says plainly when its result
cannot change; a cap on rounds (`ToolCallLimitMiddleware`, `ModelCallLimitMiddleware`, or
the graph's own counter); a prompt that asks for several tool calls in one turn where the
calls are independent ("one tool at a time" instructions double the rounds).

## Loops with no cap across layers

Each layer may have its own loop and none may have a limit; the recursion limit is then
the only stop, and one stuck request costs as much as fifty good ones. Evidence: a request
whose calls are several times the median's; `step` counts far above the others. Mechanics:
a limit per layer, and an exit when the last reply repeats the one before.

## Do not flag

- A supervisor whose workers really differ and whose dispatch varies with the request:
  the hand-off calls are the design. Removing them is the design expert's question.
- Hand-off-back messages a UI or a test depends on: name the dependency and leave them.

## Evidence required

Calls per request by agent (the `path` column), the share of tokens in the calls you call
redundant, and the two completions that say the same thing, quoted short. A change here
keeps the graph's nodes and edges: it changes what a call returns, what is passed up, or
when a loop stops.
