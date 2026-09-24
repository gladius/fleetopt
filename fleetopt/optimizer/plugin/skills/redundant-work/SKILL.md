---
name: redundant-work
description: Find redundant work in the graph. Use when the topology has a cycle, when every trace runs to MAX_ROUNDS, when sibling nodes with disjoint inputs run serially, when the same tool is called with the same arguments more than once per trace, or when runs with error set are followed by near-identical retries.
---

# Is the graph doing redundant work?

Open this when the topology shows a cycle, when `step` counts per trace are high or
identical across traces, or when the same tool appears repeatedly. Patterns 4, 5, 6, 8.

## No early exit (pattern 4)

Signature: every trace runs the loop to `MAX_ROUNDS`; later rounds add little
(compare `completion` of round n with round n-1). Fix: exit on a convergence or
confidence condition; keep `MAX_ROUNDS` as the safety cap, not the schedule.
Evidence: rounds per trace from `SELECT trace_id, MAX(step) ... GROUP BY trace_id`,
and what the last round changed.

## Serial independent calls (pattern 5)

Signature: sibling nodes with disjoint inputs whose `start_time`/`end_time` never
overlap. Fix: fan out (LangGraph `Send`, or a list of edges from one node). This saves
wall-clock, not tokens - say so; do not report it as a cost saving unless `cost_usd`
moved.

## Redundant tool calls (pattern 6)

Signature: same tool, same args, more than once per trace (`run_type = 'tool'`,
group by name + inputs). Fix: memoize within the invocation, or carry the result in
state. Each repeat also re-sends its result into the next prompt, so this compounds
with pattern 1.

## Retries hidden as cost (pattern 8)

Signature: near-identical LLM runs where the earlier ones have `error` set, or a node
that invokes twice per step. Fix the cause (schema mismatch, timeout too tight,
parse failure) and cap retries. Evidence: `SELECT node, COUNT(*) FROM runs WHERE
error IS NOT NULL GROUP BY node`.

## Do not flag

- A loop whose later rounds demonstrably change the output - that is the agent
  working, not waste.
- Parallelising nodes that share state the graph mutates - correctness first.

## Evidence required

Numbers from the traces per pattern above, `file:line` of the loop or edge, and the
measured effect after the change (`compare`), including that wall-clock and tokens
are separate claims.
