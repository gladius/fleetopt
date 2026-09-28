# Applying a review to a LangGraph agent - how to work

You are handed a review of this agent: numbered findings, each with the evidence it
rests on. A person has read it. Your job is to try the findings you were given, one
at a time, and prove each. What to look for, and which skill holds the mechanics of
each change, is in the second half of this guide.

Every claim must cite trace data or a measurement. "This node looks expensive" is
not a claim. A saving you have not measured is not a saving.

## How to work

1. **Read before measuring.** The review, then `graph_topology` and the source it
   points to. A saving that breaks what the agent is for is not a saving.
2. **The target runs only through `measure`.** Never invoke the project's command
   yourself with Bash - it spends the team's API tokens twice, captures nothing, and
   fills your context with their test output. If `measure` fails, read its error and
   report it. How the agent is started is fleetopt's to fix, not yours; do not
   reproduce the failure by hand.
3. **Query, don't read.** Traces are large. Use `query_traces` with aggregates. Pull
   full prompt text only for the one or two nodes you have singled out.
4. **Measure with n=3 first.** Go to n=5 only when `compare` says within noise and
   the trace evidence still says the effect is real. Each run costs the target's
   own API tokens.
5. **One finding at a time, one commit each.** Make the change for one finding,
   commit it with the finding's id first in the message, then measure. Batch two and
   you cannot attribute either the saving or the breakage, and the team cannot keep
   one and drop the other.
6. **The noise floor decides.** `compare` reports `within noise` when a delta sits
   inside the baseline's own spread. That is not a saving. Do not report it as one.
7. **Equivalence gates everything.** A cost reduction with a failed `judge` is a
   regression you have not noticed yet. Undo it and report it as a failure, not a
   tradeoff.
8. **Evals before the baseline.** If the repo has eval cases (deepeval tests, JSONL
   or JSON with expected answers, an evals/ folder), load them with
   `load_eval_cases` first; `judge` then grades correctness, not just "unchanged".
   fleetopt takes the agent's inputs from the same cases where it can. Never run an
   eval framework by hand - it bills the team's own graders. See `fleetopt:evals`.
9. **A finding can be wrong.** The review saw one capture; your baseline is three.
   If the baseline contradicts a finding, skip it and give the number that does.

## Reporting

One row per finding you were given: its id, what happened (stands, undone, skipped
and why), the measured result in dollars first, then latency, then tokens, and the
judge verdict. If you could not measure it, say so rather than estimating.
