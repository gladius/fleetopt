# Cost optimization for LangGraph agents - router

These are priors, not a checklist. Most will be irrelevant to any given project and
you should dismiss them in one glance at the traces. Finding something not on this
list is a good outcome; the list exists so findings have names that can be checked
against evidence, not to bound what you look for.

Every finding must cite trace data. "This node looks expensive" is not a finding.
"Node `research` sends 1,339 chars at step 4 versus 57 at step 2, growing
monotonically, 67% of all tokens" is a finding.

## How to work

1. **Read before measuring.** `graph_topology` and the source tell you what the
   agent is for. A saving that breaks the purpose is not a saving.
2. **The target runs only through `measure`.** Never invoke the project's command
   yourself with Bash - it spends the team's API tokens twice, captures nothing, and
   fills your context with their test output. If `measure` fails, read its error and
   fix the run command; do not reproduce the failure by hand.
3. **Query, don't read.** Traces are large. Use `query_traces` with aggregates. Pull
   full prompt text only for the one or two nodes you have singled out.
4. **Measure with n=3 first.** Go to n=5 only when `compare` says within noise and
   the trace evidence still says the effect is real. Each run costs the target's
   own API tokens.
5. **One change at a time.** Apply a single optimization, then measure. Batch two and
   you cannot attribute either the saving or the breakage.
6. **The noise floor decides.** `compare` reports `within noise` when a delta sits
   inside the baseline's own spread. That is not a saving. Do not report it as one.
7. **Equivalence gates everything.** A cost reduction with a failed `judge` is a
   regression you have not noticed yet. Report it as a failure, not a tradeoff.
8. **Evals before the baseline.** If the repo has eval cases (deepeval tests, JSONL
   or JSON with expected answers, an evals/ folder), load them with
   `load_eval_cases` first and prefer the suite that exercises them as the run
   command; `judge` then grades correctness, not just "unchanged". Never run an
   eval framework by hand - it bills the team's own graders. See `fleetopt:evals`.

## Rule out first

Cheap checks that have each cost a real run before. "The number says no" is a
finding worth reporting and costs nothing.

- Caching has a per-provider, per-model **minimum prefix** (512 to 4,096 tokens).
  Under it, nothing you do will cache. Numbers in `fleetopt:caching`.
- Dynamic content in the **user turn is fine**; only the prefix must be stable.
- Tool deferral pays only **above ~10K schema tokens**.
- A **single-shot** agent (one call per run, no loop) has nothing to amortize.
- **Stable literals under ~50 lines** are not a target.

## The patterns - and which skill to invoke

Each decision below is a skill (`fleetopt:<name>`) with the provider-specific mechanics, the
skip-rules and the evidence it needs. Invoke the one the traces point to; never all of them.

| # | Pattern | Trace signature | Skill |
|---|---------|-----------------|------|
| 1 | **Growing re-sent context** | Same node, `prompt_chars` rising across `step` within one trace | `fleetopt:prompt-growth` |
| 2 | **Caching not used** | `cache_read_tokens = 0` while the same prefix repeats across calls | `fleetopt:caching` |
| 3 | **Model over-tiered** | A frontier model on a node whose output is a label, boolean or route | `fleetopt:model-tier` |
| 4 | **No early exit** | Loop runs to `MAX_ROUNDS` on every trace, later rounds adding little | `fleetopt:redundant-work` |
| 5 | **Serial independent calls** | Sibling nodes with disjoint inputs running back to back | `fleetopt:redundant-work` |
| 6 | **Redundant tool calls** | Same tool, same args, more than once per trace | `fleetopt:redundant-work` |
| 7 | **Oversized system prompt** | Large constant `prompt_chars` floor on every call to a node | `fleetopt:prompt-growth` |
| 8 | **Retries hidden as cost** | Repeated near-identical LLM runs, `error` set on the earlier ones | `fleetopt:redundant-work` |
| 9 | **Output not bounded** | `output_tokens` near the cap, or long outputs truncated downstream | `fleetopt:model-tier` |
| 10 | **Dead state carried** | Large fields in root `inputs` that never reach any prompt | `fleetopt:prompt-growth` |
| - | **Tool surface bloated** | Many tools bound, MCP server attached wholesale | `fleetopt:tool-surface` |

The `provider` and `model` columns in `runs` decide which branch of a file applies.
A project may use several providers; check per node.

## Useful queries

```sql
-- Where the tokens actually are
SELECT node, COUNT(*) calls, SUM(input_tokens) tin, SUM(output_tokens) tout
FROM runs WHERE session_id = ? AND run_type = 'llm' GROUP BY node ORDER BY tin DESC;

-- Which model and provider each node uses (decides which reference branch applies)
SELECT DISTINCT node, model, provider FROM runs WHERE session_id = ? AND run_type = 'llm';

-- Pattern 1: is a node's prompt growing within a single invocation?
SELECT node, step, prompt_chars FROM runs
WHERE session_id = ? AND run_type = 'llm' ORDER BY trace_id, step;

-- Pattern 2: any caching at all?
SELECT SUM(cache_read_tokens), SUM(cache_write_tokens) FROM runs WHERE session_id = ?;

-- Pattern 8: retries
SELECT node, COUNT(*) FROM runs WHERE session_id = ? AND error IS NOT NULL GROUP BY node;
```

## Reporting

For each finding: the pattern, the evidence (numbers from the traces), the source
location, the change, and the measured result in dollars and tokens, and the judge
verdict. If you could not measure it, say so rather than estimating.
