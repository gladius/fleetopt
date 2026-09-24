---
name: model-tier
description: Decide whether a node's model tier, effort setting or output length is oversized. Use when a frontier model sits on a node whose output is a label, boolean or route, when output_tokens sit near max_tokens, or when a model swap mid-loop appears in the traces. Covers effort-before-tier for Anthropic, OpenAI reasoning models and Gemini thinking settings.
---

# Is the model or the output over-sized for this node?

Open this when `SELECT DISTINCT node, model, provider` shows a frontier model on a
node whose completions are short, structured or low-entropy, or when `output_tokens`
sits near a cap. Patterns 3 and 9. Order matters: try the cheaper lever first.

## Effort before tier (Anthropic, OpenAI reasoning models, Gemini thinking)

Dropping a tier changes the model; lowering effort keeps it. Same model means the
same cache namespace, and a stronger model at low effort often matches a weaker
model at high effort on routine work.

- Anthropic 4.6+ / Sonnet 5 / Opus 5: `output_config={"effort": "low"|"medium"|"high"}`.
  Not on Sonnet 4.5 / Haiku 4.5 (errors). Do not change effort *within* one loop -
  it invalidates the messages cache on most models.
- OpenAI reasoning models: `reasoning_effort` (`low`/`medium`/`high`).
- Gemini 2.5+: `thinking_budget` / thinking level; `0` disables on Flash.

## Model over-tiered (pattern 3)

Signature: a top-tier model on a node whose output is a label, boolean, route, or
a short extraction (`LENGTH(completion)` small and stable). Fix: a smaller model for
*that node only* (`ChatX(model=...)` per node, not globally). Must pass `judge`; a
router that starts routing differently is a regression, not a saving.

Do not flag: nodes whose completions are long, varied, or where the project's tests
or README say the model choice is deliberate.

## Output not bounded (pattern 9)

Signature: `output_tokens` at or near the cap on many calls, or long completions that
a downstream node truncates or parses only the head of. Fix: set `max_tokens` to what
the consumer reads; ask for the structured shape you need (`with_structured_output`)
instead of prose plus parsing.

## Provider notes

- Caches are per model. Swapping models mid-conversation forfeits the cache.
- Batch / flex endpoints (OpenAI Batch, Gemini Batch, Anthropic Batches) halve the
  price for anything not latency-sensitive - relevant for offline nodes only.

## Evidence required

Node, model, provider, `output_tokens` distribution, completion shape, `file:line`
of the model binding, and the `judge` result for the change.
