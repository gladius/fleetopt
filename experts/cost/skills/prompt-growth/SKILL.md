---
name: prompt-growth
description: Diagnose prompts that grow or carry dead weight. Use when prompt_chars rises across steps for the same node within one trace, when a node has a large constant prompt floor on every call, when external input (scrapes, files, rows) is concatenated with no length gate, or when root inputs carry fields that never reach any prompt.
---

# Is the prompt growing, or carrying dead weight?

Open this when `prompt_chars` rises across `step` for the same node inside one trace,
when one node's `prompt_chars` floor is large on every call, or when root `inputs`
carry big fields.

## Growing re-sent context

Signature: same node, same `trace_id`, `prompt_chars` climbing monotonically with
`step`. The node joins the whole history (`"\n".join(state["notes"])`,
`state["messages"]` un-trimmed) into every call.

Fixes, cheapest first:
- Send the last k items, or a rolling summary, to the *repeated* call. Keep the full
  history in state for the node that genuinely needs it (usually the final one).
- LangChain: `trim_messages(messages, max_tokens=N, strategy="last", token_counter=llm)`.
- Prune tool results at phase boundaries instead of carrying every verbatim result.

This changes what the model sees. `keep` decides whether it changed what the model
*says*: a separate reader compares the answers. Trimming that drops a fact the agent
needed is refused there - report that as a change that failed, not a tradeoff.

## Oversized constant prompt

Signature: large `prompt_chars` floor on every call to a node, cache_read = 0.
Look for reference docs, schemas, style guides or long few-shot blocks inlined
(`open(...).read()` into a prompt, multi-hundred-line literals), and the system prompt
describing tools that are already in the `tools` array (pure duplication - delete).

Fix order: if the prefix is stable and above the provider's cache minimum, cache it
(caching) before trimming - a cached doc is cheap and deferring it costs discovery
turns. Trim only what no call uses, and check which side of the minimum the trimmed
prefix is left on: one cut to just under it stops caching and costs more than before.
A prefix not far under the minimum is the opposite case (caching: "Under the minimum").

## Unbounded external input

Signature: prompt size varies wildly between traces of the same node. Web scrapes,
file contents, DB rows, API responses concatenated with no length gate, truncation or
token count. Fix: gate at the source (`[:N]`, `max_chars`, a summarize step), and log
what was cut.

## Dead state

Signature: fields present in root `inputs` (and threaded through every node's state)
that never appear in any `prompt`. Query: compare root `inputs` keys against the
prompts. Fix: stop threading them; state serialisation is not free with a checkpointer.

## Do not flag

- A doc that most calls consult end-to-end and that sits in a cached prefix.
- Trimming literals under ~50 lines: nothing worth a change.
- Dynamic content in the *user* turn - that is where it belongs (only the prefix must
  be stable).

## Evidence required

`file:line`, the node, the numbers (`prompt_chars` by step, or floor size), how often
it is sent (every call / per node / once), and the fix.
