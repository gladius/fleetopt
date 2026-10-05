---
name: caching
description: Decide whether prompt caching is exploitable for a node and how to apply it per provider. Use when cache_read_tokens is 0 while a node's prefix repeats across calls, when a system prompt contains per-request content such as timestamps or retrieved memories, or before proposing any cache_control or prompt-cache change. Contains the per-model minimum prefix sizes for Anthropic, OpenAI and Gemini and the mechanics for each.
---

# Is caching exploitable here?

Open this when `cache_read_tokens = 0` while a node's prefix repeats across calls
(pattern 2), or before proposing to cache anything. Decide with two facts from the
traces: the **provider/model** of the node, and the **largest stable prefix** it sends.

## Step 1 - the minimum, per provider (docs checked 2026-09-23; they move)

Below the minimum nothing caches - no error, just zero cache tokens.

| Provider / model | Minimum prefix | Code change | Read price | Write price |
|---|---|---|---|---|
| Anthropic Opus 5, Fable | 512 tok | `cache_control` needed | 0.1x | 1.25x |
| Anthropic Opus 4.8, Sonnet 5 / 4.6 / 4.5, Opus 4.1 | 1,024 | `cache_control` needed | 0.1x | 1.25x |
| Anthropic Opus 4.7 | 2,048 | `cache_control` needed | 0.1x | 1.25x |
| Anthropic **Opus 4.6 / 4.5, Haiku 4.5** | **4,096** | `cache_control` needed | 0.1x | 1.25x |
| OpenAI GPT-5.6 and later | 1,024 | `prompt_cache_options.mode` = implicit/explicit | 0.1x | 1.25x |
| OpenAI GPT-5.5 and earlier 5.x | ~1,024; 5.5 caches at 2,048-token boundaries | none (automatic) | 0.1x | none |
| OpenAI GPT-4.1, o3, o4-mini | 1,024 | none (automatic) | 0.25x | none |
| Gemini 2.5 Flash / Pro | 2,048 | none (implicit) | 0.1x | none |
| Gemini 3.x Flash, 3.1 Pro | 4,096 | none (implicit) | 0.1x | none |

Not monotonic across generations - a 3K prompt caches on Sonnet 4.5 and silently
won't on Haiku 4.5 or Gemini 3.x. If the largest stable prefix is under the minimum,
nothing caches as it stands: go to "Under the minimum" before you clear it.

## Under the minimum - can the stable part be made to reach it?

A prefix just under the minimum is billed in full on every call, while one just over it
is billed at the read price. So a larger prompt can cost less. With P the stable prefix
now, N its size once it reaches the minimum, and r the read price from the table:

- every cached call costs `N x r` for that part, where it cost `P` before;
- it pays when `P > N x r`: with a 1,024 minimum, a prefix over about 100 tokens at a
  0.1x read price, or over about 260 at 0.25x. Observed: a 725-token prefix on a model
  with a 1,024 minimum and a 0.25x read price, reported as "nothing to do";
- where writes carry a premium (the table's last column), the first call costs
  `N x 1.25`: count the calls. With k calls sharing the prefix inside the cache's
  lifetime it pays when `N x 1.25 + (k - 1) x N x r < k x P`;
- a node called once a request, minutes apart, never reads its own cache: nothing to gain.

What may fill the gap, in this order, because it is content the model is sent anyway or
should be:

1. **Stable content that sits after volatile content.** Move it up: instructions,
   tool guidance, output format, examples that are the same on every call but come
   after a date, a name or a retrieved passage. Nothing is added; the stable run just
   gets longer.
2. **Stable content sent in a later message.** A fixed preamble built into the first
   user turn belongs in the system prompt.
3. **Tool definitions.** They render before the system prompt and count toward the
   prefix: a stable, sorted tool list is part of it.
4. **Content the agent fetches on every request and that never changes** (a policy, a
   schema, a glossary): placed in the prefix once, it replaces a fetch as well.

Do not pad with text the agent has no use for. It changes what the model reads, and a
saving that depends on filler is the first thing a team will remove.

The proof is in the next measurement and nowhere else: `cache_read_tokens` above zero
on that node, cost down, and the answers as before. If cached tokens stay at zero, the
prefix is still not stable or still too short: undo it.

## Step 2 - what voids a hit (all providers)

Render order is tools -> system -> messages. Any byte change invalidates everything
after it. In source, look for:

- per-request interpolation into the system prompt: `datetime.now()`, `uuid`,
  `random`, retrieval results, user-specific memories with scores;
- tool lists built from `set()` or unsorted `glob`, or registered by import order;
- conditional `if`/`append` assembly of the system prompt;
- `model`, `effort`, `thinking`, `max_tokens` changing between turns of one loop -
  caches are per model, and most providers invalidate the messages cache on an
  effort change.

In traces: a large constant `prompt_chars` floor on a node with `cache_read = 0`.

Fix order: first make the prefix **stable** (move the volatile parts to the user turn
or the end of the messages - that is where they belong). On OpenAI and Gemini that is
usually the whole fix; caching then happens on its own. On Anthropic add the marker.

## Step 3 - mechanics

**Anthropic via langchain-anthropic.** A breakpoint is a `cache_control` on a content
block:
```python
SystemMessage(content=[{"type": "text", "text": STATIC, "cache_control": {"type": "ephemeral"}}])
```
or on the last stable message of the history. Up to 4 breakpoints. Read the installed
`langchain_anthropic` source if in doubt - the spelling drifts between versions.
Verify with `cache_read_tokens` in the next measurement; the first call writes
(1.25x), the rest read (0.1x), so a loop must make >=2 calls to break even.

**OpenAI.** GPT-5.5 and earlier: nothing to add; stability is the fix. GPT-5.6+:
set `prompt_cache_options={"mode": "implicit"}` (or explicit with breakpoints) on the
request; via langchain-openai pass it in `model_kwargs`. Cached tokens appear as
`input_token_details.cache_read` in the trace.

**Gemini.** Implicit: nothing to add; put the large common content first and send
similar-prefix requests close together (cache lifetime is minutes). Explicit context
caching is a separate object with hourly storage cost - only for a prefix reused
across many requests over a long window. Cached tokens appear as
`input_token_details.cache_read` (from `cached_content_token_count`).

## Do not flag

- Dynamic content in the user turn.
- Prefixes far under the minimum, where no stable content exists to bring them to it.
- Single-shot scripts, or loops that make one call per run - nothing to amortize.
- A node that already shows `cache_read_tokens > 0` at a high share.

## Evidence required

Node, provider, model, largest stable prefix (tokens, from `prompt_chars / 4` or the
provider's count), breaker type and `file:line`, and after the change the
`cache_read_tokens` share plus `compare` in dollars.
