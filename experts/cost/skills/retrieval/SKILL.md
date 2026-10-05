---
name: retrieval
description: What retrieval puts into prompts - how many passages, how large, fetched how often - and whether the answers use it.
---

# Is retrieval sending more than the answers use?

Whatever an agent retrieves (documents, rows, search results, memories) is input tokens
on the call that reads it, and on every later call that carries the history. Open this
when a node's prompts hold retrieved text.

## What to measure, per request

- **How often it fetches.** Count the retrieval runs (`run_type = 'retriever'`, or the
  tool that searches). The same query fetched in every round of a loop is paid for each
  time, and again in the history.
- **How much it brings back.** The share of a prompt's characters that is retrieved
  text: compare `prompt_chars` on the calls that follow a fetch with those that do not.
- **How much of it the answer uses.** Read three completions beside the passages they
  were given. A model handed ten passages that quotes one is the common case.

## Where the cut is

| What you see | The change | What to check after |
|---|---|---|
| The same query fetched again inside one request | fetch once and keep the result in state for that request | answers unchanged; retrieval runs down |
| More passages than any answer draws on | lower the count fetched, to what the answers use plus a margin | the requests that needed the most passages still answered |
| Whole documents where a part answers | return the matching section, or cut each passage to a length | the answer's source still inside what is returned |
| Fetched passages carried in every later call | keep them for the call that reads them, and a short note of what was found after | later rounds do not ask again for what was dropped |
| Retrieval on requests that never use it | fetch only on the route that needs it | the other routes' answers unchanged |

Each of these changes what the model is shown, so each is proven on answers, not on
tokens alone. Lowering the count is the riskiest: a request that needed the eighth
passage is rare in a small sample. Say how many requests the evidence rests on.

## Do not flag

- Retrieval whose passages appear in the answers and are fetched once: it is the work.
- A count you cannot show to be too high from the answers you read.

## Evidence required

Fetches per request, retrieved characters as a share of the prompt on the node that
reads them, and for the cut you propose, the answers that show the rest was not used.
