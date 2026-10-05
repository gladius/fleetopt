---
name: needs-a-model
description: Whether a step that calls a model needs one at all, or whether a function, a rule or a lookup does the same work.
---

# Does this step need a model at all?

Ask it of every node that calls a model, before asking whether the pattern around it is
warranted. A model call is the most expensive, slowest and least predictable way to do
anything a line of code can do, and a step that never needed one is a larger finding than
any pattern it sits in.

## Warranted when

The step needs to read or write language whose content varies with the request, or to
judge something no rule written in advance would get right: an answer, a summary, a
classification of free text into categories that overlap, the choice of a tool from what
was asked.

## The evidence that it does not

Count these on the recordings (`query`: `completion`, `prompt` and `inputs` by node), over
requests that differ:

| What the recordings show | What it means |
|---|---|
| The reply is the same whatever is asked | Nothing is decided: a constant |
| The reply is one of a few fixed values, and a test on the input predicts it every time (a keyword, a field, a length, a type) | A rule the code could apply |
| The reply is never read: no later prompt, branch or output carries it | A call with no effect |
| The reply restates its input in another shape (a format, a field picked out of a record, a sum, a date) | A function: a parser, a template, arithmetic |
| The reply is looked up, not composed: the same request always gets the same text | A table, or the data source itself |

One request shows nothing here: "the same whatever is asked" needs requests that differ.
Say how many did.

## The simpler thing

A function, a rule on the input, a lookup, a template, the tool called directly. Where a
model is right for the hard cases and a rule for the common one, the rule first and the
model only when it does not match.

## Before you say so

- **Read the prompt for what it asks.** A classifier that returned one label on these
  requests may return another on a request the sample did not hold. Name the kind of
  request that would need the model, or say that none can exist and why (the reply is
  discarded in source; the branch function ignores it).
- **A stand-in model proves nothing.** A fake or stubbed model returns a fixed reply by
  construction: say that the node's replies cannot be judged here, and judge only what
  the source shows (a reply that is never read is never read, whatever writes it).
- **What the model did for free.** A rule does not tolerate a typo, another language or
  a request it was not written for. Say what the step handles today that a rule would
  not, on its Risk.

Report it under "A simpler design would do". Replacing one call by code inside a node is
`small`; removing the node, or a step several others depend on, is `redesign`.
