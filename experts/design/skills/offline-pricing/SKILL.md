---
name: offline-pricing
description: Work nobody is waiting for, paid at the price of an instant answer - batch interfaces and slower service tiers. A decision about how the agent is used.
---

# Is this work paid for at interactive prices when nobody is waiting?

Providers sell the same model cheaper when the answer may come later: batch interfaces
at about half price with results within a day, and slower service tiers on some models.
Whether that fits is a fact about how the agent is used, not about its code.

## When it applies

- The agent runs on a schedule, over a queue or a file of records, or as a step of a
  pipeline whose result is read later.
- Its evals and regression runs: nobody reads those answers as they arrive.
- Never: a person or another service waits on the reply, or a tool call inside a loop
  needs its answer to take the next step.

Find out from how it is started (a scheduler, a queue consumer, a notebook, a script
over a dataset, an endpoint) and what consumes the result.

## Why it is a design finding

- A batch interface returns results hours later and changes how the caller is written.
- Only the team knows whether an hour's wait is acceptable.
- fleetopt prices every call at the model's list rate, so a cheaper tier would show no
  saving in any measurement: nothing here can prove it.

Report it under "A simpler design would do", size `redesign`, with what you found: how
the agent is started, which calls nobody waits on, their share of the tokens (`query`,
by node), and the provider's discount for that way of calling. Name the interface for
the provider in use, and say it is a list price you read, not a measured saving.
