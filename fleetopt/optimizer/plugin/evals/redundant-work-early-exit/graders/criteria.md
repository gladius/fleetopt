---
type: llm
weight: 1
---

A successful response meets all of these:

- Identifies 'no early exit': the loop runs to the cap on every trace and the last round adds almost nothing.
- Fix: add a convergence or confidence condition that ends the loop early, and keep MAX_ROUNDS as the safety cap, not the schedule.
- Says to measure the saving and pass the judge before reporting it, and does not simply propose lowering MAX_ROUNDS to a smaller fixed number as the primary fix.
