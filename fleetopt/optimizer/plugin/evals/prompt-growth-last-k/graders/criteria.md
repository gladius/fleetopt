---
type: llm
weight: 1
---

A successful response meets all of these:

- Names the pattern: growing re-sent context, the node re-sends the whole accumulated history each step.
- Cheapest fix first: send only the last k notes or a rolling summary to the repeated `research` call, while `write_report` keeps the full history.
- Cites the numbers (800 to 4,700 chars, 67%) as evidence and says to measure after one change, not to batch fixes.
