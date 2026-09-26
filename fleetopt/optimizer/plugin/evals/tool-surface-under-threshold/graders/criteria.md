---
type: llm
weight: 1
---

A successful response meets all of these:

- Estimates the schema cost as roughly chars/4, about 4,500 tokens, and states the method.
- Concludes this is under the ~10K schema-token threshold where deferred loading or tool search pays off, so the answer is to stop: deferral would add a discovery step costing more than it saves.
- Does not recommend deferred loading; may mention binding fewer tools per node only as a minor, separate point.
