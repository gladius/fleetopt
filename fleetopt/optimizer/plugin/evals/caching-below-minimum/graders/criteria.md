---
type: llm
weight: 1
---

A successful response meets all of these:

- States that prompt caching cannot help here because the stable prefix (~2,900 tokens) is below the minimum cacheable prefix for Haiku 4.5, and gives that minimum as 4,096 tokens.
- Does not recommend adding cache_control as a saving on this model.
- Treats 'the number says no' as the finding, optionally noting that a model with a lower minimum or a longer stable prefix would change the answer.
