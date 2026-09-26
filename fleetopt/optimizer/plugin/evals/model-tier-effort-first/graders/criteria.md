---
type: llm
weight: 1
---

A successful response meets all of these:

- Recommends lowering effort on the same model first (Anthropic output_config effort low) before switching the node to a cheaper model, and gives the reason: same model keeps the same cache namespace / prompt cache.
- Warns not to change effort within one loop iteration because it invalidates the messages cache, or notes effort is unsupported on Haiku 4.5 / Sonnet 4.5.
- If a tier drop to Haiku is mentioned, frames it as a measured second step gated by the judge, not the first move.
