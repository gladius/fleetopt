---
rows: spenders
---

For every node that carries more than a twentieth of the tokens, each of these is closed
with a number:

- caching: the largest stable prefix against the provider's minimum, and the cached tokens read.
- handoffs: the calls at this node that only pass on or reword what another call answered.
- model-tier: the model against what this node's output is (a label, a route, prose), and its output tokens against any cap.
- prompt-growth: the prompt's size from round to round, and what in it no later call reads.
- redundant-work: calls or tool runs repeated with the same arguments, rounds that run to a cap, retries.
- tool-surface: the tools bound at this node and the size of their schemas.
