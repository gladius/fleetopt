---
max_turns: 8
runs: 1
allowed_tools: [Read, Glob, Grep, Skill]
---

You are fleetopt's cost optimizer reviewing evidence from a LangGraph agent's traces. All facts are below; there are no files to read. Give your decision and the exact change, or say why no change should be made. Be concise.

Node `route` runs on Anthropic `claude-sonnet-5`. Across 30 calls its completion is always one of three labels (`billing`, `technical`, `other`), 1-2 output tokens, ~1,800 input tokens each. The rest of the graph also runs on Sonnet 5 inside the same loop. What is the first change to try, and what must you avoid?
