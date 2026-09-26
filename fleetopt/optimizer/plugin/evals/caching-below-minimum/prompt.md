---
max_turns: 8
runs: 1
allowed_tools: [Read, Glob, Grep, Skill]
---

You are fleetopt's cost optimizer reviewing evidence from a LangGraph agent's traces. All facts are below; there are no files to read. Give your decision and the exact change, or say why no change should be made. Be concise.

Node `answer` runs on Anthropic `claude-haiku-4-5-20251001`. 40 LLM calls in the baseline. Its system prompt is constant across all calls and measures ~2,900 tokens; the user turn differs each call. `cache_read_tokens` and `cache_write_tokens` are 0 on every call. The team asks: should we add `cache_control` to the system prompt?
