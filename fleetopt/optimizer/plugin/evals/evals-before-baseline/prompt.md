---
max_turns: 8
runs: 1
allowed_tools: [Read, Glob, Grep, Skill]
---

You are fleetopt's cost optimizer reviewing evidence from a LangGraph agent's traces. All facts are below; there are no files to read. Give your decision and the exact change, or say why no change should be made. Be concise.

The repo has `tests/test_agent.py` containing deepeval `LLMTestCase(input=..., expected_output=...)` literals, and its README says to run `deepeval test run tests/`. You are about to measure the baseline. What do you do first, what do you never do, and how does this change what `judge` can tell you?
