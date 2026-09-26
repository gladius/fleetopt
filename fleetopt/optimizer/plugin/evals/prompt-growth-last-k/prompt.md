---
max_turns: 8
runs: 1
allowed_tools: [Read, Glob, Grep, Skill]
---

You are fleetopt's cost optimizer reviewing evidence from a LangGraph agent's traces. All facts are below; there are no files to read. Give your decision and the exact change, or say why no change should be made. Be concise.

Node `research` is called four times within each trace (same `trace_id`, steps 1-4). Its `prompt_chars` per step: 800, 1,600, 2,900, 4,700; the pattern repeats in all 5 baseline traces and the node accounts for 67% of all input tokens. The source joins `state["notes"]` in full into every call. A final node `write_report` reads the notes once.
