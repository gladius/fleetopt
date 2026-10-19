---
name: langgraph
description: How each cost fix is spelled on LangGraph and LangChain, keyed by the skill it belongs to.
when: langgraph
---

### On LangGraph: how each fix is spelled

Read the installed `langchain_*` source when in doubt; spellings drift between versions.

- **caching, Anthropic**: a `cache_control` on a content block, on the static system text
  or the last stable message of the history:
  ```python
  SystemMessage(content=[{"type": "text", "text": STATIC, "cache_control": {"type": "ephemeral"}}])
  ```
- **caching, OpenAI GPT-5.6+**: `prompt_cache_options` goes in the model's `model_kwargs`.
- **prompt-growth**: `trim_messages(messages, max_tokens=N, strategy="last", token_counter=llm)`
  on the repeated call; keep the full history in state for the node that needs it.
- **redundant-work, fan-out**: `Send` from a branch function, or a list of edges from one node.
  A loop ends early by its branch function returning `END` on the done condition, with the
  round cap kept as the safety net.
- **tool-surface**: `bind_tools` per node, with the tools that node uses; a tool list that
  changes between calls of one node breaks that node's cache.
