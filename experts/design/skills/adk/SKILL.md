---
name: adk
description: Dated facts about Google's ADK (Agent Development Kit) that a design recommendation must get right.
when: adk
---

### ADK facts for a recommendation

Checked 2026-10-19 against google-adk 2.12.0, read from package source and observed in a run.
When the project's installed version differs, trust its `site-packages` over this list, and
say which version you read.

- An agent is an `LlmAgent(name, model, instruction, tools=[...], sub_agents=[...], output_key=...)`.
  Workflow agents hold others and call no model themselves: `SequentialAgent`, `ParallelAgent`,
  `LoopAgent(max_iterations=N)`. `adk run` starts a module's `app` (an `App`) if it has one,
  else its `root_agent`.
- An `LlmAgent` with `sub_agents` hands a request over with the `transfer_to_agent` tool, which
  ADK adds for it; after the hand-off the sub-agent answers and the parent does not resume in
  that turn. `disallow_transfer_to_parent` and `disallow_transfer_to_peers` remove the way back.
- A `LoopAgent` runs its children in order, round after round, until `max_iterations` or until
  a child escalates (`actions.escalate = True` from a tool's `tool_context` or an agent
  callback's `callback_context`). A loop with no escalation runs to its cap every time.
- Each `LlmAgent` is sent the whole session's events by default (`include_contents='default'`);
  `'none'` sends only its instruction and what that instruction pulls from state with
  `{key}` / `{key?}`. `output_key` writes an agent's answer into state.
- The model is per agent: a name (`"gemini-2.5-flash"`) or a `BaseLlm` instance. Claude runs
  through `google.adk.models.anthropic_llm.Claude`, other vendors through LiteLLM.
  `generate_content_config=types.GenerateContentConfig(max_output_tokens=...)` bounds the answer.
- Caching is one setting for the tree, `context_cache_config=ContextCacheConfig(...)` on the
  `App`; compaction of old turns is `events_compaction_config=EventsCompactionConfig(...)` on it;
  `RunConfig.max_llm_calls` caps an invocation. Plugins (`BasePlugin`) on the App or Runner see
  every agent, model and tool call of the tree without touching the agents; fleetopt's probe is one.
- Callbacks per agent: `before_agent_callback`, `after_agent_callback`, `before_model_callback`,
  `after_model_callback`, `before_tool_callback`, `after_tool_callback`. ADK makes no
  after-agent call to an agent that handed the request on.
