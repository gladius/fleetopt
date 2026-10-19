---
name: adk
description: How each cost fix is spelled on Google's ADK, keyed by the skill it belongs to.
when: adk
---

### On ADK: how each fix is spelled

Checked against google-adk 2.12.0; read the project's own `google/adk` source when in doubt.

- **caching, any model**: one setting for the whole tree, on the App the team ships (or the
  one to add around `root_agent`): `App(name=..., root_agent=...,
  context_cache_config=ContextCacheConfig(min_tokens=N, ttl_seconds=S, cache_intervals=K))`.
  ADK then places the breakpoints itself: `cache_control` for Claude, a context cache for
  Gemini. Without it, a hand-off between agents swaps the system instruction and the tool
  set, so the prefix changes at every transfer and nothing is read from cache (ADK warns of
  exactly this at start-up).
- **prompt-growth**: every `LlmAgent` is sent the whole session so far by default
  (`include_contents='default'`), so a loop's critic and reviser each re-read every earlier
  round. `include_contents='none'` with the step's own inputs in its instruction (`{draft?}`,
  `{critique?}` from state, written by `output_key`) sends only what that step needs. For
  long sessions, `EventsCompactionConfig` on the App summarises old invocations.
- **redundant-work, loops**: the loop is a `LoopAgent(max_iterations=N)`; it ends early when
  a step escalates: a tool setting `tool_context.actions.escalate = True`, or an
  `after_agent_callback` setting `callback_context.actions.escalate = True` when the
  critique repeats. `RunConfig.max_llm_calls` caps a whole invocation.
- **redundant-work, fan-out**: a `ParallelAgent` over the independent agents.
- **tool-surface**: each `LlmAgent` has its own `tools=[...]`; every sub-agent adds a
  `transfer_to_agent` tool to its parent and, unless `disallow_transfer_to_parent` /
  `disallow_transfer_to_peers`, to itself.
