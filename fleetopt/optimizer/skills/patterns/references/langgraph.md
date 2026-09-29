# LangGraph and LangChain facts for a reviewer

Checked 2026-09-28 against installed packages: langchain 1.4.2, langgraph 1.2.12.
Each line was read from package source or observed in a run, not recalled. When the
target's installed version differs, trust its `site-packages` over this file.

- The standard tool-calling agent is `langchain.agents.create_agent(model, tools,
  system_prompt=..., middleware=[...])`. One model call per round chooses the tool, writes
  its arguments and decides when to stop.
- `langgraph.prebuilt.create_react_agent` is marked deprecated since LangGraph 1.0: "has
  been moved to `langchain.agents` ... `from langchain.agents import create_agent`".
  Recommend `create_agent` when the target has langchain 1.x.
- In `create_agent`, an exception raised inside a tool ends the run. Only argument
  validation errors go back to the model by default. `ToolErrorMiddleware(on_error)`
  returns a failure to the model as an error message; `on_error(exc, request)` returns the
  text, or `None` to let it propagate. Observed: one page fetch returning an HTTP error
  killed a whole request until this was added.
- Limits are middleware: `ModelCallLimitMiddleware(run_limit=N)`,
  `ToolCallLimitMiddleware`. Context: `SummarizationMiddleware`,
  `ContextEditingMiddleware`. Human approval: `HumanInTheLoopMiddleware`.
  Retries: `ToolRetryMiddleware`, `ModelRetryMiddleware`.
- A state field that holds messages needs a reducer: `Annotated[list, add_messages]`. A
  plain `list` is replaced by each node's return, which drops the history.
- Providers require every tool result to follow the assistant message that holds its
  tool call. Anthropic rejects an orphan with a 400, "unexpected `tool_use_id` found in
  `tool_result` blocks". Code that slices or rebuilds the message history by hand is
  where this breaks.
- `langgraph-supervisor` was archived on 2026-07-15. Its README recommends building the
  supervisor pattern directly with tools: sub-agents exposed as tools of one agent.
