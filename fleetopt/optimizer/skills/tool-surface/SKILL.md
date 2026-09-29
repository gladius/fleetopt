---
name: tool-surface
description: Decide whether the tool surface is bloated. Use when a node binds many tools, an MCP server is attached wholesale, or the same tools are bound to every node regardless of use. Contains the ~10K schema-token threshold below which deferral does not pay.
---

# Is the tool surface bloated?

Open this when the source binds many tools, an MCP server wholesale, or the same
tools to every node. Tool schemas render into every request, first in the prefix.

## Decide with one number

Estimate schema tokens as chars/4 of the rendered JSON schema: name + description +
every parameter description + enums, for all tools bound on the call. Report the
estimate and the method.

**Under ~10K schema tokens: stop.** Deferred loading / tool search adds a discovery
step that costs more than it saves. This is the most common correct answer.

## Above the threshold

- **Rarely-used tools loaded always** - admin, debug, fallback tools on every call.
  Anthropic: `defer_loading: true` plus a tool-search tool. OpenAI / Gemini: no
  built-in deferral - select tools per node in code.
- **MCP servers bound wholesale** - filter to the tools the node uses.
- **Same tools bound to every node** - in LangGraph, `bind_tools` per node; a node
  that cannot call a tool should not pay for its schema.
- **Verbose descriptions** - paragraph-long field docs, long enums, nested objects,
  duplicated boilerplate across tools.

## Do not flag

- Descriptions doing real disambiguation between similar tools - that text earns
  its tokens.
- Tool surfaces under the threshold, however untidy.

## Evidence required

Tool count, estimated schema tokens, top offenders, `file:line` of the binding,
and - after the change - `compare` on input tokens for the affected node.
