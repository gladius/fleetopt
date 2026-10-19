---
rows: nodes
---

For every node of the graph, run or not; `n/a` with why for a node that binds no tool and
reads no input a user did not type:

- agency: the tools bound here that act on the world (write, send, pay, delete, run), and what sits between the model and each act: count of acting tools, count with a gate.
- provenance: the arguments of this node's tools that the model writes and that reach a shell, a query, a path, a host or an amount: count of such arguments, calls recorded carrying model-composed values.
- secrets: key-shaped strings or env names in this node's prompts, completions or tool arguments: calls checked, calls carrying one.
- unchecked-input: text no user typed (retrieval, files, web, tool output, another agent, a store) entering this node's prompt and going on to a tool argument, the answer or a store, with no boundary: sources, prompts carrying them.
- egress: what leaves the process from this node and to which system (HTTP, mail, queue, database, tracing, logs): systems, what is sent.
- supply-chain: MCP servers, skills, plugins or packages this node depends on that the team did not write, and where each comes from: count, count pinned or from a known source.
