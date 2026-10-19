---
name: agency
description: Which tools act on the world, what the model controls in them, and what counts as a gate between model and act.
---

### Agency: what the agent can do

Classify every tool by its worst act, from the code of the tool, not its docstring:

| Class | Examples | Count as acting |
|---|---|---|
| destroys | delete, drop, rm, overwrite, cancel, revoke | yes |
| pays | charge, refund, transfer, order, provision | yes |
| sends | email, post, message, webhook, ticket, PR | yes |
| writes | insert, update, file write, store put, memory save | yes |
| runs | shell, code execution, SQL the model wrote, HTTP the model addressed | yes |
| reads | fetch, search, select with fixed query, file read with fixed path | no, unless the model writes the path, query or address |

For each acting tool, say what the model controls: which arguments it writes, and whether
the code fixes, validates or allow-lists them before the act (a parameter set in code, an
enum, a regex, a path joined under a fixed root and checked, an id looked up in the user's
own records).

A **gate** is something that stops the act until a person or the code agrees: an
`interrupt_before`/`interrupt_after` on the node, an `interrupt()` in the node, a
confirmation step the graph routes through, a dry-run flag the code enforces. A sentence in
the system prompt ("ask before deleting") is not a gate: count it as none. A gate that is
on in the code but off in how the team runs it (`langgraph.json`, the driver's config) is
none too; say which.

How the recordings settle it: `SELECT node, name, COUNT(*) FROM runs WHERE session_id = ?
AND run_type = 'tool' GROUP BY node, name` is what actually ran; a tool bound but never
called is still on the Reach list, marked "not seen in these runs". `shape` lists the nodes
that paused for a human, which is the gate firing.

Count for the check: acting tools at the node, and how many of them have a gate.
