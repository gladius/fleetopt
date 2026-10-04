---
name: requests
description: Writing requests that exercise a LangGraph agent, one per route and tool, in its users' words.
---

### Writing requests that exercise an agent

Start from what decides the path, not from what the agent is called.

| What to exercise | Where to read it | What the request needs |
|---|---|---|
| A branch on a model's label (a router) | the prompt that asks for the label, and its allowed values | wording that a reader would label that way; one request a label |
| A branch on a rule | the branch function: a keyword, a field, a length, a count | the keyword, the field set, the size that flips it |
| A tool | its name and docstring as the model sees them, and the system prompt that mentions it | a need only that tool answers |
| A nested agent or worker | the hand-off or tool that reaches it, and the supervisor's prompt | a task that prompt assigns to that worker and to no other |
| A loop's exit | the condition that ends it | one request that ends it early, one that runs it long |
| Failure handling | what a tool does on a bad argument or an empty result | a request about something that does not exist |
| A human approval | the interrupt and what resumes it | a request that reaches it (the run pauses there: say so) |

Ways requests go wrong:

- **Too alike.** Eight phrasings of one question exercise one path. Vary the route, not the
  wording.
- **Too clean.** Real users send fragments, typos and two questions at once; include one.
- **Asking the stub.** When a tool returns canned data, a request outside that data only
  shows the stub: say so beside the request.
- **Private data.** A request that needs a person gets an invented one. Never anything
  about whoever runs this tool, and nothing copied from a real customer record without it
  being in the project already.

A dimension list keeps them different in kind: write down, for this agent, the routes, the
tools, the kinds of user and the kinds of trouble, then pick combinations that cover each
value once before any is used twice.
