---
name: unchecked-input
description: Text no user typed that reaches a prompt, and whether anything stands between it and the next act.
---

### Unchecked input: instructions that arrive without a user

Every instruction that reaches a model did not necessarily come from the user. Sources, each
a place an attacker can write to without sending the agent a request:

| Source | Where it enters | Who can write to it |
|---|---|---|
| retrieval | documents, chunks, search results into a prompt | whoever can get a document indexed |
| files | a file the agent reads by name or path | whoever can place a file |
| web | pages, APIs, scraped text | the page's owner |
| tool output | an error message, a record, a payload copied whole | the system behind the tool |
| another agent | a worker's answer into a supervisor's prompt | whoever reached the worker |
| memory or store | what an earlier request saved | an earlier user |

For each source, in the code: which node's prompt it enters, how much of it (whole payload or
a field), and what it goes on to: another prompt, a tool argument, the answer, a store. A
**boundary** is something the code does between the source and the next act: a schema the
text is parsed into before use, a tool whose arguments are then fixed, a human gate, a
separate model call that only classifies. Delimiters and "treat the following as data" in
the prompt are not boundaries.

A store written from user content and read into later prompts is the case to call out on its
own: one request poisons every later one.

How the recordings settle it: `prompt_chars` per node and step shows how much arrives; the
tool runs before a model call show which payload was copied; a prompt that contains a tool
output verbatim is found by comparing the tool run's `outputs` length with the growth in the
next call's `prompt_chars`.

Count for the check: sources entering this node's prompt, and prompts recorded carrying one
with no boundary before the next act.
