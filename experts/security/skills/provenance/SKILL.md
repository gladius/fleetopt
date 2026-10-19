---
name: provenance
description: Following a model-written argument to where it is used: shell, query, path, host, amount, and what checks it on the way.
---

### Provenance: where the model's words end up

For each tool argument the model writes, follow it in the code to its sink:

| Sink | What to look for | Fine when |
|---|---|---|
| shell or code | `subprocess`, `os.system`, `exec`, `eval`, a sandbox API | the argument is one of a fixed set, or runs in a sandbox the code sets up |
| query | SQL, Cypher, a search DSL, a filter string | parameters bound, not formatted into the string |
| path | `open`, `Path`, `os.*` | joined under a fixed root and checked to stay inside it |
| host or URL | `httpx`, `requests`, `aiohttp`, a webhook | the host is fixed or allow-listed in code |
| amount or recipient | money, quantity, an address, an account | looked up from the user's own records, not taken from the text |
| identity | a user id, a tenant, a role in the arguments | taken from the request's context, never from what the model wrote |

The last row is the one teams miss: a tool that takes `user_id` from the model's arguments
lets any request act as any user, whatever the prompt says.

How the recordings settle it: the arguments the model actually wrote are in each tool run's
`inputs`. Query the calls of a tool across all requests and compare the arguments with the
request text in the root run's `inputs`: a path, host or id that came from the request is
model-composed; one that is identical every run is fixed.

Count for the check: arguments at this node that reach a sink unchecked, and recorded calls
that carried a model-composed value into one.
