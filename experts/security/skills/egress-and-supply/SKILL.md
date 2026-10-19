---
name: egress-and-supply
description: What leaves the process and to whom, and what the agent runs that the team did not write.
---

### Egress: what leaves

Start from the census's external systems by package, then find the call in the code. For
each: which node, which tool or logger, what is sent (the user's text, a document, a record,
a prompt, an answer), to which system, under which credential name. Two that teams forget:

- **tracing and observability**: LangSmith, Langfuse, OTel exporters and the like ship every
  prompt and answer to a vendor by default; `LANGCHAIN_TRACING_V2`, `LANGSMITH_TRACING` or
  an exporter in the code is egress of everything the model sees; ADK ships OpenTelemetry
  tracing, exported wherever the code or the environment points it (Cloud Trace, an OTLP
  endpoint). Say whether it is on in how the team runs it.
- **logs**: a `print` or `logger.info` of a prompt, a payload or a request writes it to
  wherever logs go.

Count for the check: systems this node sends to, and what each is sent.

### Supply chain: what runs that the team did not write

- **MCP servers**: each one in `langgraph.json`, the code or a config, with its source
  (a package, a URL, a local command); an MCP server attached wholesale binds every tool
  it offers, so each is on the Reach list under agency with the server as its author.
- **skills, plugins, prompt files pulled at runtime**: from where, pinned to what.
- **packages**: whether the lock file pins them; packages that reach outside the process
  (the census list) matter most. `pip-audit` or `uv audit` output is a number when the team
  has run one; do not run it yourself.
- **models**: which provider and model each node calls, and whether the endpoint is the
  vendor's, a gateway, or a URL set by environment (`*_BASE_URL`), which is where a request
  can be redirected.

Count for the check: dependencies at this node the team did not write, and how many are
pinned or from a source the project names.
