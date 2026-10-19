---
name: secrets
description: Finding where a credential could be seen by the model or leave with its answer, by shape and name only, never by value.
---

### Secrets in the model's view

A secret in a prompt is a secret in the vendor's logs, in the team's traces and one request
away from the answer. Look for the shapes, never the values:

- key shapes: `sk-`, `sk-ant-`, `AKIA`, `ghp_`, `xox[bp]-`, `Bearer `, `-----BEGIN`, a JWT
  (`eyJ` with two dots), a 32+ character hex or base64 run beside a word like key, token,
  secret, password;
- env names: the names the census listed, and any name the code reads with `os.environ`,
  `os.getenv` or a settings class, appearing in a prompt, a completion or a tool argument;
- whole config objects or env dicts passed into a prompt or a tool's input (a `settings`
  object formatted into text, `os.environ` copied into a tool call);
- a tool that reads files by a model-written path, with `.env`, a key file or a config
  folder reachable from it.

How the recordings settle it, with `query` and SQL `LIKE` or `GLOB`; count rows, do not
print them:

```sql
SELECT node, COUNT(*) FROM runs WHERE session_id = ? AND run_type = 'llm'
  AND (prompt LIKE '%sk-%' OR prompt LIKE '%AKIA%' OR prompt LIKE '%Bearer %' OR prompt LIKE '%BEGIN %KEY%')
GROUP BY node;
SELECT node, COUNT(*) FROM runs WHERE session_id = ? AND (completion LIKE '%<ENV_NAME>%' OR outputs LIKE '%<ENV_NAME>%')
GROUP BY node;
```

Where a count is above zero, report the node, the shape and the count, and find the cause
in the code (which line formats what into the prompt). Never select the matching text
itself into your context. If a row must be read to tell a key from a look-alike, read its
length and its first three characters only.

Count for the check: calls checked at the node, and calls carrying a key shape or an env
name.
