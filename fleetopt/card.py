"""The agent card: what fleetopt knows of a project's agent, assembled by code from its records.

What an agent needs documented in 2026 is what the stations already establish: what it is
for and how it starts (the survey), what it spends (the measurement), what checks it (the
proof), what each expert found and left open (its run). The card is those records put
together, each line dated and tied to the code state it was true for; no model writes a
word of it. It lives with the other records (.fleetopt/agent.md), and a team copies it
where it wants it, into its AGENTS.md included.
"""

import json
import pathlib


def _runs(out, project):
    """The newest run.json of each expert for this project, newest first."""
    latest = {}
    for path in sorted(pathlib.Path(out).glob("runs/*/run.json"), reverse=True):
        try:
            facts = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(facts.get("project")) == str(project):
            latest.setdefault(facts.get("expert"), (path.parent.name[:8], facts))
    return latest


def render(out, project, survey=None):
    project = pathlib.Path(project)
    runs = _runs(out, project)
    when = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    lines = [f"# {project.name}: agent card",
             "Assembled by fleetopt from its records, never written by a model. Each line says when it was "
             "established and on which code (the state in `.fleetopt/runs/*/run.json`)."]
    if survey:
        lines += ["", "## What it is, how it starts, what checks it", "", survey.strip()]
    measured = [(d, f) for d, f in runs.values() if f.get("measured") and f.get("baseline")]
    if measured:
        d, f = max(measured, key=lambda pair: pair[0])  # two on one day: either, they measured the same code
        b = f["baseline"]
        cost = "not priced" if b.get("cost_usd") is None else f"${b['cost_usd']:.4f}"
        lines += ["", "## As measured", "",
                  f"- {f.get('agent')} ({f.get('graph')}), {when(d)}, code {str(f.get('code_state'))[:12]}: {cost} a run, "
                  f"{b.get('input_tokens') or 0:,.0f} tokens in, {b.get('output_tokens') or 0:,.0f} out, "
                  f"{b.get('llm_calls') or 0:g} model calls, {b.get('completed')} requests finished",
                  f"- checked by: {f.get('proof')}"] + ([f"- reached: {f['reach']}"] if f.get("reach") else [])
    if runs:
        lines += ["", "## Stations", "", "| expert | when | code | result | checks | report |", "|---|---|---|---|---|---|"]
        for name, (d, f) in sorted(runs.items()):
            checks = f.get("checks") or {}
            result = (f"{f['kept']} change(s) kept: {f.get('whole')}" if f.get("kept") else "nothing kept" if f.get("mode") == "apply"
                      else "reviewed" if f.get("measured") else "could not measure")
            lines.append(f"| {name}@{f.get('expert_version', '?')} | {when(d)} | {str(f.get('code_state'))[:12]} | {result} | "
                         f"{checks.get('closed', 0)} of {checks.get('total', 0)} closed | {pathlib.Path(f['run_dir']).name}/report.md |")
    return "\n".join(lines) + "\n"


def write(out, project, survey=None):
    out = pathlib.Path(out)
    path = out / "agent.md"
    path.write_text(render(out, project, survey), encoding="utf-8")
    return path
