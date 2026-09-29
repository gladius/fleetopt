"""The reviewer: a separate, read-only session with its own context.

It looks at one capture of the agent and the source, and reports two things: where
the agent wastes money, and whether its design fits its job. It changes nothing and
measures nothing. Its findings are numbered and saved against the exact code they
were made on, so `fleetopt apply` can work from them later, and only from them.

Same separation as the judge: whoever patches does not get to grade the design in
the same breath. (A Claude Code subagent was tried first for this: it ran in the
background, the caller ended its turn, and the review was lost.)
"""

import datetime
import hashlib
import json
import os
import pathlib
import re

from fleetopt import config

_HERE = pathlib.Path(__file__).parent
_SKILLS = _HERE / "plugin" / "skills"
_PATTERNS = _SKILLS / "patterns"
COST_SKILLS = ("caching", "model-tier", "prompt-growth", "redundant-work", "tool-surface")


def _body(path):
    """A skill's text without its frontmatter: the reviewer reads it, nothing loads it."""
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip() if text.startswith("---") else text


COST = (_HERE / "COST.md").read_text(encoding="utf-8")
MECHANICS = "\n\n".join(f"<!-- fleetopt:{name} -->\n" + _body(_SKILLS / name / "SKILL.md") for name in COST_SKILLS)
GUIDE = _body(_PATTERNS / "SKILL.md")
# Dated framework facts, read from installed packages. Local files, no network: the
# reviewer reads another team's code and gets no browser.
REFERENCES = "\n\n".join(f.read_text(encoding="utf-8") for f in sorted((_PATTERNS / "references").glob("*.md")))

# The shape `findings` parses and `fleetopt apply` works from. Change both together.
FORMAT = """# The report

Your final message is the report, in exactly this shape, and nothing else:

Job: <one sentence: what this agent is for>
Evidence: <n> traces of <m> distinct inputs, from one capture. Eval cases: <as given to you>

## Cost

### C1 - <short title>
pattern: <name> on <node(s)>
evidence: <the number from the traces>
source: <file:line>
change: <what to change, in one or two sentences>
effect: <what the traces say it would remove per request: calls, tokens, hops. Not measured>
risk: <what could change in the answers, or what an input outside the sample could do>

{design_section}## Checked and fine

- <what you checked>: <the number that cleared it>

{rules}
Number the findings in the order you would apply them, largest effect first. The same
change is one finding, not a C and a D. A pattern that fits is not a finding: it goes
under "Checked and fine". Under a heading with nothing to report write "None found."
You saw one capture, so you report evidence and the change it points to, never a
measured saving."""

COST_RULE = """- C, cost: the graph keeps its nodes and edges. What changes is what is sent, to which
  model, how much comes back, or when a loop that already exists stops: caching, a
  trimmed or no longer re-sent prompt, a bounded output, a smaller model or lower
  effort on a node, an early exit, an identical call not repeated. fleetopt checks this:
  a cost change that alters the graph is undone.
"""
DESIGN_RULE = """- D, design: nodes or edges are removed, merged or rewired. Tier one is mechanical
  (hardwire a branch always taken, drop a round that never changes anything, merge two
  calls, turn a fixed-order supervisor into edges). Tier two is a redesign.
"""
TAIL = """Every change that is tried is measured and judged before it is kept, and a person
reads the branch before anything is merged. So never hold a finding back because it feels
risky: say what could go wrong on the risk line.
"""
NO_DESIGN = """Only cost is reviewed in this run. A change that would remove, merge or rewire nodes
or edges is not a cost finding: leave it out, and if the design looks badly wrong for
the job, say so in one line under "Checked and fine" (for example "design not reviewed:
a supervisor consulted on every step that never varies its order").
"""
INTRO = ("You review the LangGraph agent in the current directory for a central AI team. You may read "
         "source and query the capture database with the fleetopt tools; you never edit files, never run "
         "anything, never patch. Every finding cites a number from graph_shape or query_traces and a source "
         "location. Where a guide below says to measure, judge or patch, that is for whoever applies your "
         "findings later, not for you.\n\n")


def system(design=False):
    """The reviewer's instructions: cost always, design only when asked for."""
    body = "# Part one: cost\n\n" + COST + "\n\n## The mechanics each pattern refers to\n\n" + MECHANICS
    if design:
        body += "\n\n# Part two: design\n\n" + GUIDE + "\n\n" + REFERENCES
        rules = "C or D is a fact about the change, not a judgement of its risk:\n" + COST_RULE + DESIGN_RULE
        section = "## Design\n\n### D1 - <short title>\n(the same lines, and then)\ntier: <one | two>\n\n"
    else:
        rules, section = "What a cost finding is:\n" + COST_RULE + NO_DESIGN, ""
    return INTRO + body + "\n\n" + FORMAT.replace("{design_section}", section).replace("{rules}", rules + TAIL)


SYSTEM = system(design=True)

READ_ONLY = ["Read", "Grep", "Glob", "mcp__fleetopt__graph_shape",
             "mcp__fleetopt__graph_topology", "mcp__fleetopt__query_traces"]

KINDS = ("cost", "design", "redesign")
LEVELS = ("fit", "wasteful", "over-built", "wrong shape", "broken")
_FINDING = re.compile(r"^#{2,4}\s*\**([CD]\d+)\**\s*[-:\u2013\u2014]\s*(.+?)\s*$", re.M)
_TIER = re.compile(r"^\W*tier\W*:\W*(one|two|1|2)\b", re.M | re.I)
_NO_CHANGE = re.compile(r"^\W*change\W*:\W*(none|nothing|n/?a)\b", re.M | re.I)


def findings(report):
    """The numbered findings of a report: [{id, title, kind}].

    The kind is read off the report by rule, and what may be tried follows from the
    kind in cli.chosen, not from an opinion of the reviewer's (observed: left to
    choose, a reviewer marked every cost finding "needs cases" and nothing was tried).
    A design finding whose tier cannot be read is taken as the larger change."""
    parts = _FINDING.split(report)  # [before, id, title, body, id, title, body, ...]
    found = []
    for i in range(1, len(parts) - 2, 3):
        if _NO_CHANGE.search(parts[i + 2]):  # observed: "C3 - Nothing to cache here", numbered like a finding
            continue
        name, tier = parts[i].upper(), _TIER.search(parts[i + 2])
        if name.startswith("C"):
            kind = "cost"
        else:
            kind = "design" if tier and tier.group(1).lower() in ("one", "1") else "redesign"
        found.append({"id": name, "title": parts[i + 1].strip("* "), "kind": kind})
    return found


def level(found, unfinished=0):
    """How far the agent is from where it should be, 0 to 4: the largest kind of change
    the review found, and above them all an agent that does not finish its requests."""
    if unfinished:
        return 4
    return max([KINDS.index(f["kind"]) + 1 for f in found], default=0)


def _pointer(out, project, run_cmd, design=False):
    from fleetopt.drive import entry  # one agent is one way of starting it

    kind = "-design" if design else ""  # a review of cost alone does not answer a request for design
    return entry.path_for(out, project, "review-" + hashlib.sha1(run_cmd.encode()).hexdigest()[:8] + kind)


def remember(out, project, run_cmd, state, label, run_dir, found, unfinished=0, design=False):
    """Note which code the newest review of this agent was made on. Returns the note."""
    record = {"code_state": state, "label": label, "run_dir": str(run_dir), "findings": found,
              "unfinished": unfinished, "level": level(found, unfinished),
              "when": datetime.datetime.now().isoformat(sep=" ", timespec="minutes")}
    path = _pointer(out, project, run_cmd, design)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def saved(out, project, run_cmd, state, design=False):
    """The newest review of this agent, with its text, if it was made on this exact
    code. `state=None` asks for the newest review whatever code it saw."""
    path = _pointer(out, project, run_cmd, design)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    text = pathlib.Path(record["run_dir"]) / "review.md"
    if not text.exists() or not record.get("code_state"):
        return None
    if state is not None and record["code_state"] != state:
        return None
    report = text.read_text(encoding="utf-8")
    found = findings(report)  # the report is the record; the rule may have moved
    return {**record, "text": report, "findings": found, "level": level(found, record.get("unfinished", 0))}


async def run(project, label, purpose, model=None, max_usd=1.0, design=False):
    """Returns (report text, cost in USD). Raises if the session fails."""
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKError, ResultMessage, TextBlock,
                                  query)

    from fleetopt.optimizer import tools

    options = ClaudeAgentOptions(
        cwd=str(project),
        model=model or os.environ.get("FLEETOPT_REVIEW_MODEL") or "claude-sonnet-5",
        system_prompt=system(design),
        tools=["Read", "Grep", "Glob"],
        # A read-only server: nothing that runs the agent, changes a file or grades a change.
        mcp_servers={"fleetopt": tools.server(READ_ONLY)},
        allowed_tools=READ_ONLY,
        disallowed_tools=["AskUserQuestion", *config.DENY_READS],
        skills=[],
        setting_sources=config.SETTING_SOURCES,
        extra_args=config.sdk_args(),
        strict_mcp_config=True,
        env=config.SDK_ENV,
        max_turns=30,
        max_budget_usd=max_usd,
        permission_mode="default",
    )
    prompt = (f"Review the agent in this directory. Its traces are under the label {label!r}. "
              f"{purpose}")
    texts, final, failure, cost = [], None, None, None
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                texts += [b.text for b in message.content if isinstance(b, TextBlock) and b.text.strip()]
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    failure = message.result or message.subtype
                final = message.result
                cost = getattr(message, "total_cost_usd", None)
    except ClaudeSDKError as exc:  # out of turns or budget arrives as an exception, not as a message
        failure = str(exc).splitlines()[0][:300]
    if failure:
        raise RuntimeError(f"reviewer failed: {failure}")
    report = (final or (texts[-1] if texts else "")).strip()
    if not report:
        raise RuntimeError("reviewer returned nothing")
    return report, cost
