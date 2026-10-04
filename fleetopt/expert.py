"""An expert, loaded from a folder of prose.

An expert is a folder: GUIDE.md, whose header says what it is asked to do and what its
changes must earn, and skills/<name>/SKILL.md. Nothing in the folder is code, so it can be
written, reviewed, zipped and handed over like any document. fleetopt looks for expert
folders in its own experts/ directory, then in every directory named in FLEETOPT_EXPERTS,
then in ~/.config/fleetopt/experts: put a folder there and it is an expert.

What stays here, shared and in code, is what a central team owns: experts/GUIDE.md (how to
start a team's agent and work with the tools), the tools themselves (tools.py), and the
rules below for what a change must earn. A header names a rule or a tool; it cannot define
one, so no expert can loosen its own proof.
"""

import dataclasses
import os
import pathlib
import sys

BUILT_IN = pathlib.Path(__file__).parent / "experts"
HOME = pathlib.Path.home() / ".config" / "fleetopt" / "experts"


# --- the rules a header may name: what a measured change must show to be kept ----------------

GAINS = ("cost_usd", "wall_ms", "completed", "cost_per_completed")


def gain(result):
    """What got better past the noise, or None. Nothing may finish less often and cost may
    not rise. A clean cut in tokens counts too, with neither count worse: total cost can sit
    inside the noise because the length of a model's answers varies from run to run, which
    the change does not control (observed: input tokens -20% on every run, cost -11% called
    noise)."""
    verdict = lambda key: (result.get(key) or {}).get("verdict")
    if "regressed" in (verdict("completed"), verdict("cost_usd")):
        return None
    first = next((k for k in GAINS if verdict(k) in ("improved", "baseline finished nothing")), None)
    tokens = (verdict("input_tokens"), verdict("output_tokens"))
    if first or "regressed" in tokens:
        return first
    return next((k for k, v in zip(("input_tokens", "output_tokens"), tokens) if v == "improved"), None)


RULES = {"cheaper": (gain, "nothing got better past the noise")}


# --- the folder ------------------------------------------------------------------------------

def header(path):
    """(the header of a markdown file as {key: value}, its body). The header is the lines
    between the first two `---`: `key: value`, a value going on over indented lines."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}, text.strip()
    _, head, body = text.split("---", 2)
    meta, key = {}, None
    for line in head.splitlines():
        if line[:1] in (" ", "\t") and key:
            meta[key] = (meta[key] + " " + line.strip()).strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            meta[key] = value.strip()
    return meta, body.strip()


def _names(value):
    return tuple(n.strip() for n in (value or "").split(",") if n.strip())


@dataclasses.dataclass(frozen=True)
class Expert:
    folder: pathlib.Path
    name: str
    does: str                 # in a few words, for the command line
    review: str               # what it is asked when it only looks
    apply: str | None = None  # ... and when it changes the agent; None: it only looks
    earns: str | None = None  # the rule a change of its must meet (RULES)
    keeps_shape: bool = True  # a change to the graph's nodes or edges is refused
    tools: tuple = ()         # shared tools it uses beside the ones every expert has
    checks: tuple = ()        # what it checks on every node that matters: its report accounts for each

    @property
    def earned(self):
        return RULES[self.earns][0] if self.earns else (lambda result: None)

    @property
    def nothing(self):
        return RULES[self.earns][1] if self.earns else "this expert only reviews"

    @property
    def skills(self):
        return sorted(p.name for p in (self.folder / "skills").glob("*") if (p / "SKILL.md").is_file())

    def system(self):
        """The shared guide, this expert's, then the mechanics of each skill it has: all in
        the prompt, cached after the first turn, so it never works without the one it needs."""
        return ((BUILT_IN / "GUIDE.md").read_text(encoding="utf-8").strip() + "\n\n" + header(self.folder / "GUIDE.md")[1]
                + "\n\n## The mechanics each pattern refers to\n\n"
                + "\n\n".join(f"<!-- fleetopt:{n} -->\n" + header(self.folder / "skills" / n / "SKILL.md")[1]
                              for n in self.skills))


def read(folder):
    """The expert in a folder. ValueError when its header does not make one."""
    folder = pathlib.Path(folder)
    meta, _ = header(folder / "GUIDE.md")
    if not meta.get("review"):
        raise ValueError(f"{folder}: GUIDE.md needs a header with at least `review:` (what the expert is asked)")
    if meta.get("earns") and meta["earns"] not in RULES:
        raise ValueError(f"{folder}: `earns: {meta['earns']}` is not a rule fleetopt has ({', '.join(RULES)})")
    if meta.get("apply") and not meta.get("earns"):
        raise ValueError(f"{folder}: an expert that changes code (`apply:`) must name what a change earns (`earns:`)")
    return Expert(folder=folder, name=meta.get("name") or folder.name, does=meta.get("does", ""), review=meta["review"],
                  apply=meta.get("apply") or None, earns=meta.get("earns") or None,
                  keeps_shape=meta.get("structure", "kept") != "free", tools=_names(meta.get("tools")),
                  checks=_names(meta.get("checks")))


def places():
    """Where expert folders are looked for, fleetopt's own first: a folder elsewhere cannot
    replace a built-in expert."""
    named = [pathlib.Path(p).expanduser() for p in (os.environ.get("FLEETOPT_EXPERTS") or "").split(os.pathsep) if p]
    return [BUILT_IN, *named, HOME]


def load():
    found = {}
    for place in places():
        for guide in sorted(place.glob("*/GUIDE.md")) if place.is_dir() else ():
            try:
                expert = read(guide.parent)
            except ValueError as exc:  # one bad folder must not take the others with it
                print(f"[fleetopt] not an expert: {exc}", file=sys.stderr)
                continue
            found.setdefault(expert.name, expert)
    return found


EXPERTS = load()
COST, DESIGN = EXPERTS["cost"], EXPERTS["design"]
