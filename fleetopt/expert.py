"""An expert, loaded from a folder of prose, fetched from the central catalogue when needed.

An expert is a folder: GUIDE.md, whose header says what it is asked to do, what its changes
must earn and which experts it may call; CHECKS.md, what it tracks (which rows, and what
settles each check); and skills/<name>/SKILL.md. Nothing in the folder is code, so it can be
written, reviewed, zipped and served like any document. No expert lives in fleetopt's code.

Where an expert is found, first hit wins:
1. a directory named in FLEETOPT_EXPERTS (someone's deliberate choice);
2. `experts/` beside the code, when fleetopt runs from a checkout: the catalogue itself;
3. ~/.config/fleetopt/experts: what this machine has pulled;
4. the central catalogue at FLEETOPT_CENTRAL (http): `<address>/<name>.zip` is pulled into 3.

What stays in the code, shared, is what a central team owns: GUIDE.md beside this file (how
to start a team's agent and work with the tools), the tools themselves (tools.py), and the
rules below for what a change must earn. A header names a rule or a tool; it cannot define
one, so no expert can loosen its own proof.
"""

import dataclasses
import hashlib
import io
import os
import pathlib
import shutil
import sys
import urllib.request
import zipfile

SHARED = pathlib.Path(__file__).parent / "GUIDE.md"
SOURCE = pathlib.Path(__file__).parent.parent / "experts"
LOCAL = pathlib.Path.home() / ".config" / "fleetopt" / "experts"


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
# What a list of checks may be made over. Code makes the rows from what was recorded:
# spenders: the nodes that carry more than a twentieth of the tokens; nodes: every node of the
# graph, run or not; branches: every branch a branch point declares.
ROWS = ("spenders", "nodes", "branches")


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
    calls: tuple = ()         # experts it may hand a task to
    rows: str = "spenders"    # what its list of checks is made over (ROWS)
    checks: tuple = ()        # the checks: each is closed, on every row, with what settles it
    checks_text: str = ""     # CHECKS.md as written: what each check means and what settles it

    @property
    def earned(self):
        return RULES[self.earns][0] if self.earns else (lambda result: None)

    @property
    def nothing(self):
        return RULES[self.earns][1] if self.earns else "this expert only reviews"

    @property
    def version(self):
        """A fingerprint of everything in the folder: which expert a run used, exactly."""
        digest = hashlib.sha1()
        for path in sorted(p for p in self.folder.rglob("*") if p.is_file()):
            digest.update(path.relative_to(self.folder).as_posix().encode() + b"\0" + path.read_bytes())
        return digest.hexdigest()[:8]

    @property
    def skills(self):
        return sorted(p.name for p in (self.folder / "skills").glob("*") if (p / "SKILL.md").is_file())

    def system(self):
        """The shared guide, this expert's, then the mechanics of each skill it has: all in
        the prompt, cached after the first turn, so it never works without the one it needs."""
        return (SHARED.read_text(encoding="utf-8").strip() + "\n\n" + header(self.folder / "GUIDE.md")[1]
                + (f"\n\n## What you check\n\n{self.checks_text}" if self.checks_text else "")
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
    rows, checks, text = meta.get("rows", "spenders"), _names(meta.get("checks")), ""
    if (folder / "CHECKS.md").is_file():  # what it tracks, in its own file: `- name: what settles it`, one a line
        tracked, text = header(folder / "CHECKS.md")
        rows = tracked.get("rows", rows)
        checks = tuple(line[2:].partition(":")[0].strip(" *`") for line in text.splitlines()
                       if line.startswith("- ") and ":" in line)
    if rows not in ROWS:
        raise ValueError(f"{folder}: `rows: {rows}` is not something fleetopt can list ({', '.join(ROWS)})")
    return Expert(folder=folder, name=meta.get("name") or folder.name, does=meta.get("does", ""), review=meta["review"],
                  apply=meta.get("apply") or None, earns=meta.get("earns") or None,
                  keeps_shape=meta.get("structure", "kept") != "free", tools=_names(meta.get("tools")),
                  calls=_names(meta.get("calls")), rows=rows, checks=checks, checks_text=text)


def places():
    """Where expert folders are looked for, in order: the first one with a name wins."""
    named = [pathlib.Path(p).expanduser() for p in (os.environ.get("FLEETOPT_EXPERTS") or "").split(os.pathsep) if p]
    return [*named, SOURCE, LOCAL]


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


def pull(name, central=None):
    """Fetch an expert from the central catalogue into this machine's folder, replacing the one
    there. The catalogue serves `<address>/<name>.zip`, the folder's files at its top level."""
    central = (central or os.environ.get("FLEETOPT_CENTRAL") or "").rstrip("/")
    if not central:
        raise ValueError("no central catalogue is set (FLEETOPT_CENTRAL)")
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"{name!r} is not an expert's name")
    try:
        with urllib.request.urlopen(f"{central}/{name}.zip", timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not pull {name} from {central}: {exc}") from None
    staging = LOCAL / f".{name}.pulling"
    shutil.rmtree(staging, ignore_errors=True)
    for item in archive.infolist():
        target = (staging / item.filename).resolve()
        if not target.is_relative_to(staging.resolve()) or not item.filename.endswith((".md", "/")):
            shutil.rmtree(staging, ignore_errors=True)  # prose only, and only inside its own folder
            raise ValueError(f"{name} from {central} holds {item.filename!r}: an expert is markdown files in its own folder")
    archive.extractall(staging)
    try:
        read(staging)
    except (ValueError, OSError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"{name} from {central} is not an expert: {exc}") from None
    shutil.rmtree(LOCAL / name, ignore_errors=True)
    staging.rename(LOCAL / name)
    EXPERTS.clear()
    EXPERTS.update(load())
    return read(LOCAL / name)


def get(name):
    """The expert of that name: one found here, else pulled from the central catalogue."""
    if name in EXPERTS:
        return EXPERTS[name]
    if os.environ.get("FLEETOPT_CENTRAL"):
        pull(name)
        if name in EXPERTS:
            return EXPERTS[name]
    raise ValueError(f"no expert named {name!r} (here: {', '.join(sorted(EXPERTS)) or 'none'}"
                     + ("" if os.environ.get("FLEETOPT_CENTRAL") else "; no central catalogue is set (FLEETOPT_CENTRAL)") + ")")


EXPERTS = load()
COST, DESIGN = EXPERTS.get("cost"), EXPERTS.get("design")
