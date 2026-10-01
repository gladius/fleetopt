"""The experts fleetopt can send to a team's agent.

An expert is a folder of markdown, its GUIDE.md and its skills, and a few facts here: what
it is asked to do, and what a change of its must earn before it is kept. Everything else is
the same for all of them: the session and its guards (session.py), the tools (tools.py), and
GUIDE.md in this folder, the part of the guide they share: how to start a team's agent and
how to work with the tools.
"""

import dataclasses
import pathlib

_HERE = pathlib.Path(__file__).parent


def _body(path):
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[2].strip() if text.startswith("---") else text


@dataclasses.dataclass(frozen=True)
class Expert:
    name: str
    does: str                  # in a few words, for the command line
    review: str                # what it is asked when it only looks
    apply: str | None = None   # ... and when it changes the agent; None: it only looks
    earned: object = None      # what a measured change improved past the noise, or None
    nothing: str = ""          # why keep refuses a change that improved nothing
    keeps_shape: bool = True   # a change to the graph's nodes or edges is refused
    tools: tuple = ()          # tools of its own, beside the ones every expert has

    @property
    def skills(self):
        return sorted(p.name for p in (_HERE / self.name / "skills").iterdir() if p.is_dir())

    def system(self):
        """The shared guide, this expert's, then the mechanics of each skill it names: all in
        the prompt, cached after the first turn, so it never works without the one it needs."""
        guide = lambda path: path.read_text(encoding="utf-8").strip()
        return (guide(_HERE / "GUIDE.md") + "\n\n" + guide(_HERE / self.name / "GUIDE.md")
                + "\n\n## The mechanics each pattern refers to\n\n"
                + "\n\n".join(f"<!-- fleetopt:{n} -->\n" + _body(_HERE / self.name / "skills" / n / "SKILL.md")
                              for n in self.skills))


# --- cost: cheaper, and nothing broken -------------------------------------------------------

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


COST = Expert(
    name="cost", does="where it wastes tokens and money",
    review="Review the LangGraph agent in this project: start it if needed, measure it once as it is, find where "
           "it wastes tokens and money, and report what is worth changing. This run changes nothing and does not "
           "run the team's evals; say what the project has to check the agent against (an eval suite and how it "
           "is run, a golden dataset, examples of what it is sent) or that it has none of these.",
    apply="Make the LangGraph agent in this project cost less without changing what it answers: start it if "
          "needed, measure it, find the waste, change it, prove each change, look again, and report.",
    earned=gain, nothing="nothing got better past the noise")

# --- design: does the structure fit the job. It only looks, for now ---------------------------

DESIGN = Expert(
    name="design", does="whether its design fits its job, and what a simpler one would do",
    review="Review the design of the LangGraph agent in this project: start it if needed, on requests that differ in "
           "kind, measure it once as it is, name the patterns it is built from by what ran, check each against the "
           "numbers, and report what is broken and where a simpler design would do the same job. This run changes "
           "nothing and does not run the team's evals; say what the project has to check the agent against (an eval "
           "suite and how it is run, a golden dataset, examples of what it is sent) or that it has none of these.",
    tools=("shape",))

EXPERTS = {e.name: e for e in (COST, DESIGN)}
