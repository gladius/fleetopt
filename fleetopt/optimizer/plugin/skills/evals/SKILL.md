---
name: evals
description: Find and load the project's existing eval cases (inputs with expected answers) before the baseline is measured. Use at the start of every run - when the repo has a tests/, evals/, deepeval, promptfoo, datasets or goldens folder, when a LangSmith or Galileo dataset name appears in the code, or when --evals was supplied. With cases loaded, judge grades correctness against the expected answers, not just "unchanged since the last run".
---

# Eval cases: turn "unchanged" into "correct"

Without expected answers the judge can only say the patch did not change the
output. With them it can say the output is still correct, and whether the
baseline was even correct to begin with. Teams rarely write golden data on
request, but many already have some. Find it.

## Where to look, in order

| Source | Files | A case looks like |
|---|---|---|
| deepeval | `test_*.py`, `evals/`, `.deepeval/` | `LLMTestCase(input="...", expected_output="...")` or `Golden(...)` |
| JSONL / JSON | `evals/`, `datasets/`, `goldens/`, `tests/fixtures/`, `*.jsonl` | objects with an input key (`input`, `query`, `question`, `prompt`) and an expected key (`expected`, `expected_output`, `reference`, `answer`, `ground_truth`) |
| promptfoo | `promptfooconfig.yaml` | `tests:` with `vars` and `assert` (not loadable yet - report the path) |
| pytest goldens | `tests/**/expected_*.json`, snapshot dirs | input and expected side by side |
| LangSmith | `create_dataset(`, `evaluate(`, `dataset_name=` in code | a dataset name (not fetchable yet - report the name for a human) |
| Galileo, Braintrust | imports of `galileo`, `promptquality`, `braintrust` | a project or dataset name (report it) |

Grep for these before reading files: `expected_output`, `LLMTestCase`, `Golden(`,
`dataset_name`, `ground_truth`, `"expected"`.

## What to do

1. Found JSONL, JSON or deepeval files: call `load_eval_cases` with the file or
   folder. It reports how many cases loaded and from where.
2. Prefer the suite that exercises those cases as the run command, so the captured
   runs match the cases. `judge` only grades runs whose input matches a case; the
   rest are reported as unmatched, and that means the run command did not
   exercise them.
3. Found only a dataset name (LangSmith, Galileo, promptfoo): do not try to fetch
   it. Put the name in the report so a human can export it.
4. Found nothing: say so in the report in one line - "correctness not checked: no
   eval cases in the repo". That is a finding about the team's agent, not a
   failure of the run.

## Rules

- Never run the eval framework yourself (`deepeval test run`, `promptfoo eval`).
  It invokes the agent on every case and bills the team for its own graders. The
  target runs only through `measure`.
- Never invent expected answers. A case without an expected output is not a case.
- Never write eval files into the team's repo. Loaded cases are kept in fleetopt's
  own output folder.
- A candidate must not pass fewer cases than the baseline. Equal is fine; the
  baseline may already fail some, and that is worth a line in the report.
