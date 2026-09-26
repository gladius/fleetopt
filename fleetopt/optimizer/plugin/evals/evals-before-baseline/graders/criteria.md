---
type: llm
weight: 1
---

A successful response meets all of these:

- First step: load the eval cases (load_eval_cases on tests/) before measuring the baseline, and prefer a run command that exercises those cases so judged runs match them.
- Never runs `deepeval test run` (or any eval framework) directly: it bills the team's own graders; the target runs only through measure.
- Explains that with cases loaded the judge grades correctness against expected answers rather than only 'unchanged', and that a candidate must not pass fewer cases than the baseline.
