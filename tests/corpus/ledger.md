# Corpus ledger: fleetopt against real agents

One line per run on an open-source agent. This is how fleetopt improves: every
break becomes a test in `tests/`, every recurring pattern becomes a line in a
skill, and the same repos are re-run after the fix. Candidates come from
`corpus_build.py` (GitHub search, `corpus.md`) and the hand-verified
`survey-2026-09-26.md`. Cloned targets live in `targets/` (gitignored).

Columns: capture = did the zero-edit probe see runs; skills = which fleetopt skills
fired; saving = measured cost delta on the headline metric with the compare verdict;
judge = equivalence and, when eval cases loaded, correctness k/m before -> after;
broke = what failed in fleetopt, and the test or fix that came out of it.

| Date | Repo | Pattern | Size | Run command | Capture | Skills | Saving | Judge | Review findings | Broke -> fix |
|---|---|---|---|---|---|---|---|---|---|---|
| planned | langchain-ai/react-agent | single ReAct | S | `langgraph dev` (Tavily key) | | | | | | smoke test first |
| planned | langchain-ai/memory-agent | ReAct + memory store | S | `langgraph dev` | | | | | | already in targets/ |
| planned | langchain-ai/data-enrichment | ReAct + reflection | M | `langgraph dev` (Tavily key) | | | | | | |
| planned | langchain-ai/agents-from-scratch | router + ReAct + HITL | M | `langgraph dev` (OpenAI key) | | | | | | expected tool-call dataset |
| planned | langchain-ai/open_deep_research | supervisor, parallel researchers | L | `langgraph dev` (OpenAI, Tavily) | | | | | | archived; or deepagents/examples/deep_research |
| planned | assafelovic/gpt-researcher (multi_agents) | supervisor + review/revise loop | XL | `python main.py` (OpenAI, Tavily) | | | | | | review target |
| planned | langchain-ai/executive-ai-assistant | router + reflection graphs, two providers | L | needs Gmail OAuth | | | | | | review only, not a runner |
