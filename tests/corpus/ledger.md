# Corpus ledger: fleetopt against real agents

One line per run on an open-source agent. This is how fleetopt improves: every
break becomes a test in `tests/`, every recurring pattern becomes a line in a
skill, and the same repos are re-run after the fix. Candidates come from
`corpus_build.py` (GitHub search with a token, 2026-09-27: 1,506 repos seen, 584 active
candidates inspected, 304 with LangGraph as a confirmed dependency, 207 of them
applications; full table in `corpus.md`) and the hand-verified `survey-2026-09-26.md`. Cloned targets live in `targets/` (gitignored).

Columns: capture = did the zero-edit probe see runs; skills = which fleetopt skills
fired; saving = measured cost delta on the headline metric with the compare verdict;
judge = equivalence and, when eval cases loaded, correctness k/m before -> after;
broke = what failed in fleetopt, and the test or fix that came out of it.

| Date | Repo | Pattern | Size | Run command | Capture | Skills | Saving | Judge | Review findings | Broke -> fix |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-27 | hoangsonww/Agentic-AI-Pipeline | hand-built plan / decide / act / reflect loop over 7 tools | L | driver calling its `run_chat`, Haiku 4.5 via env, 3 inputs | 20 runs, crashed on the second tool round ($0.01) | review only | - | - | 4 defects with source lines: state field has no reducer so tool results replace history; `act` rebuilds a prompt that orphans tool results (Anthropic 400); `reflect` filters out tool output and never sees evidence; "exactly one tool" is prompt text, not enforced. First pass missed the redesign finding; the second pass, on the same capture after the guide and evidence change, made it: 6 model calls per round of tool use, replace the loop with a tool-calling agent (tier two) | failure message had no output (fixed + test); review refused crashed runs (fixed); single-trace noise (fixed); catalog lacked the hand-built loop (added); app never exports its key file (run command) |
| 2026-09-28 | hoangsonww/Agentic-AI-Pipeline, the redesign tried | loop replaced by one tool-calling agent, on a branch of our copy (2 files, 169 lines out, 37 in; 6 nodes to 2) | L | same driver, 3 requests, 3 runs each side, Haiku 4.5 | before 0/9 requests finish, after 9/9 | change made by hand, not by the optimizer | model calls per run 15 to 8, wall -34%, cost per run $0.0215 to $0.0275, which is $0.009 per finished request against a baseline that finished nothing | equivalence not applicable (the baseline produces no output); correctness on three reference cases written by us: 0/3 to 2/3, the email draft is saved to a file and missing from the answer | - | compare called it a cost regression (added finished requests and cost per finished request + test); the first replacement lost tolerance for failing tools (framework middleware); the agent writes files into its own repo, which dirties the tree and would shift the code fingerprint if tracked (open) |
| 2026-09-27 | kevin333353/jobsmith | pipeline + rule-based gate + 3-way fan-out + critic + human gate | L | its own CLI on 2 job descriptions, one mismatched; models fixed in code (Haiku, Sonnet 4.6, Opus 4.8) | 46 runs, exit 0 ($0.14; first attempt $0.12) | review only | - | - | critic raises on every call (Opus rejects `temperature`), node swallows it and reports the check as passed; an LLM supervisor module exists but is never wired in; rule-based gate, fan-out and human gate all fit | node failures were invisible in the structural evidence (added + test); target prompts on stdin (run command) |
| planned | langchain-ai/react-agent | single ReAct | S | `langgraph dev` (Tavily key) | | | | | | smoke test first |
| planned | langchain-ai/memory-agent | ReAct + memory store | S | `langgraph dev` | | | | | | already in targets/ |
| planned | langchain-ai/data-enrichment | ReAct + reflection | M | `langgraph dev` (Tavily key) | | | | | | |
| planned | langchain-ai/agents-from-scratch | router + ReAct + HITL | M | `langgraph dev` (OpenAI key) | | | | | | expected tool-call dataset |
| planned | langchain-ai/open_deep_research | supervisor, parallel researchers | L | `langgraph dev` (OpenAI, Tavily) | | | | | | archived; or deepagents/examples/deep_research |
| planned | assafelovic/gpt-researcher (multi_agents) | supervisor + review/revise loop | XL | `python main.py` (OpenAI, Tavily) | | | | | | review target |
| planned | langchain-ai/executive-ai-assistant | router + reflection graphs, two providers | L | needs Gmail OAuth | | | | | | review only, not a runner |
| planned | langchain-ai/langgraph-fullstack-python | chat template | S | `langgraph dev` (Anthropic or OpenAI) | | | | | | from corpus: smallest runnable |
| planned | infiniumtek/terraform-review-agent | single agent, tools | M | `langgraph dev` (Anthropic, Google or OpenAI) | | | | | | from corpus: tests present |
| planned | bernatsampera/event-deep-research | supervisor + research | M | `langgraph dev` (Google, Ollama or OpenAI) | | | | | | from corpus: small supervisor |
| planned | langchain-ai/langsmith-agent-lifecycle-workshop | supervisor + reflection + RAG + HITL | L | `langgraph dev` (Anthropic or OpenAI) | | | | | | from corpus: many patterns, eval files, teaching code |
| planned | SalesforceAIResearch/enterprise-deep-research | research + router | L | `langgraph dev` (Anthropic, Groq or OpenAI) | | | | | | from corpus: enterprise-shaped research agent |
| planned | langchain-ai/open-swe | coding agent: router + reflection + swarm + HITL | XL | `langgraph dev` (Anthropic) | | | | | | from corpus: review target, heavy |
