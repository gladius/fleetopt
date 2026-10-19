"""An ADK agent tree for fleetopt's own tests: a coordinator that hands a fleet question to a
draft-then-review pipeline, or to a lookup agent with a tool. The review loop never stops
early (the hotspot), and the lookup branch is never taken by the example requests.

    coordinator (LlmAgent, transfers)
    ├── pipeline (SequentialAgent)
    │   ├── draft (LlmAgent, tool fetch_notes)
    │   └── review_loop (LoopAgent, 3 rounds)
    │       ├── critic (LlmAgent)
    │       └── reviser (LlmAgent)
    └── lookup (LlmAgent, tool lookup_fleet)
"""

import asyncio
import json
import pathlib

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from fake_llm import BudgetFakeLlm


def fetch_notes(topic: str) -> dict:
    """Notes the fleet team keeps on a topic."""
    return {"notes": [f"{topic}: note {i}" for i in range(3)]}


def lookup_fleet(vehicle_id: str) -> dict:
    """One vehicle's record."""
    return {"vehicle_id": vehicle_id, "status": "in service", "mileage": 120_000}


draft = LlmAgent(name="draft", model=BudgetFakeLlm(tool="fetch_notes", tool_args={"topic": "fleet"},
                                                   reply="draft: " + "finding " * 60, output_tokens=240),
                 instruction="Draft an answer from the notes.", tools=[fetch_notes], output_key="draft")
critic = LlmAgent(name="critic", model=BudgetFakeLlm(reply="needs more detail on costs"),
                  instruction="Critique the draft: {draft?}", output_key="critique")
reviser = LlmAgent(name="reviser", model=BudgetFakeLlm(reply="revised: " + "finding " * 60, output_tokens=240),
                   instruction="Revise the draft {draft?} after the critique {critique?}", output_key="draft")
review_loop = LoopAgent(name="review_loop", sub_agents=[critic, reviser], max_iterations=3)
pipeline = SequentialAgent(name="pipeline", sub_agents=[draft, review_loop])
lookup = LlmAgent(name="lookup", model=BudgetFakeLlm(tool="lookup_fleet", tool_args={"vehicle_id": "V1"},
                                                     reply="V1 is in service"),
                  instruction="Look a vehicle up.", tools=[lookup_fleet])
root_agent = LlmAgent(name="coordinator", model=BudgetFakeLlm(transfer="pipeline"),
                      instruction="Send a question about the fleet to the pipeline, a vehicle id to lookup.",
                      sub_agents=[pipeline, lookup])


async def main():
    runner = InMemoryRunner(agent=root_agent, app_name="fleet")
    for line in (pathlib.Path(__file__).parent / "inputs.jsonl").read_text().splitlines():
        text = json.loads(line)["input"]
        session = await runner.session_service.create_session(app_name="fleet", user_id="demo")
        async for event in runner.run_async(user_id="demo", session_id=session.id,
                                            new_message=types.Content(role="user", parts=[types.Part(text=text)])):
            if event.is_final_response() and event.content and event.content.parts:
                print(event.author, ":", (event.content.parts[0].text or "")[:60])


if __name__ == "__main__":
    asyncio.run(main())
