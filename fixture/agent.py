"""A throwaway LangGraph agent used only to prove the capture harness works.

Deliberately shaped like a real one: a plan step, a research loop that re-sends a
growing context every iteration, and a summarize step. The fake chat model emits
usage_metadata so the whole thing runs offline with no API cost.
"""

from typing import TypedDict

from fake_model import BudgetFakeChat
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    topic: str
    notes: list[str]
    rounds: int
    summary: str


planner = BudgetFakeChat(reply="1. survey 2. dig 3. write", output_tokens=40)
researcher = BudgetFakeChat(reply="found something " * 40, output_tokens=320, latency_s=0.06)
writer = BudgetFakeChat(reply="summary of findings", output_tokens=120)

MAX_ROUNDS = 3


def plan(state: State) -> dict:
    reply = planner.invoke([HumanMessage(content=f"Plan research on {state['topic']}")])
    return {"notes": [reply.content], "rounds": 0}


def research(state: State) -> dict:
    # The hotspot: every round re-sends all prior notes instead of a summary.
    context = "\n".join(state["notes"])
    reply = researcher.invoke([HumanMessage(content=f"Given:\n{context}\nResearch further.")])
    return {"notes": state["notes"] + [reply.content], "rounds": state["rounds"] + 1}


def summarize(state: State) -> dict:
    context = "\n".join(state["notes"])
    reply = writer.invoke([HumanMessage(content=f"Summarize:\n{context}")])
    return {"summary": reply.content}


def keep_going(state: State) -> str:
    return "research" if state["rounds"] < MAX_ROUNDS else "summarize"


builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("research", research)
builder.add_node("summarize", summarize)
builder.add_edge(START, "plan")
builder.add_edge("plan", "research")
builder.add_conditional_edges("research", keep_going, ["research", "summarize"])
builder.add_edge("summarize", END)

# Compiled at import time on purpose - the harness must instrument before this.
graph = builder.compile()


def main() -> None:
    for topic in ("battery degradation", "route optimization"):
        result = graph.invoke({"topic": topic, "notes": [], "rounds": 0, "summary": ""})
        print(f"{topic}: {result['summary']}")


if __name__ == "__main__":
    main()
