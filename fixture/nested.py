"""Third fixture: two experts built alike behind a router, each a sub-graph with a `model`
and a `tools` node.

A request about numbers goes to the math expert only, and an expert uses its tools only when
asked to check: so nodes of the same name run in one sub-graph and never in the other.
Same fake model as agent.py, so it runs offline at no cost.
"""

from typing import TypedDict

from fake_model import BudgetFakeChat
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    answer: str


llm = BudgetFakeChat(reply="an answer", output_tokens=20)


def math_model(state: State) -> dict:
    reply = llm.invoke([HumanMessage(content=f"Work out: {state['question']}")])
    return {"answer": reply.content}


def math_tools(state: State) -> dict:
    return {"answer": state["answer"] + " (checked with the calculator)"}


def research_model(state: State) -> dict:
    reply = llm.invoke([HumanMessage(content=f"Look up: {state['question']}")])
    return {"answer": reply.content}


def research_tools(state: State) -> dict:
    return {"answer": state["answer"] + " (checked against the archive)"}


def wants_a_check(state: State) -> str:
    return "tools" if "check" in state["question"] else END


def expert(model, tools):
    builder = StateGraph(State)
    builder.add_node("model", model)
    builder.add_node("tools", tools)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", wants_a_check, ["tools", END])
    builder.add_edge("tools", END)
    return builder.compile()


def by_subject(state: State) -> str:
    return "math" if any(c.isdigit() for c in state["question"]) else "research"


builder = StateGraph(State)
builder.add_node("math", expert(math_model, math_tools))
builder.add_node("research", expert(research_model, research_tools))
builder.add_conditional_edges(START, by_subject, ["math", "research"])
builder.add_edge("math", END)
builder.add_edge("research", END)
graph = builder.compile()


def main() -> None:
    for question in ("sum of 2 and 3", "check 12 times 12", "the history of rail", "check the history of rail"):
        print(f"{question}: {graph.invoke({'question': question, 'answer': ''})['answer']}")


if __name__ == "__main__":
    main()
