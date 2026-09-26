"""Second fixture: structurally over-built on purpose, cheap to run.

Three planted smells for the architecture review, none of them a cost defect:
- a router with three branches, of which one is ever taken;
- a "supervisor" that consults a model before every step and then dispatches three
  workers in the same fixed order every time - a pipeline dressed as a supervisor;
- a reflection loop that runs three rounds and never changes the draft.
Same fake model as agent.py, so it runs offline at no cost:  python supervisor.py
"""

from typing import TypedDict

from fake_model import BudgetFakeChat
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    route: str
    findings: list[str]
    draft: str
    rounds: int


router = BudgetFakeChat(reply="technical", output_tokens=2)
supervisor_llm = BudgetFakeChat(reply="proceed", output_tokens=4)
worker_llm = BudgetFakeChat(reply="finding " * 30, output_tokens=90, latency_s=0.03)
writer = BudgetFakeChat(reply="Draft answer covering the findings.", output_tokens=60)
critic = BudgetFakeChat(reply="The draft is complete and accurate. No changes.", output_tokens=12)

ORDER = ["worker_a", "worker_b", "worker_c", "draft"]
MAX_REFLECTIONS = 3


def route(state: State) -> dict:
    reply = router.invoke([HumanMessage(content=f"Classify as billing, technical or other: {state['question']}")])
    return {"route": reply.content.strip()}


def by_route(state: State) -> str:
    return state["route"]  # billing | technical | other - the fake always says technical


def billing(state: State) -> dict:
    return {"findings": ["billing lookup"]}


def other(state: State) -> dict:
    return {"findings": ["general reply"]}


def technical(state: State) -> dict:
    return {"findings": []}


def supervisor(state: State) -> dict:
    # Asks the model what to do next, then ignores the answer: the order is fixed.
    supervisor_llm.invoke([HumanMessage(content=f"Findings so far: {len(state['findings'])}. Which worker next?")])
    return {}


def dispatch(state: State) -> str:
    return ORDER[len(state["findings"])]


def _worker(name: str):
    def node(state: State) -> dict:
        reply = worker_llm.invoke([HumanMessage(content=f"{name}: investigate {state['question']}")])
        return {"findings": state["findings"] + [reply.content]}
    node.__name__ = name
    return node


def draft(state: State) -> dict:
    reply = writer.invoke([HumanMessage(content="Write an answer from:\n" + "\n".join(state["findings"]))])
    return {"draft": reply.content, "rounds": 0}


def reflect(state: State) -> dict:
    # A critic that never asks for a change, three times per run.
    critic.invoke([HumanMessage(content=f"Critique this draft:\n{state['draft']}")])
    return {"draft": state["draft"], "rounds": state["rounds"] + 1}


def keep_reflecting(state: State) -> str:
    return "reflect" if state["rounds"] < MAX_REFLECTIONS else END


builder = StateGraph(State)
for name, fn in [("route", route), ("billing", billing), ("technical", technical), ("other", other),
                 ("supervisor", supervisor), ("worker_a", _worker("worker_a")),
                 ("worker_b", _worker("worker_b")), ("worker_c", _worker("worker_c")),
                 ("draft", draft), ("reflect", reflect)]:
    builder.add_node(name, fn)
builder.add_edge(START, "route")
builder.add_conditional_edges("route", by_route, ["billing", "technical", "other"])
builder.add_edge("billing", END)
builder.add_edge("other", END)
builder.add_edge("technical", "supervisor")
builder.add_conditional_edges("supervisor", dispatch, ORDER)
for w in ("worker_a", "worker_b", "worker_c"):
    builder.add_edge(w, "supervisor")
builder.add_edge("draft", "reflect")
builder.add_conditional_edges("reflect", keep_reflecting, ["reflect", END])
graph = builder.compile()


def main() -> None:
    for question in ("VPN drops every hour", "invoice shows a double charge"):
        result = graph.invoke({"question": question, "route": "", "findings": [], "draft": "", "rounds": 0})
        print(f"{question}: {result['draft']}")


if __name__ == "__main__":
    main()
