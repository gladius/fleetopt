"""Target that already has its own callback handler wired in."""
from langchain_core.callbacks.base import BaseCallbackHandler
from agent import graph

class TheirHandler(BaseCallbackHandler):
    def __init__(self): self.n = 0
    def on_llm_end(self, *a, **k): self.n += 1

theirs = TheirHandler()
graph.invoke({"topic":"x","notes":[],"rounds":0,"summary":""}, config={"callbacks":[theirs]})
print(f"[their-handler] saw {theirs.n} llm calls")
