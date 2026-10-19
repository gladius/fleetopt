"""The ADK fixture's fake model: no network, no cost, real-looking token usage.

One instance an agent, each with a script: hand the work to another agent, call a tool, or
answer. After a tool or a hand-off answered, it answers.
"""

import asyncio

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class BudgetFakeLlm(BaseLlm):
    model: str = ""  # no name: nothing to bill, as a fake has
    reply: str = "ok"
    transfer: str | None = None       # hand the request to this agent first
    tool: str | None = None           # call this tool first
    tool_args: dict = {}
    output_tokens: int = 50
    latency_s: float = 0.02

    @classmethod
    def supported_models(cls):
        return []

    async def generate_content_async(self, llm_request, stream=False):
        await asyncio.sleep(self.latency_s)
        contents = llm_request.contents or []
        chars = len(str(llm_request.config.system_instruction or "")) + sum(len(str(c)) for c in contents)
        last = contents[-1].parts if contents and contents[-1].parts else []
        answered = any(p.function_response for p in last)
        if not answered and self.transfer:
            part = types.Part(function_call=types.FunctionCall(name="transfer_to_agent", args={"agent_name": self.transfer}))
        elif not answered and self.tool:
            part = types.Part(function_call=types.FunctionCall(name=self.tool, args=dict(self.tool_args)))
        else:
            part = types.Part(text=self.reply)
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100 + chars // 4, candidates_token_count=self.output_tokens,
                total_token_count=100 + chars // 4 + self.output_tokens),
        )
