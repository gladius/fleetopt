"""The fixtures' fake chat model: no network, no cost, real-looking token usage."""

import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class BudgetFakeChat(BaseChatModel):
    """Fake model that reports token usage, so capture has something to read."""

    reply: str = "ok"
    input_tokens: int = 100
    output_tokens: int = 50
    latency_s: float = 0.02

    @property
    def _llm_type(self) -> str:
        return "budget-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.latency_s)
        # Grow with the prompt the way a real model's input tokens would.
        prompt_chars = sum(len(str(m.content)) for m in messages)
        message = AIMessage(
            content=self.reply,
            usage_metadata={
                "input_tokens": self.input_tokens + prompt_chars // 4,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + prompt_chars // 4 + self.output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
