"""Replay a recorded run (session.jsonl) as the model, with fleetopt's tools live.

A recording is every SDK message of one run. Replayed, the model's turns come from the
tape and every fleetopt tool call in them is made for real against the project, so what
the tools do is tested without a model and without spend; the built-in tools' results
(Read, Grep) come from the tape, since the files are the same. The final report is the
tape's, so a replay checks the mechanics under a real expert's run, not the expert.
"""

import json
import pathlib

import claude_agent_sdk as sdk

from fleetopt import tools

BLOCKS = {"text": sdk.TextBlock, "thinking": sdk.ThinkingBlock, "id": sdk.ToolUseBlock, "tool_use_id": sdk.ToolResultBlock}


def _block(d):
    for key, cls in BLOCKS.items():  # by the field only that kind has
        if key in d and (key != "id" or "name" in d):
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    return None


def load(path):
    """The messages on a tape, as SDK objects."""
    out = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        kind = d.pop("type")
        cls = getattr(sdk, kind)
        if isinstance(d.get("content"), list):
            d["content"] = [b for b in map(_block, d["content"]) if b]
        out.append(cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__}))
    return out


def player(path, seen=None):
    """A `query` that plays the tape: each fleetopt tool call is made live and its real result
    returned; other results are the tape's. `seen`, when given, collects (tool, live, taped)."""
    messages = load(path)

    async def query(prompt, options):
        live = {}  # tool_use_id -> live result text
        for message in messages:
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.ToolUseBlock) and block.name.startswith("mcp__fleetopt__"):
                        handler = getattr(tools, block.name.removeprefix("mcp__fleetopt__")).handler
                        result = await handler(block.input)
                        live[block.id] = " ".join(c.get("text", "") for c in result.get("content", []))
                yield message
            elif isinstance(message, sdk.UserMessage) and isinstance(message.content, list):
                blocks = []
                for block in message.content:
                    if isinstance(block, sdk.ToolResultBlock) and block.tool_use_id in live:
                        if seen is not None:
                            seen.append((block.tool_use_id, live[block.tool_use_id], block.content))
                        block = sdk.ToolResultBlock(tool_use_id=block.tool_use_id, content=live[block.tool_use_id])
                    blocks.append(block)
                yield sdk.UserMessage(content=blocks)
            else:
                yield message

    return query
