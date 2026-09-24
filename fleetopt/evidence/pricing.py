"""Tokens -> dollars.

These rates are a starting point, NOT an authority - list prices move and this
table will rot. Override with a JSON file of the same shape via FLEETOPT_PRICES
rather than editing here, and treat any figure from an unpriced model as missing
data rather than as zero.
"""

import json
import os

# USD per million tokens: (input, output, cache_read, cache_write)
PRICES = {
    # Verified against Anthropic's model table 2026-09-23. cache_read = 0.1x input,
    # cache_write (5-minute TTL) = 1.25x input, except where the docs say otherwise.
    "claude-opus-5": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-8": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-7": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-6": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-5": (5.0, 25.0, 0.50, 6.25),
    "claude-opus-4-1": (15.0, 75.0, 1.50, 18.75),
    "claude-opus-4": (15.0, 75.0, 1.50, 18.75),
    "claude-sonnet-5": (2.0, 10.0, 0.20, 2.50),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-4-5": (3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-4": (3.0, 15.0, 0.30, 3.75),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25),
    "gpt-4o": (2.50, 10.0, 1.25, 0.0),
    "gpt-4o-mini": (0.15, 0.60, 0.075, 0.0),
    # OpenAI, list prices 2026-09-23. Writes are billed as ordinary input except on
    # GPT-5.6+, which charges 1.25x for explicit cache writes.
    "gpt-5": (1.25, 10.0, 0.125, 0.0),
    "gpt-5-mini": (0.25, 2.0, 0.025, 0.0),
    "gpt-5-nano": (0.05, 0.40, 0.005, 0.0),
    "gpt-5.4": (2.50, 15.0, 0.25, 0.0),
    "gpt-5.4-mini": (0.75, 4.50, 0.075, 0.0),
    "gpt-5.4-nano": (0.20, 1.25, 0.02, 0.0),
    "gpt-5.5": (5.0, 30.0, 0.50, 0.0),
    "gpt-5.6-sol": (4.0, 20.0, 0.40, 5.0),
    "gpt-5.6-terra": (2.0, 12.0, 0.20, 2.50),
    "gpt-5.6-luna": (0.20, 1.20, 0.02, 0.25),
    "gpt-4.1": (2.0, 8.0, 0.50, 0.0),
    "gpt-4.1-mini": (0.40, 1.60, 0.10, 0.0),
    "gpt-4.1-nano": (0.10, 0.40, 0.025, 0.0),
    "o3": (2.0, 8.0, 0.50, 0.0),
    "o4-mini": (1.10, 4.40, 0.275, 0.0),
    # Gemini, paid tier, <=200K-context rate, list prices 2026-09-23. Implicit
    # caching bills reads at 0.1x; no write premium. Rows marked * infer the read
    # rate from that 0.1x pattern because the page did not print it - verify.
    "gemini-2.5-pro": (1.25, 10.0, 0.125, 0.0),
    "gemini-2.5-flash": (0.30, 2.50, 0.03, 0.0),        # *
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01, 0.0),   # *
    "gemini-3.1-pro-preview": (2.0, 12.0, 0.20, 0.0),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025, 0.0),  # *
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03, 0.0),   # *
    "gemini-3.5-flash": (0.75, 3.75, 0.075, 0.0),
    "gemini-3.6-flash": (0.75, 3.75, 0.075, 0.0),
    "gemini-3.7-flash": (0.75, 3.75, 0.075, 0.0),
    "gemini-3.8-flash": (0.75, 3.75, 0.075, 0.0),
}


def _table():
    override = os.environ.get("FLEETOPT_PRICES")
    if not override:
        return PRICES
    with open(override) as fh:
        return {k: tuple(v) for k, v in json.load(fh).items()}


def rate(model):
    """Longest-prefix match, so `claude-sonnet-5-20260101` finds `claude-sonnet-5`."""
    if not model:
        return None
    table = _table()
    matches = [k for k in table if model.startswith(k)]
    return table[max(matches, key=len)] if matches else None


def cost(model, input_tokens=0, output_tokens=0, cache_read=0, cache_write=0):
    """USD for one call, or None if we don't have a price for this model.

    None is deliberate: a savings claim built on a silently-zero model is worse
    than no claim at all.
    """
    r = rate(model)
    if r is None:
        return None
    # Cached reads replace ordinary input tokens; they aren't billed twice.
    billed_input = max(input_tokens - cache_read - cache_write, 0)
    return (
        billed_input * r[0] + output_tokens * r[1] + cache_read * r[2] + cache_write * r[3]
    ) / 1_000_000
