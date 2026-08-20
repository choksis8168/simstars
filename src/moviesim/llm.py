"""Thin wrapper around the Anthropic client for structured (tool-forced)
JSON output. Every agent in this system (enrichment, character, director,
critic) wants a typed result back, not free text to parse — so every call
goes through here with a one-tool schema and forced tool_choice.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from moviesim.config import require_anthropic_key

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=require_anthropic_key())
    return _client


def call_structured(
    *,
    model: str,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Call Claude with a single forced tool, return its parsed input dict."""
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": input_schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise RuntimeError(f"Model did not return the expected '{tool_name}' tool call")
