"""Thin wrapper around the Anthropic client for structured (tool-forced)
JSON output. Every agent in this system (enrichment, character, director,
critic) wants a typed result back, not free text to parse — so every call
goes through here with a one-tool schema and forced tool_choice.

Also the home of prompt-caching support: `system`/`user` accept either a
plain string or a list of pre-built content blocks (see `cached_block`),
so callers can mark a stable prefix as cacheable. This matters a lot here
specifically because agent prompts re-send the whole transcript-so-far on
every single turn with no caching, cost grows faster than linearly as a
run progresses - a live run was observed making several hundred calls (see
docs/design.md), and re-transmitting/re-billing an ever-longer transcript
on each of those calls is the single biggest avoidable cost in that total.

Every call goes through `with_retry` (shared with production.py's
ElevenLabs calls - see retry.py) - a run can make hundreds of these calls
over several minutes, and simulation.py's branching previews run several in
parallel (see its asyncio.gather usage), so without this, one transient
blip on any single call used to kill the entire multi-hundred-call, several-
minutes-long, real-money job. See docs/design.md verification notes.
"""

from __future__ import annotations

from typing import Any, Union

from anthropic import Anthropic

from simstars.config import require_anthropic_key
from simstars.retry import with_retry

_client: Anthropic | None = None

Content = Union[str, list[dict[str, Any]]]


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=require_anthropic_key())
    return _client


def cached_block(text: str) -> dict[str, Any]:
    """A text content block marked as an ephemeral cache breakpoint -
    Anthropic caches everything up to and including this block and reuses
    it on a later call whose prefix matches, so this only pays off once a
    call's actual content is placed *after* the cached block, and only
    once the cumulative prefix is long enough to clear the provider's
    minimum cacheable length (short prompts silently get no benefit, not
    an error - safe to mark early, low-cost turns too).
    """
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def call_structured(
    *,
    model: str,
    system: Content,
    user: Content,
    tool_name: str,
    tool_description: str,
    input_schema: dict[str, Any],
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Call Claude with a single forced tool, return its parsed input dict.
    `system`/`user` may be a plain string or a list of content blocks (use
    `cached_block` for a stable prefix, plain `{"type": "text", "text": ...}`
    for the volatile remainder) - passed straight through to the SDK, which
    accepts both forms for `system` and for a message's `content`.
    """
    client = _get_client()

    def call():
        return client.messages.create(
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

    response = with_retry(call)
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise RuntimeError(f"Model did not return the expected '{tool_name}' tool call")
