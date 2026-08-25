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

import contextvars
import threading
from typing import Any, Union

from anthropic import Anthropic

from simstars.config import (
    ANTHROPIC_PRICING_PER_MILLION_TOKENS,
    CACHE_READ_PRICE_MULTIPLIER,
    CACHE_WRITE_PRICE_MULTIPLIER,
    require_anthropic_key,
)
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


# --- Usage tracking (see docs/design.md verification notes: "a live run
# was observed making several hundred calls" - this turns that anecdote
# into a real per-Run number). One generate() call can fan out across many
# threads (branching previews via asyncio.to_thread - see simulation.py)
# and several generate()/play() jobs can run concurrently in different
# worker threads (see jobs.py) - a plain module-level counter would let
# concurrent runs' usage bleed into each other. A ContextVar gives each
# top-level reset_usage_tracking() call its own accumulator dict; nested
# calls on the same call tree (including ones asyncio.to_thread spawns,
# which copies the calling context) mutate that same dict object in place
# rather than rebinding the var, which is what keeps the isolation intact
# across threads while still letting them all report into one place.

_usage_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar("simstars_llm_usage", default=None)
_usage_lock = threading.Lock()  # protects mutating whatever dict _usage_ctx currently points to

UsageSnapshot = dict[str, dict[str, int]]


def reset_usage_tracking() -> None:
    """Starts a fresh, isolated usage accumulator for the current call tree
    - call once at the top of whatever you want a cost total for (see
    pipeline.generate()). Calls made before this is ever called in a given
    thread/task simply aren't tracked - fine, since untracked callers
    (enrichment at session-creation time, voice casting) aren't part of
    what this is meant to total.
    """
    _usage_ctx.set({})


def usage_snapshot() -> UsageSnapshot:
    bucket = _usage_ctx.get()
    if bucket is None:
        return {}
    with _usage_lock:
        return {model: dict(counts) for model, counts in bucket.items()}


def estimated_cost_usd(snapshot: UsageSnapshot | None = None) -> float:
    """Rough estimate from config.py's published per-model rates and the
    standard Anthropic cache-token multipliers - not a substitute for the
    real bill (exact discounts/multipliers can change), but far more useful
    than guessing. A model with no price entry is skipped, not raised on.
    """
    snapshot = snapshot if snapshot is not None else usage_snapshot()
    total = 0.0
    for model, counts in snapshot.items():
        rates = ANTHROPIC_PRICING_PER_MILLION_TOKENS.get(model)
        if rates is None:
            continue
        total += counts["input_tokens"] * rates["input"] / 1_000_000
        total += counts["output_tokens"] * rates["output"] / 1_000_000
        total += counts["cache_creation_input_tokens"] * rates["input"] * CACHE_WRITE_PRICE_MULTIPLIER / 1_000_000
        total += counts["cache_read_input_tokens"] * rates["input"] * CACHE_READ_PRICE_MULTIPLIER / 1_000_000
    return total


def _record_usage(model: str, usage: Any) -> None:
    bucket = _usage_ctx.get()
    if bucket is None or usage is None:
        return
    with _usage_lock:
        counts = bucket.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        )
        counts["calls"] += 1
        counts["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        counts["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        counts["cache_creation_input_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        counts["cache_read_input_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0


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
    _record_usage(model, getattr(response, "usage", None))
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise RuntimeError(f"Model did not return the expected '{tool_name}' tool call")
