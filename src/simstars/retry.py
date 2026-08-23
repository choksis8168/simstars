"""Generic retry-with-backoff for a synchronous callable. Shared by llm.py
(Anthropic) and production.py (ElevenLabs) - both call external APIs from
inside a worker thread (see the asyncio.to_thread usage in each), so a
blocking time.sleep() here only holds up that one thread, never the event
loop. Pulled out of production.py (which had this exact logic already, just
for ElevenLabs only) so a single transient blip on an Anthropic call can't
kill an entire multi-hundred-call run the same way one already couldn't for
ElevenLabs - see docs/design.md verification notes.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any transient API failure retries
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
    assert last_error is not None
    raise last_error
