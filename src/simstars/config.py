"""Environment/config loading and scope guardrails.

Guardrail values live here (not scattered through the engine) so they're
easy to find and tune: see docs/design.md "Scope guardrails".
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# Models: Haiku for the many per-turn character calls (speed over depth),
# Sonnet for director/critic/enrichment (fewer calls, needs judgment).
CHARACTER_MODEL = "claude-haiku-4-5-20251001"
DIRECTOR_MODEL = "claude-sonnet-5"
CRITIC_MODEL = "claude-sonnet-5"
ENRICHMENT_MODEL = "claude-sonnet-5"

# Scope guardrails (cost/complexity control)
MIN_CHARACTERS = 3
MAX_CHARACTERS = 5
MIN_LOCATIONS = 2
MAX_LOCATIONS = 4
MIN_TURN_BUDGET = 24
MAX_TURN_BUDGET = 38
MAX_CRITIC_RETRIES = 2
DIRECTOR_WRAP_UP_WINDOW = 7  # turns remaining before the director is told to steer toward resolution

# Live-tuning note (2026-08-19): first real run with 15-25 turns and a
# WRAP_UP_WINDOW of 4 hit turn_budget on all 3 critic attempts, cutting off
# mid-climax every time - a good scene never got the runway to resolve.
# Widened both in response; see docs/design.md verification notes.

# Branching lookahead (story-quality variance fix - see plan follow-on
# section). The turn budget is grouped into segments; at each segment
# boundary BRANCH_FACTOR short previews are generated in parallel and the
# most dramatically promising one is committed, rather than committing to
# one linear path and only judging it after the fact. Previewing only
# PREVIEW_LENGTH turns (not the full segment) before comparing keeps this
# at ~1.7x baseline generation cost instead of ~3x for full-segment
# branching - see docs/design.md follow-on plan for the cost comparison.
SEGMENT_LENGTH = 6
BRANCH_FACTOR = 3
PREVIEW_LENGTH = 2
MAX_SEGMENT_ROUNDS = 2  # re-preview attempts per segment if even the best candidate is still flat

# Live-tuning note (2026-08-22): a real `play` run hit ElevenLabs 429
# concurrent_limit_exceeded ("maximum of 3 concurrent requests") - a scene
# with more than 3 dialogue lines/SFX cues fired that many TTS/SFX calls at
# once with no cap. Kept below the account's actual limit (3), not equal to
# it, since this only bounds concurrency *within* one produce() call - two
# `play` jobs running in different threads at the same time (see jobs.py)
# would each open their own budget and could still combine to exceed 3.
# Fine for the current single-local-user scope; a real limitation if that
# scope ever changes.
MAX_CONCURRENT_ELEVENLABS_CALLS = 2

DATA_ROOT = Path(os.environ.get("SIMSTARS_DATA_ROOT", Path.cwd() / "sessions"))
DB_PATH = DATA_ROOT / "simstars.db"


def require_anthropic_key() -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ANTHROPIC_API_KEY


def require_elevenlabs_key() -> str:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ELEVENLABS_API_KEY
