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
MIN_TURN_BUDGET = 15
MAX_TURN_BUDGET = 25
MAX_CRITIC_RETRIES = 2
DIRECTOR_WRAP_UP_WINDOW = 4  # turns remaining before the director is told to steer toward resolution

DATA_ROOT = Path(os.environ.get("MOVIESIM_DATA_ROOT", Path.cwd() / "sessions"))
DB_PATH = DATA_ROOT / "moviesim.db"


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
