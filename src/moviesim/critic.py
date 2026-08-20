"""Critic pass: because generation isn't live, it can afford to grade its
own work before spending anything on audio. Scores the finished transcript
against story-shape criteria; the caller (pipeline.py) rerolls the whole
simulation on failure, bounded by MAX_CRITIC_RETRIES, then ships the best
attempt regardless.
"""

from __future__ import annotations

from moviesim.config import CRITIC_MODEL
from moviesim.llm import call_structured
from moviesim.models import EndReason, Event

_SYSTEM = """You are a story editor evaluating a transcript produced by an \
autonomous multi-agent drama simulation. Judge only whether it has a real \
dramatic shape - not writing quality, not whether you personally enjoyed \
it. Be strict: a sequence of pleasant, low-stakes small talk with no real \
conflict should fail, even if it's coherent."""


def evaluate(transcript: list[Event], end_reason: EndReason) -> dict:
    """Returns {"passes": bool, "reasoning": str}."""
    log = "\n".join(f"[{e.index}] ({e.location}) {e.actor}: {e.content}" for e in transcript)
    user = (
        f"End reason: {end_reason.value} "
        f"({'director called cut' if end_reason == EndReason.RESOLVED else 'hit the turn budget with no clean cut'})\n\n"
        f"Transcript:\n{log}\n\n"
        "Does this have a real story shape: a genuine conflict, escalation, "
        "and a resolution (not just a hard stop)? Judge strictly."
    )
    return call_structured(
        model=CRITIC_MODEL,
        system=_SYSTEM,
        user=user,
        tool_name="grade_story",
        tool_description="Grade whether the transcript has a genuine dramatic shape.",
        input_schema={
            "type": "object",
            "properties": {
                "has_real_conflict": {"type": "boolean"},
                "has_escalation": {"type": "boolean"},
                "has_resolution": {"type": "boolean"},
                "passes": {
                    "type": "boolean",
                    "description": "True only if all three criteria above are met.",
                },
                "reasoning": {"type": "string"},
            },
            "required": ["has_real_conflict", "has_escalation", "has_resolution", "passes", "reasoning"],
        },
    )
