"""Critic pass: because generation isn't live, it can afford to grade its
own work before spending anything on audio. Scores the finished transcript
against story-shape criteria; the caller (pipeline.py) rerolls the whole
simulation on failure, bounded by MAX_CRITIC_RETRIES, then ships the best
attempt regardless.

Includes an audio-specific check: only `dialogue` events are ever voiced in
production (see production.py) - action/director-injected events are
unspoken stage directions. A transcript can look complete on the page while
its key reveals only ever land in unvoiced action text. The critic is given
the dialogue-only track alongside the full transcript specifically to catch
that failure mode and reroll on it, rather than relying solely on the
generation-time prompt nudges (docs/design.md verification notes: prompt-only
nudging was tried first and was not reliable enough on its own).
"""

from __future__ import annotations

from moviesim.config import CRITIC_MODEL
from moviesim.llm import call_structured
from moviesim.models import EndReason, Event, EventType

_SYSTEM = """You are a story editor evaluating a transcript produced by an \
autonomous multi-agent drama simulation, for an AUDIO-ONLY production. Judge \
whether it has a real dramatic shape - not writing quality, not whether you \
personally enjoyed it. Be strict: a sequence of pleasant, low-stakes small \
talk with no real conflict should fail, even if it's coherent.

Critically: only dialogue is ever voiced in the final audio. Action and \
director-injected event beats are written stage directions that the \
audience will never hear. You are given both the full transcript and a \
dialogue-only extract - if a key reveal, decision, or turn in the full \
transcript is never actually spoken by a character in the dialogue-only \
extract, the audience will miss it entirely. That is a failure, even if \
the full transcript reads well on the page."""


def _dialogue_only_log(transcript: list[Event]) -> str:
    lines = [f"[{e.index}] {e.actor}: {e.content}" for e in transcript if e.type == EventType.DIALOGUE]
    return "\n".join(lines) if lines else "(no dialogue at all)"


def evaluate(transcript: list[Event], end_reason: EndReason) -> dict:
    """Returns {"passes": bool, "reasoning": str, ...}."""
    full_log = "\n".join(f"[{e.index}] ({e.location}) {e.actor}: {e.content}" for e in transcript)
    dialogue_log = _dialogue_only_log(transcript)
    user = (
        f"End reason: {end_reason.value} "
        f"({'director called cut' if end_reason == EndReason.RESOLVED else 'hit the turn budget with no clean cut'})\n\n"
        f"Full transcript (includes unvoiced action/event beats):\n{full_log}\n\n"
        f"Dialogue-only extract (this is ALL the audience will actually hear):\n{dialogue_log}\n\n"
        "Judge strictly: does the full transcript have a genuine story shape "
        "(real conflict, escalation, a resolution - not just a hard stop)? And "
        "separately: does the dialogue-only extract, on its own, actually "
        "convey that same story to a listener - or does it depend on reveals "
        "that only ever happened in unvoiced action text?"
    )
    return call_structured(
        model=CRITIC_MODEL,
        system=_SYSTEM,
        user=user,
        tool_name="grade_story",
        tool_description="Grade whether the transcript has a genuine dramatic shape that the audio alone will actually convey.",
        max_tokens=1024,
        input_schema={
            "type": "object",
            "properties": {
                "has_real_conflict": {"type": "boolean"},
                "has_escalation": {"type": "boolean"},
                "has_resolution": {"type": "boolean"},
                "dialogue_carries_the_story": {
                    "type": "boolean",
                    "description": "False if any key reveal/decision/turn only ever happened in unvoiced action/event text, not in spoken dialogue.",
                },
                "passes": {
                    "type": "boolean",
                    "description": "True only if all four criteria above are met.",
                },
                "reasoning": {"type": "string"},
            },
            "required": [
                "has_real_conflict",
                "has_escalation",
                "has_resolution",
                "dialogue_carries_the_story",
                "passes",
                "reasoning",
            ],
        },
    )
