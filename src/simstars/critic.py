"""Critic pass: because generation isn't live, it can afford to grade its
own work before spending anything on audio. `evaluate()` scores a finished
transcript against story-shape criteria; the caller (pipeline.py) rerolls
the whole simulation on failure, bounded by MAX_CRITIC_RETRIES, then ships
the best-scoring attempt. `compare_previews()` is the finer-grained sibling
used by the branching lookahead in simulation.py: rather than judging a
finished story after the fact, it picks the most dramatically promising of
several short candidate continuations from the same point, before any of
them is committed to.

Includes an audio-specific check: only `dialogue` events are ever voiced in
production (see production.py) - action/director-injected events are
unspoken stage directions. A transcript can look complete on the page while
its key reveals only ever land in unvoiced action text. Both functions are
given the dialogue-only track alongside the full transcript specifically to
catch that failure mode, rather than relying solely on the generation-time
prompt nudges (docs/design.md verification notes: prompt-only nudging was
tried first and was not reliable enough on its own).
"""

from __future__ import annotations

from simstars.config import CRITIC_MODEL
from simstars.llm import call_structured
from simstars.models import EndReason, Event, EventType

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

_COMPARE_SYSTEM = """You are a story editor comparing several short candidate \
continuations of an in-progress autonomous drama simulation, for an \
AUDIO-ONLY production - only dialogue is ever voiced, so judge each \
candidate primarily on what it actually says out loud, not on unspoken \
action description. Pick whichever candidate most effectively advances real \
dramatic conflict: raises tension, pursues a hidden goal, forces something \
into the open, deepens a rift - rather than staying comfortable, marking \
time with routine business, or resolving too easily. Be strict: if every \
candidate is equally flat, say so."""


def _full_log(events: list[Event]) -> str:
    return "\n".join(f"[{e.index}] ({e.location}) {e.actor}: {e.content}" for e in events)


def _dialogue_only_log(transcript: list[Event]) -> str:
    lines = [f"[{e.index}] {e.actor}: {e.content}" for e in transcript if e.type == EventType.DIALOGUE]
    return "\n".join(lines) if lines else "(no dialogue at all)"


def evaluate(transcript: list[Event], end_reason: EndReason) -> dict:
    """Returns {"passes": bool, "reasoning": str, ...}."""
    full_log = _full_log(transcript)
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


def compare_previews(context_events: list[Event], previews: list[list[Event]]) -> dict:
    """Picks the most dramatically promising of several short candidate
    continuations from the same point in the story. One comparative call
    across all candidates, rather than independent absolute scores per
    candidate - cheaper and more reliable than calibrating scores that then
    have to be compared against each other.

    Returns {"best_index": int, "still_flat": bool, "reasoning": str}.
    `still_flat` is true if even the best candidate fails to meaningfully
    advance the story - the caller uses this to trigger one more round of
    previews with this `reasoning` fed back to the director as feedback.
    """
    context_log = _full_log(context_events) if context_events else "(nothing has happened yet)"
    blocks = []
    for i, preview in enumerate(previews):
        blocks.append(
            f"--- Candidate {i} ---\n"
            f"Full:\n{_full_log(preview) or '(nothing happened)'}\n"
            f"Dialogue only (what the audience actually hears):\n{_dialogue_only_log(preview)}"
        )
    user = (
        f"Story so far:\n{context_log}\n\n"
        f"Here are {len(previews)} candidate continuations from this exact point, each independently "
        "generated - pick whichever best advances real dramatic conflict.\n\n" + "\n\n".join(blocks)
    )
    return call_structured(
        model=CRITIC_MODEL,
        system=_COMPARE_SYSTEM,
        user=user,
        tool_name="pick_best_continuation",
        tool_description="Pick which candidate continuation most advances the story, and flag if even the best one is still flat.",
        max_tokens=1024,
        input_schema={
            "type": "object",
            "properties": {
                "best_index": {"type": "integer", "description": "0-based index of the best candidate."},
                "still_flat": {
                    "type": "boolean",
                    "description": "True if even the best candidate fails to meaningfully advance conflict.",
                },
                "reasoning": {"type": "string"},
            },
            "required": ["best_index", "still_flat", "reasoning"],
        },
    )
