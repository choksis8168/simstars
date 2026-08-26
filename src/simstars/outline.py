"""Pre-generation story outline: one Sonnet call, made before simulate()
runs a single turn, that sketches a rough dramatic arc from the cast's
hidden material (secrets/wounds/goals/relationship seeds) - see
docs/design.md "Simulation engine". Branching lookahead and the critic
pass both react to flatness *after* something was generated; this gives
the director something to steer *toward* from the very first turn instead
of discovering a good shape only through trial.

This is guidance, not a script: fed into DirectorAgent's prompt alongside
the world/forcing-mechanic block (see simulation.py), never shown to
characters. Characters improvise every actual line regardless - nothing
here is ever spoken verbatim, and the director is explicitly told it's
allowed to deviate. Keeping this director-only (not character-visible)
matches the same omniscience boundary that already governs hidden
secrets/wounds/goals - see simulation.py's module docstring.
"""

from __future__ import annotations

from simstars.config import ENRICHMENT_MODEL as OUTLINE_MODEL  # same tier of judgment call as enrichment
from simstars.llm import call_structured
from simstars.models import Character, Session

_SYSTEM = """You are a story editor sketching a rough dramatic arc for an \
autonomous multi-agent drama simulation, before any of it has been \
generated. This is guidance for a director agent - not a script, and \
never shown to the characters themselves, who will still improvise every \
actual line. Your job is to find the most promising shape given this \
specific cast's hidden material: whose secret is most likely to surface \
and how, whose goals actually collide, what forces a confrontation, and \
how it could plausibly resolve. Be specific to this cast and world, not \
generic ("someone's secret comes out") - name who and what. 3-5 short \
beats, each one sentence. Do not write dialogue or narrate outcomes as \
certainties - a director following this should still discover exactly how \
it plays out."""


def generate_outline(session: Session, characters: list[Character], producer_note: str | None = None) -> str:
    """Called once per generate() call (see pipeline.py), not once per
    retry attempt - the intended shape shouldn't change just because an
    earlier attempt was rerolled; the director naturally arrives at a
    different concrete execution against the same guidance each attempt.
    """
    cast_summary = "\n".join(
        f"- {c.name} ({c.role}, {c.traits}): secret={c.secret}; wound={c.wound}; "
        f"wants={c.hidden_goal}; history with others={c.relationship_seeds}"
        for c in characters
    )
    note = f"\nProducer's note for this run: {producer_note}" if producer_note else ""
    user = (
        f"World: {session.world_description}\n"
        f"Forcing mechanic: {session.forcing_mechanic}\n"
        f"Locations: {', '.join(session.location_list())}\n\n"
        f"Cast (full hidden material):\n{cast_summary}{note}\n\n"
        "Sketch the dramatic arc most likely to emerge well from this specific cast."
    )
    result = call_structured(
        model=OUTLINE_MODEL,
        system=_SYSTEM,
        user=user,
        tool_name="sketch_outline",
        tool_description="Sketch a rough dramatic arc to guide the simulation's director.",
        max_tokens=768,
        input_schema={
            "type": "object",
            "properties": {
                "beats": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 short beats describing the intended shape, in order.",
                }
            },
            "required": ["beats"],
        },
    )
    return "\n".join(f"- {beat}" for beat in result["beats"])
