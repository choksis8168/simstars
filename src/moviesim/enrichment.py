"""Hidden enrichment: guarantees combustible material exists even when the
user's characters/world are thin, without the user ever seeing it happen.

Two jobs:
  1. Per character: invent a secret, a wound, a hidden goal that conflicts
     with at least one other character's want, and any relevant
     relationship seeds.
  2. Per world: if the user's description doesn't already force contact
     (a bounded space / ticking clock / scarce resource), invent one.

Both run once, at session-creation time, and are never echoed back to the
user (see models.Character docstring for the hidden-at-creation-only rule).
"""

from __future__ import annotations

from moviesim.config import ENRICHMENT_MODEL
from moviesim.llm import call_structured
from moviesim.models import Character

_WORLD_SYSTEM = """You design the hidden dramatic infrastructure for a movie \
simulation. You never write plot or dialogue yourself - you only invent the \
pressure that will force autonomous characters into conflict with each \
other. Keep everything grounded in interpersonal/emotional stakes: secrets, \
betrayal, jealousy, moral compromise, competing loyalties. Never gratuitous \
violence or hate content."""

_CHARACTER_SYSTEM = """You are the hidden writers' room for a movie \
simulation. Given a character's public description and the rest of the \
cast, invent what the user did NOT specify: a secret they're hiding, an old \
wound, and a hidden goal that puts them into conflict with at least one \
other named character in the cast. This will never be shown to the user \
before the movie runs - your only job is to guarantee real dramatic fuel. \
Keep it interpersonal/emotional, not violent."""


def enrich_world(world_description: str, locations: list[str]) -> str | None:
    """Returns a forcing mechanic, or None if the user's world already has one."""
    result = call_structured(
        model=ENRICHMENT_MODEL,
        system=_WORLD_SYSTEM,
        user=(
            f"World description: {world_description}\n"
            f"Locations: {', '.join(locations)}\n\n"
            "Does this world already force the characters into repeated contact "
            "(a bounded space, a ticking clock, a scarce resource, a reason no one "
            "can just leave)? If yes, return has_forcing_mechanic=true and briefly "
            "restate it. If no, return has_forcing_mechanic=false and invent one "
            "that fits the setting."
        ),
        tool_name="set_forcing_mechanic",
        tool_description="Report whether the world already forces contact, and supply the mechanic.",
        input_schema={
            "type": "object",
            "properties": {
                "has_forcing_mechanic": {"type": "boolean"},
                "forcing_mechanic": {
                    "type": "string",
                    "description": "One or two sentences describing the mechanic that forces contact.",
                },
            },
            "required": ["has_forcing_mechanic", "forcing_mechanic"],
        },
    )
    return result["forcing_mechanic"]


def enrich_character(character: Character, cast: list[Character]) -> None:
    """Fills in character.secret / wound / hidden_goal / relationship_seeds in place."""
    other_names = [c.name for c in cast if c.id != character.id]
    result = call_structured(
        model=ENRICHMENT_MODEL,
        system=_CHARACTER_SYSTEM,
        user=(
            f"Character to enrich: {character.name} ({character.role})\n"
            f"Public description: {character.traits}\n"
            f"Starts at: {character.starting_location}\n"
            f"Rest of the cast: {', '.join(other_names) or '(none)'}\n\n"
            "Invent their secret, wound, hidden goal, and relationship seeds."
        ),
        tool_name="set_enrichment",
        tool_description="Report the character's hidden dramatic material.",
        input_schema={
            "type": "object",
            "properties": {
                "secret": {"type": "string", "description": "Something they're actively hiding."},
                "wound": {"type": "string", "description": "A past hurt that shapes how they react under pressure."},
                "hidden_goal": {
                    "type": "string",
                    "description": "What they actually want in this story, which conflicts with at least one other character's want.",
                },
                "relationship_seeds": {
                    "type": "string",
                    "description": "Pre-existing ties to other named cast members, if any.",
                },
            },
            "required": ["secret", "wound", "hidden_goal", "relationship_seeds"],
        },
    )
    character.secret = result["secret"]
    character.wound = result["wound"]
    character.hidden_goal = result["hidden_goal"]
    character.relationship_seeds = result["relationship_seeds"]


def enrich_session(world_description: str, locations: list[str], cast: list[Character]) -> str:
    """Enriches the whole cast in place and returns the world's forcing mechanic."""
    forcing_mechanic = enrich_world(world_description, locations)
    for character in cast:
        enrich_character(character, cast)
    return forcing_mechanic
