"""Library entrypoints: new_session / generate / play. The CLI (cli.py) is
a thin caller of exactly these three functions and nothing else - see
docs/design.md "CLI (Typer)" for why that separation matters for a future
web UI.
"""

from __future__ import annotations

import asyncio
import json
import random

from moviesim import production
from moviesim.config import (
    MAX_CHARACTERS,
    MAX_CRITIC_RETRIES,
    MAX_LOCATIONS,
    MAX_TURN_BUDGET,
    MIN_CHARACTERS,
    MIN_LOCATIONS,
    MIN_TURN_BUDGET,
)
from moviesim.critic import evaluate
from moviesim.db import audio_dir, get_session, run_dir
from moviesim.enrichment import enrich_session
from moviesim.models import Character, EndReason, Event, Run, Screenplay, Session
from moviesim.screenplay import build_screenplay
from moviesim.simulation import simulate


class CharacterSpec:
    def __init__(self, name: str, role: str, traits: str, starting_location: str):
        self.name = name
        self.role = role
        self.traits = traits
        self.starting_location = starting_location


def new_session(world_description: str, locations: list[str], cast_specs: list[CharacterSpec]) -> Session:
    if not (MIN_CHARACTERS <= len(cast_specs) <= MAX_CHARACTERS):
        raise ValueError(f"Cast must have {MIN_CHARACTERS}-{MAX_CHARACTERS} characters, got {len(cast_specs)}.")
    if not (MIN_LOCATIONS <= len(locations) <= MAX_LOCATIONS):
        raise ValueError(f"World must have {MIN_LOCATIONS}-{MAX_LOCATIONS} locations, got {len(locations)}.")
    for spec in cast_specs:
        if spec.starting_location not in locations:
            raise ValueError(f"{spec.name}'s starting location '{spec.starting_location}' isn't one of {locations}.")

    session = Session(world_description=world_description, locations=", ".join(locations))
    characters = [
        Character(
            session_id=session.id,
            name=s.name,
            role=s.role,
            traits=s.traits,
            starting_location=s.starting_location,
        )
        for s in cast_specs
    ]

    forcing_mechanic = enrich_session(world_description, locations, characters)
    session.forcing_mechanic = forcing_mechanic
    production.cast_voices(characters)

    with get_session() as db:
        db.add(session)
        for c in characters:
            db.add(c)
        db.commit()
        db.refresh(session)
        for c in characters:
            db.refresh(c)
        # Force the `characters` relationship to load *while the db session
        # is still open* - otherwise it's an unloaded lazy relationship on a
        # detached instance, and any access after this `with` block exits
        # raises DetachedInstanceError. Accessing it here caches the loaded
        # collection on the instance so it's safe to read afterward.
        len(session.characters)

    return session


def _load_session(session_id: str) -> tuple[Session, list[Character]]:
    with get_session() as db:
        session = db.get(Session, session_id)
        if session is None:
            raise ValueError(f"No session '{session_id}'.")
        characters = list(session.characters)
        # detach values we need after the db session closes
        session.locations
        return session, characters


def generate(session_id: str, producer_note: str | None = None) -> tuple[Run, list[Event], Screenplay]:
    """GENERATE phase only: simulate -> critic (with reroll) -> screenplay.
    Persists the Run with transcript/screenplay filled in, no audio yet.
    This is what `movie script` calls - no ElevenLabs cost.
    """
    session, characters = _load_session(session_id)

    best_events: list[Event] | None = None
    best_end_reason: EndReason | None = None
    attempts = 0

    while True:
        turn_budget = random.randint(MIN_TURN_BUDGET, MAX_TURN_BUDGET)
        events, end_reason = simulate(session, characters, turn_budget, producer_note)
        attempts += 1
        grade = evaluate(events, end_reason)

        if best_events is None:
            best_events, best_end_reason = events, end_reason

        if grade["passes"] or attempts > MAX_CRITIC_RETRIES:
            best_events, best_end_reason = events, end_reason
            break

    screenplay = build_screenplay(best_events)

    run = Run(
        session_id=session_id,
        producer_note=producer_note,
        end_reason=best_end_reason,
        critic_attempts=attempts,
        transcript_json=json.dumps([e.model_dump(mode="json") for e in best_events]),
        screenplay_json=screenplay.model_dump_json(),
    )
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    return run, best_events, screenplay


def play(session_id: str, producer_note: str | None = None) -> Run:
    """Full generate -> produce -> release."""
    session, characters = _load_session(session_id)
    run, _events, screenplay = generate(session_id, producer_note)

    voice_by_name = {c.name: c.voice_id for c in characters if c.voice_id}
    out_dir = audio_dir(session_id, run.id)
    final_path = asyncio.run(production.produce(screenplay, voice_by_name, out_dir))

    run.final_audio_path = str(final_path)
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    return run
