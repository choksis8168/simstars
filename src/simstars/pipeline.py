"""Library entrypoints: new_session / generate / play. The CLI (cli.py) is
a thin caller of exactly these three functions and nothing else - see
docs/design.md "CLI (Typer)" for why that separation matters for a future
web UI.
"""

from __future__ import annotations

import asyncio
import json
import random

from sqlmodel import select

from simstars import production
from simstars.config import (
    MAX_CHARACTERS,
    MAX_CRITIC_RETRIES,
    MAX_LOCATIONS,
    MAX_TURN_BUDGET,
    MIN_CHARACTERS,
    MIN_LOCATIONS,
    MIN_TURN_BUDGET,
)
from simstars.critic import evaluate
from simstars.db import audio_dir, get_session, run_dir
from simstars.enrichment import enrich_session
from simstars.llm import estimated_cost_usd, reset_usage_tracking, usage_snapshot
from simstars.models import Character, EndReason, Event, Run, Screenplay, Session
from simstars.screenplay import build_screenplay
from simstars.simulation import simulate


class CharacterSpec:
    def __init__(self, name: str, role: str, traits: str, starting_location: str):
        self.name = name
        self.role = role
        self.traits = traits
        self.starting_location = starting_location


def _validate_session_input(world_description: str, locations: list[str], cast_specs: list[CharacterSpec]) -> None:
    """Pulled out of new_session() so guardrail boundaries (exactly
    MIN/MAX, not just one-off-either-side) can be unit-tested without
    touching enrichment/production/the DB at all.
    """
    if not (MIN_CHARACTERS <= len(cast_specs) <= MAX_CHARACTERS):
        raise ValueError(f"Cast must have {MIN_CHARACTERS}-{MAX_CHARACTERS} characters, got {len(cast_specs)}.")
    if not (MIN_LOCATIONS <= len(locations) <= MAX_LOCATIONS):
        raise ValueError(f"World must have {MIN_LOCATIONS}-{MAX_LOCATIONS} locations, got {len(locations)}.")
    for spec in cast_specs:
        if spec.starting_location not in locations:
            raise ValueError(f"{spec.name}'s starting location '{spec.starting_location}' isn't one of {locations}.")


def new_session(world_description: str, locations: list[str], cast_specs: list[CharacterSpec]) -> Session:
    # Defensive trim, not just the frontend's job: a stray trailing space
    # from any caller (this exact bug surfaced live via the web form -
    # locations got trimmed before being sent but a character's
    # starting_location didn't, so an exact-match comparison silently
    # failed) would otherwise produce a confusing "isn't one of [...]"
    # error even though the strings look identical to a human.
    world_description = world_description.strip()
    locations = [loc.strip() for loc in locations]
    for spec in cast_specs:
        spec.name = spec.name.strip()
        spec.role = spec.role.strip()
        spec.traits = spec.traits.strip()
        spec.starting_location = spec.starting_location.strip()

    _validate_session_input(world_description, locations, cast_specs)

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
    session.narrator_voice_id = production.cast_narrator_voice()

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


def list_sessions() -> list[Session]:
    """Newest first. Used by the web app's session-list view - no CLI
    equivalent exists today (the CLI only ever operates on a session id
    you already have)."""
    with get_session() as db:
        sessions = list(db.exec(select(Session).order_by(Session.created_at.desc())).all())
        for s in sessions:
            len(s.characters)  # force-load before detaching - see new_session()'s comment
        return sessions


def get_session_detail(session_id: str) -> tuple[Session, list[Character], list[Run]] | None:
    """Session + its characters + its runs (newest first), or None if the
    id doesn't exist. Used by the web app's session-detail view."""
    with get_session() as db:
        session = db.get(Session, session_id)
        if session is None:
            return None
        characters = list(session.characters)
        runs = sorted(session.runs, key=lambda r: r.created_at, reverse=True)
        session.locations  # force-load, same reason as _load_session
        return session, characters, runs


def get_run(run_id: str) -> Run | None:
    """Used by the web app's run-detail view and its audio-serving route."""
    with get_session() as db:
        return db.get(Run, run_id)


Attempt = tuple[list[Event], EndReason, dict, int]  # (events, end_reason, grade, branch_rounds_used)


def _score_grade(grade: dict) -> int:
    return sum(
        [
            grade["has_real_conflict"],
            grade["has_escalation"],
            grade["has_resolution"],
            grade["dialogue_carries_the_story"],
        ]
    )


def _select_best_attempt(attempts_data: list[Attempt]) -> Attempt:
    """Ships attempts_data[-1] if it passed (the loop in generate() only
    ever exits on a pass or on exhaustion, so a passing last attempt means
    that's the one that passed). Otherwise every attempt failed the critic:
    ship whichever scored highest on the critic's own criteria, ties broken
    toward the later attempt (later attempts had the benefit of
    prior-attempt feedback - see simulate()'s prior_feedback threading).
    This replaces the previous behavior of always shipping whichever
    attempt merely ran *last* regardless of how it scored.
    """
    if attempts_data[-1][2]["passes"]:
        return attempts_data[-1]

    best = attempts_data[0]
    best_score = _score_grade(best[2])
    for candidate in attempts_data[1:]:
        score = _score_grade(candidate[2])
        if score >= best_score:
            best = candidate
            best_score = score
    return best


def generate(session_id: str, producer_note: str | None = None) -> tuple[Run, list[Event], Screenplay]:
    """GENERATE phase only: simulate -> critic (with reroll) -> screenplay.
    Persists the Run with transcript/screenplay filled in, no audio yet.
    This is what `movie script` calls - no ElevenLabs cost.
    """
    session, characters = _load_session(session_id)
    # Fresh accumulator for this generate() call specifically - includes
    # every attempt's usage (a rejected/hard-failed attempt still cost real
    # money), not just the winning one's. See llm.py's usage-tracking note
    # on why this is a ContextVar rather than a plain global.
    reset_usage_tracking()

    # Every attempt is kept, not just the last one - see docs/design.md
    # follow-on plan: the previous version of this loop only ever shipped
    # whichever attempt ran *last* on exhaustion, not whichever actually
    # scored best. See _select_best_attempt().
    attempts_data: list[Attempt] = []
    attempts = 0

    while True:
        turn_budget = random.randint(MIN_TURN_BUDGET, MAX_TURN_BUDGET)
        attempts += 1
        try:
            events, end_reason, rounds_used = asyncio.run(simulate(session, characters, turn_budget, producer_note))
        except Exception:
            # A hard failure (e.g. every branch in some segment failed even
            # after call_structured's own retries - see llm.py/retry.py)
            # used to skip this retry loop entirely and fail the whole job
            # on attempt 1, even though a critic *rejection* gets up to
            # MAX_CRITIC_RETRIES more tries. Give a hard failure the same
            # budget rather than treating the two failure modes differently.
            if attempts > MAX_CRITIC_RETRIES:
                raise
            continue

        grade = evaluate(events, end_reason)
        attempts_data.append((events, end_reason, grade, rounds_used))

        if grade["passes"] or attempts > MAX_CRITIC_RETRIES:
            break

    best_events, best_end_reason, best_grade, best_rounds = _select_best_attempt(attempts_data)

    screenplay = build_screenplay(best_events)

    snapshot = usage_snapshot()
    total_calls = sum(counts["calls"] for counts in snapshot.values())
    total_input = sum(counts["input_tokens"] for counts in snapshot.values())
    total_output = sum(counts["output_tokens"] for counts in snapshot.values())
    total_cache_read = sum(counts["cache_read_input_tokens"] for counts in snapshot.values())

    run = Run(
        session_id=session_id,
        producer_note=producer_note,
        end_reason=best_end_reason,
        critic_attempts=attempts,
        critic_reasoning=best_grade.get("reasoning"),
        branch_rounds_used=best_rounds,
        transcript_json=json.dumps([e.model_dump(mode="json") for e in best_events]),
        screenplay_json=screenplay.model_dump_json(),
        llm_calls=total_calls,
        llm_input_tokens=total_input,
        llm_output_tokens=total_output,
        llm_cache_read_tokens=total_cache_read,
        estimated_cost_usd=estimated_cost_usd(snapshot),
    )
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    return run, best_events, screenplay


def produce_run(run_id: str) -> Run:
    """Runs (or re-runs) just the PRODUCE phase against an already-persisted
    run's screenplay. This is the recovery path for when production fails
    after simulation already succeeded: the transcript/screenplay are
    persisted before production starts specifically so a production
    failure never loses that (expensive) work (see production.py's module
    docstring) - but until this function existed, there was no way to
    actually act on that safety net. The only recovery path was clicking
    Play Movie again, which re-ran the entire simulation from scratch just
    to retry a comparatively cheap production step. Also doubles as "turn
    an existing script I already like into audio" without re-rolling it.
    """
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"No run '{run_id}'.")
    if not run.screenplay_json:
        raise ValueError(f"Run '{run_id}' has no screenplay yet - nothing to produce.")

    session, characters = _load_session(run.session_id)
    voice_by_name = {c.name: c.voice_id for c in characters if c.voice_id}
    screenplay = Screenplay.model_validate_json(run.screenplay_json)

    out_dir = audio_dir(run.session_id, run.id)
    final_path = asyncio.run(production.produce(screenplay, voice_by_name, out_dir, session.narrator_voice_id))

    run.final_audio_path = str(final_path)
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    return run


def play(session_id: str, producer_note: str | None = None) -> Run:
    """Full generate -> produce -> release."""
    run, _events, _screenplay = generate(session_id, producer_note)
    return produce_run(run.id)
