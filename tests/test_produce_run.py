"""produce_run() retries (or runs for the first time) just the PRODUCE
phase against an already-persisted run's screenplay - see pipeline.py's
docstring on it: this is the actual recovery path for the safety net
production.py's persist-before-produce ordering was always supposed to
provide, which had no callable path before this."""

import pytest

from simstars import pipeline
from simstars.db import get_session
from simstars.models import Character, Run, Session


def _seed_session_and_character() -> Session:
    session = Session(world_description="A test world", locations="Kitchen, Lobby", forcing_mechanic="stuck")
    with get_session() as db:
        db.add(session)
        db.commit()
        db.refresh(session)
        db.add(
            Character(
                session_id=session.id, name="Ana", role="tester", traits="curious",
                starting_location="Kitchen", voice_id="voice-123",
            )
        )
        db.commit()
        db.refresh(session)
        len(session.characters)
        return session


def _seed_run(session_id: str, *, screenplay_json: str | None = '{"scenes": []}') -> Run:
    run = Run(session_id=session_id, screenplay_json=screenplay_json)
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)
        return run


def test_produce_run_raises_for_unknown_run(temp_db):
    with pytest.raises(ValueError, match="No run"):
        pipeline.produce_run("does-not-exist")


def test_produce_run_raises_when_there_is_no_screenplay_yet(temp_db):
    session = _seed_session_and_character()
    run = _seed_run(session.id, screenplay_json=None)

    with pytest.raises(ValueError, match="no screenplay yet"):
        pipeline.produce_run(run.id)


def test_produce_run_calls_production_with_the_persisted_screenplay_and_voices(temp_db, monkeypatch, tmp_path):
    session = _seed_session_and_character()
    run = _seed_run(session.id, screenplay_json='{"scenes": []}')

    captured = {}

    async def fake_produce(screenplay, voice_by_name, out_dir):
        captured["voice_by_name"] = voice_by_name
        return tmp_path / "final_movie.mp3"

    monkeypatch.setattr(pipeline.production, "produce", fake_produce)

    updated = pipeline.produce_run(run.id)

    assert captured["voice_by_name"] == {"Ana": "voice-123"}
    assert updated.final_audio_path == str(tmp_path / "final_movie.mp3")

    with get_session() as db:
        persisted = db.get(Run, run.id)
        assert persisted.final_audio_path == str(tmp_path / "final_movie.mp3")


def test_play_is_generate_then_produce_run(temp_db, monkeypatch, tmp_path):
    """play() should just be generate() + produce_run() - verifies the
    refactor didn't change play()'s externally-visible behavior."""
    session = _seed_session_and_character()

    async def fake_simulate(session_, characters, turn_budget, note):
        from simstars.models import EndReason, Event, EventType

        return [Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi")], EndReason.RESOLVED, 0

    def fake_evaluate(events, end_reason):
        return {
            "has_real_conflict": True, "has_escalation": True, "has_resolution": True,
            "dialogue_carries_the_story": True, "passes": True, "reasoning": "ok",
        }

    from simstars.models import Screenplay

    async def fake_produce(screenplay, voice_by_name, out_dir):
        return tmp_path / "final_movie.mp3"

    monkeypatch.setattr(pipeline, "simulate", fake_simulate)
    monkeypatch.setattr(pipeline, "evaluate", fake_evaluate)
    monkeypatch.setattr(pipeline, "build_screenplay", lambda events: Screenplay(scenes=[]))
    monkeypatch.setattr(pipeline.production, "produce", fake_produce)

    run = pipeline.play(session.id)

    assert run.final_audio_path == str(tmp_path / "final_movie.mp3")
    assert run.screenplay_json is not None  # generate()'s half of the work happened too
