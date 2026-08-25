"""generate() persists Anthropic usage/cost on the Run it produces - see
llm.py's usage tracking. Mocks simulate/evaluate/build_screenplay and the
llm usage functions directly, since what's under test here is the wiring
(reset called, snapshot read, totals/cost land on the Run), not the
tracking math itself (covered in test_llm_usage.py) or story generation."""

from simstars import pipeline
from simstars.db import get_session
from simstars.models import Character, EndReason, Event, EventType, Screenplay, Session


def _seed_session() -> Session:
    session = Session(world_description="A test world", locations="Kitchen, Lobby", forcing_mechanic="stuck")
    with get_session() as db:
        db.add(session)
        db.commit()
        db.refresh(session)
        db.add(Character(session_id=session.id, name="Ana", role="tester", traits="curious", starting_location="Kitchen"))
        db.commit()
        db.refresh(session)
        len(session.characters)
        return session


def _passing_grade():
    return {
        "has_real_conflict": True, "has_escalation": True, "has_resolution": True,
        "dialogue_carries_the_story": True, "passes": True, "reasoning": "ok",
    }


def test_generate_persists_usage_and_cost_on_the_run(temp_db, monkeypatch):
    session = _seed_session()

    async def fake_simulate(session_, characters, turn_budget, note):
        events = [Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi")]
        return events, EndReason.RESOLVED, 0

    reset_calls = []
    fake_snapshot = {
        "claude-sonnet-5": {"calls": 5, "input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 500},
        "claude-haiku-4-5-20251001": {"calls": 3, "input_tokens": 300, "output_tokens": 100, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    }

    monkeypatch.setattr(pipeline, "simulate", fake_simulate)
    monkeypatch.setattr(pipeline, "evaluate", lambda events, end_reason: _passing_grade())
    monkeypatch.setattr(pipeline, "build_screenplay", lambda events: Screenplay(scenes=[]))
    monkeypatch.setattr(pipeline, "reset_usage_tracking", lambda: reset_calls.append(True))
    monkeypatch.setattr(pipeline, "usage_snapshot", lambda: fake_snapshot)
    monkeypatch.setattr(pipeline, "estimated_cost_usd", lambda snapshot: 0.42)

    run, events, screenplay = pipeline.generate(session.id)

    assert reset_calls == [True]  # tracking was reset for this call
    assert run.llm_calls == 8  # 5 + 3, summed across both models
    assert run.llm_input_tokens == 1300
    assert run.llm_output_tokens == 300
    assert run.llm_cache_read_tokens == 500
    assert run.estimated_cost_usd == 0.42
