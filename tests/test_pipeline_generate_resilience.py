"""generate()'s retry loop used to only retry on a critic *rejection*
(a completed-but-flat story) - a hard failure (simulate() raising, e.g.
every branch failed even after call_structured's own retries) skipped the
loop entirely and failed the whole job on attempt 1, despite
MAX_CRITIC_RETRIES nominally allowing more tries. These test the fix
directly against generate(), not just the extracted pieces, since the bug
was in the loop's control flow itself."""

import pytest

from simstars import pipeline
from simstars.config import MAX_CRITIC_RETRIES
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
        "has_real_conflict": True,
        "has_escalation": True,
        "has_resolution": True,
        "dialogue_carries_the_story": True,
        "passes": True,
        "reasoning": "ok",
    }


def _events():
    return [Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi")]


def _make_flaky_simulate(fail_times: int):
    """Raises for the first `fail_times` calls, then succeeds."""
    state = {"calls": 0}

    async def fake(session, characters, turn_budget, note):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise RuntimeError(f"simulated hard failure #{state['calls']}")
        return _events(), EndReason.RESOLVED, 0

    return fake, state


def test_generate_retries_a_hard_failure_and_succeeds_within_budget(temp_db, monkeypatch):
    session = _seed_session()
    fake_simulate, state = _make_flaky_simulate(fail_times=1)
    monkeypatch.setattr(pipeline, "simulate", fake_simulate)
    monkeypatch.setattr(pipeline, "evaluate", lambda events, end_reason: _passing_grade())
    monkeypatch.setattr(pipeline, "build_screenplay", lambda events: Screenplay(scenes=[]))

    run, events, screenplay = pipeline.generate(session.id)

    assert state["calls"] == 2  # failed once, succeeded on the retry
    assert run.critic_attempts == 2
    assert events == _events()


def test_generate_gives_up_after_exhausting_the_retry_budget_on_hard_failures(temp_db, monkeypatch):
    session = _seed_session()
    # fails every time - more than MAX_CRITIC_RETRIES + 1 total attempts worth
    fake_simulate, state = _make_flaky_simulate(fail_times=100)
    monkeypatch.setattr(pipeline, "simulate", fake_simulate)
    monkeypatch.setattr(pipeline, "evaluate", lambda events, end_reason: _passing_grade())

    with pytest.raises(RuntimeError, match=f"simulated hard failure #{MAX_CRITIC_RETRIES + 1}"):
        pipeline.generate(session.id)

    assert state["calls"] == MAX_CRITIC_RETRIES + 1  # same total-attempt budget a quality rejection would get


def test_generate_uses_a_later_success_after_earlier_hard_failures(temp_db, monkeypatch):
    session = _seed_session()
    # fails on the first MAX_CRITIC_RETRIES attempts, succeeds on the last one available
    fake_simulate, state = _make_flaky_simulate(fail_times=MAX_CRITIC_RETRIES)
    monkeypatch.setattr(pipeline, "simulate", fake_simulate)
    monkeypatch.setattr(pipeline, "evaluate", lambda events, end_reason: _passing_grade())
    monkeypatch.setattr(pipeline, "build_screenplay", lambda events: Screenplay(scenes=[]))

    run, events, screenplay = pipeline.generate(session.id)

    assert state["calls"] == MAX_CRITIC_RETRIES + 1
    assert run.end_reason == EndReason.RESOLVED
