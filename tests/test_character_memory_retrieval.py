"""CharacterAgent's retrieval-vs-full-memory branch (see simulation.py's
decide_action and memory_store.py's module docstring). Mocks
retrieve_relevant_memories entirely - no real DB/embedding model touched -
since what's under test here is *which path gets used and what gets passed
to it*, not memory_store's own ranking logic (covered separately in
tests/test_memory_store.py against a real Postgres instance).
"""

from simstars.config import MEMORY_RETRIEVAL_THRESHOLD
from simstars.models import Character, EventType, Session
from simstars.simulation import CharacterAgent, WorldState
import simstars.simulation as simulation_module


def _session() -> Session:
    return Session(world_description="test world", locations="Kitchen,Lobby", forcing_mechanic="stuck here")


def _character(name: str = "Ana") -> Character:
    return Character(
        session_id="s1", name=name, role="role", traits="traits", starting_location="Kitchen",
        secret="s", wound="w", hidden_goal="g", relationship_seeds="r",
    )


def _state_with_witnessed_count(character_name: str, count: int) -> WorldState:
    state = WorldState.initial([_character(character_name)])
    for i in range(1, count + 1):
        state.apply_event(
            simulation_module.Event(index=i, type=EventType.DIALOGUE, actor=character_name, location="Kitchen", content=f"line {i}")
        )
    return state


def _fake_call_structured(**kwargs):
    return {"type": "dialogue", "content": "ok"}


def test_uses_full_chronological_memory_at_or_below_the_threshold(monkeypatch):
    monkeypatch.setattr("simstars.simulation.call_structured", _fake_call_structured)
    called = {"count": 0}
    monkeypatch.setattr(
        "simstars.simulation.retrieve_relevant_memories",
        lambda **kwargs: called.__setitem__("count", called["count"] + 1) or [],
    )
    agent = CharacterAgent(_character(), _session())
    state = _state_with_witnessed_count("Ana", MEMORY_RETRIEVAL_THRESHOLD)

    agent.decide_action(state, turn_index=MEMORY_RETRIEVAL_THRESHOLD + 1)

    assert called["count"] == 0


def test_switches_to_retrieval_once_past_the_threshold(monkeypatch):
    monkeypatch.setattr("simstars.simulation.call_structured", _fake_call_structured)
    captured = {}

    def fake_retrieve(*, memories, query_text, current_turn):
        captured["memories"] = memories
        captured["query_text"] = query_text
        captured["current_turn"] = current_turn
        return ["[3] (Kitchen) Ana: line 3"]

    monkeypatch.setattr("simstars.simulation.retrieve_relevant_memories", fake_retrieve)
    agent = CharacterAgent(_character(), _session())
    count = MEMORY_RETRIEVAL_THRESHOLD + 1
    state = _state_with_witnessed_count("Ana", count)

    agent.decide_action(state, turn_index=count + 1)

    assert captured["current_turn"] == count + 1
    assert len(captured["memories"]) == count
    assert captured["memories"][0] == (1, "[1] (Kitchen) Ana: line 1")
    assert "line 1" not in captured["query_text"]  # query summarizes location/presence/most-recent, not the full log


def test_falls_back_to_a_placeholder_when_retrieval_returns_nothing(monkeypatch):
    monkeypatch.setattr("simstars.simulation.call_structured", _fake_call_structured)
    monkeypatch.setattr("simstars.simulation.retrieve_relevant_memories", lambda **kwargs: [])
    agent = CharacterAgent(_character(), _session())
    count = MEMORY_RETRIEVAL_THRESHOLD + 1
    state = _state_with_witnessed_count("Ana", count)

    # Should not raise, and should still produce a valid Event.
    event = agent.decide_action(state, turn_index=count + 1)

    assert event.actor == "Ana"
