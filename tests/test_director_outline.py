"""DirectorAgent includes the pre-generation outline (see outline.py) in
its own prompt when given one, and never passes it to CharacterAgent -
matching the same omniscience boundary hidden secrets/wounds/goals already
get. Mocks call_structured entirely - no real API calls."""

from simstars.models import Character, Session
from simstars.simulation import CharacterAgent, DirectorAgent, WorldState


def _as_text(content) -> str:
    """system/user may be a plain string or a list of content blocks (see
    llm.cached_block) - flatten to plain text for substring assertions."""
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content)


def _session() -> Session:
    return Session(world_description="test world", locations="Kitchen,Lobby", forcing_mechanic="stuck here")


def _character(name: str) -> Character:
    return Character(session_id="s1", name=name, role="role", traits="traits", starting_location="Kitchen")


def test_director_includes_the_outline_in_its_prompt_when_given_one(monkeypatch):
    captured = {}

    def fake_call_structured(**kwargs):
        captured["user"] = kwargs["user"]
        return {"action": "cut", "cut_reason": "done"}

    monkeypatch.setattr("simstars.simulation.call_structured", fake_call_structured)
    characters = [_character("Ana")]
    director = DirectorAgent(_session(), characters, outline="- Ana's secret comes out")
    state = WorldState.initial(characters)

    director.decide_turn(state, turn_index=1, turns_remaining=10, producer_note=None)

    assert "Ana's secret comes out" in _as_text(captured["user"])


def test_director_omits_the_outline_block_when_none_given(monkeypatch):
    captured = {}

    def fake_call_structured(**kwargs):
        captured["user"] = kwargs["user"]
        return {"action": "cut", "cut_reason": "done"}

    monkeypatch.setattr("simstars.simulation.call_structured", fake_call_structured)
    characters = [_character("Ana")]
    director = DirectorAgent(_session(), characters)  # no outline
    state = WorldState.initial(characters)

    director.decide_turn(state, turn_index=1, turns_remaining=10, producer_note=None)

    assert "Intended dramatic arc" not in _as_text(captured["user"])


def test_character_agent_never_sees_the_outline(monkeypatch):
    # CharacterAgent has no outline parameter at all - this pins that down
    # so a future refactor can't accidentally thread it through.
    captured = {}

    def fake_call_structured(**kwargs):
        captured["system"] = kwargs["system"]
        captured["user"] = kwargs["user"]
        return {"type": "dialogue", "content": "hi"}

    monkeypatch.setattr("simstars.simulation.call_structured", fake_call_structured)
    character = _character("Ana")
    agent = CharacterAgent(character, _session())
    state = WorldState.initial([character])

    agent.decide_action(state, turn_index=1)

    combined = str(captured["system"]) + str(captured["user"])
    assert "dramatic arc" not in combined.lower()
