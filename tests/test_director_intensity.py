"""DirectorAgent's per-turn intensity value gets attached to whichever
Event that turn actually produces (character dialogue or a director-
injected event) - see simulation.py's _run_turns. This is what lets
production.py build emotional delivery across a scene instead of every
line resetting to the same flat setting. Mocks call_structured entirely -
no real API calls."""

from simstars.models import Character, EventType, Session
from simstars.simulation import CharacterAgent, DirectorAgent, WorldState, _run_turns


def _session() -> Session:
    return Session(world_description="test world", locations="Kitchen,Lobby", forcing_mechanic="stuck here")


def _character(name: str, location: str = "Kitchen") -> Character:
    return Character(session_id="s1", name=name, role="role", traits="traits", starting_location=location)


def _setup(char_locations: dict[str, str]):
    session = _session()
    characters = [_character(name, loc) for name, loc in char_locations.items()]
    director = DirectorAgent(session, characters)
    agents = {c.name: CharacterAgent(c, session) for c in characters}
    state = WorldState.initial(characters)
    return state, director, agents


def _dispatch(direct_response):
    """Returns the given decision for "direct" (director) calls, and a
    trivial dialogue line for "act" (character) calls - character turns
    always call through to CharacterAgent too."""

    def fake(**kwargs):
        if kwargs["tool_name"] == "direct":
            return direct_response
        return {"type": "dialogue", "content": "a line"}

    return fake


def test_intensity_is_attached_to_a_character_turns_event(monkeypatch):
    monkeypatch.setattr(
        "simstars.simulation.call_structured",
        _dispatch({"action": "character", "character_name": "Ana", "intensity": 0.8}),
    )
    state, director, agents = _setup({"Ana": "Kitchen"})

    _run_turns(state, director, agents, start_turn=1, num_turns=1, max_turns=1, producer_note=None, prior_feedback=None)

    assert state.events[0].intensity == 0.8


def test_intensity_is_attached_to_a_director_injected_events_event(monkeypatch):
    monkeypatch.setattr(
        "simstars.simulation.call_structured",
        lambda **kwargs: {"action": "event", "location": "Kitchen", "content": "a phone rings", "intensity": 0.3},
    )
    state, director, agents = _setup({"Ana": "Kitchen"})

    _run_turns(state, director, agents, start_turn=1, num_turns=1, max_turns=1, producer_note=None, prior_feedback=None)

    assert state.events[0].intensity == 0.3
    assert state.events[0].type == EventType.DIRECTOR


def test_missing_intensity_defaults_to_none_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(
        "simstars.simulation.call_structured",
        _dispatch({"action": "character", "character_name": "Ana"}),  # no "intensity" key at all
    )
    state, director, agents = _setup({"Ana": "Kitchen"})

    _run_turns(state, director, agents, start_turn=1, num_turns=1, max_turns=1, producer_note=None, prior_feedback=None)

    assert state.events[0].intensity is None


def test_intensity_survives_the_reveal_enforcement_backstop(monkeypatch):
    # The backstop (see _run_turns) overrides *who* acts after a director-
    # injected event, replacing `decision` wholesale - the original turn's
    # intensity reading must still make it onto the resulting event.
    fake = _FakeSequence(
        [
            {"action": "event", "location": "Kitchen", "content": "phone rings", "intensity": 0.2},
            {"action": "event", "location": "Kitchen", "content": "phone rings again", "intensity": 0.9},
        ]
    )
    import simstars.simulation as simulation_module

    monkeypatch.setattr(simulation_module, "call_structured", fake)
    state, director, agents = _setup({"Ana": "Kitchen", "Ben": "Kitchen"})

    _run_turns(state, director, agents, start_turn=1, num_turns=2, max_turns=2, producer_note=None, prior_feedback=None)

    assert state.events[1].type == EventType.DIALOGUE  # backstop forced a character reaction
    assert state.events[1].intensity == 0.9  # but the director's own intensity reading for that turn survived


class _FakeSequence:
    """Pops from a queue of "direct" (director) responses in order; any
    "act" (character) call - the backstop's forced reaction included -
    gets a trivial dialogue line, since it's not what these tests check."""

    def __init__(self, direct_responses):
        self._direct_responses = list(direct_responses)

    def __call__(self, **kwargs):
        if kwargs["tool_name"] == "direct":
            return self._direct_responses.pop(0)
        return {"type": "dialogue", "content": "a line"}
