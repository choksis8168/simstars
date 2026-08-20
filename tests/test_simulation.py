"""Covers the property the plan calls out as the whole point of the
location model: a character only perceives events from its own location,
except GLOBAL events which everyone perceives regardless."""

from moviesim.models import Character, Event, EventType
from moviesim.simulation import GLOBAL, WorldState


def _character(name: str, location: str) -> Character:
    return Character(
        session_id="s1",
        name=name,
        role="role",
        traits="traits",
        starting_location=location,
    )


def test_character_only_witnesses_events_in_its_own_location():
    ana = _character("Ana", "Kitchen")
    ben = _character("Ben", "Lobby")
    state = WorldState.initial([ana, ben])

    kitchen_event = Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi")
    state.apply_event(kitchen_event)

    assert kitchen_event in state.memory_for("Ana")
    assert kitchen_event not in state.memory_for("Ben")


def test_global_event_is_witnessed_by_everyone_regardless_of_location():
    ana = _character("Ana", "Kitchen")
    ben = _character("Ben", "Lobby")
    state = WorldState.initial([ana, ben])

    broadcast = Event(index=1, type=EventType.DIRECTOR, actor="director", location=GLOBAL, content="alarm sounds")
    state.apply_event(broadcast)

    assert broadcast in state.memory_for("Ana")
    assert broadcast in state.memory_for("Ben")


def test_movement_updates_location_and_future_events_are_witnessed_at_new_location():
    ana = _character("Ana", "Kitchen")
    ben = _character("Ben", "Lobby")
    state = WorldState.initial([ana, ben])

    move = Event(index=1, type=EventType.MOVEMENT, actor="Ana", location="Kitchen", content="Lobby")
    state.apply_event(move)
    assert state.character_locations["Ana"] == "Lobby"

    lobby_event = Event(index=2, type=EventType.DIALOGUE, actor="Ben", location="Lobby", content="hey")
    state.apply_event(lobby_event)
    assert lobby_event in state.memory_for("Ana")  # Ana is now in the Lobby too


def test_character_does_not_retroactively_witness_events_before_arriving():
    ana = _character("Ana", "Kitchen")
    ben = _character("Ben", "Lobby")
    state = WorldState.initial([ana, ben])

    lobby_event = Event(index=1, type=EventType.DIALOGUE, actor="Ben", location="Lobby", content="early line")
    state.apply_event(lobby_event)

    move = Event(index=2, type=EventType.MOVEMENT, actor="Ana", location="Kitchen", content="Lobby")
    state.apply_event(move)

    assert lobby_event not in state.memory_for("Ana")
