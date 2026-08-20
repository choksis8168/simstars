"""Scene-grouping is deterministic (no LLM call), so it's tested directly
without mocking - the cue-adding LLM call (_add_cues) is exercised
separately via the mocked dry run, not here."""

from moviesim.models import Event, EventType
from moviesim.screenplay import _format_line, _group_scenes
from moviesim.simulation import GLOBAL


def test_consecutive_same_location_events_form_one_scene():
    events = [
        Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi"),
        Event(index=2, type=EventType.DIALOGUE, actor="Ben", location="Kitchen", content="hey"),
    ]
    scenes = _group_scenes(events)
    assert len(scenes) == 1
    assert scenes[0].location == "Kitchen"
    assert len(scenes[0].lines) == 2
    assert len(scenes[0].events) == 2


def test_location_change_starts_a_new_scene():
    events = [
        Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi"),
        Event(index=2, type=EventType.MOVEMENT, actor="Ana", location="Kitchen", content="Lobby"),
        Event(index=3, type=EventType.DIALOGUE, actor="Ana", location="Lobby", content="made it"),
    ]
    scenes = _group_scenes(events)
    assert [s.location for s in scenes] == ["Kitchen", "Lobby"]


def test_global_event_stays_in_the_current_scene_rather_than_splitting_it():
    events = [
        Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi"),
        Event(index=2, type=EventType.DIRECTOR, actor="director", location=GLOBAL, content="alarm"),
        Event(index=3, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="what was that"),
    ]
    scenes = _group_scenes(events)
    assert len(scenes) == 1
    assert len(scenes[0].lines) == 3


def test_format_line_dialogue_includes_target_aside():
    e = Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi", target="Ben")
    assert _format_line(e) == "ANA (to Ben): hi"
