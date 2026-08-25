"""Scene-grouping is deterministic (no LLM call), so it's tested directly
without mocking. _add_cues (narration/SFX/music, the one LLM call in this
module) is tested separately below with call_structured mocked."""

from simstars.models import Event, EventType
from simstars.screenplay import _add_cues, _format_line, _group_scenes
from simstars.simulation import GLOBAL


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


def test_empty_transcript_produces_no_scenes():
    assert _group_scenes([]) == []


def test_leading_global_event_is_not_dropped():
    # Regression test: a run of global events before any located event used
    # to leave `current_location` as None forever, so flush()'s old guard
    # (`current_location is not None`) meant these events were silently
    # dropped from the screenplay - and therefore from the audio.
    events = [
        Event(index=1, type=EventType.DIRECTOR, actor="director", location=GLOBAL, content="power goes out"),
        Event(index=2, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hello?"),
    ]
    scenes = _group_scenes(events)
    total_events = sum(len(s.events) for s in scenes)
    assert total_events == 2
    assert any("power goes out" in line for s in scenes for line in s.lines)


def test_transcript_of_only_global_events_still_produces_one_scene():
    events = [
        Event(index=1, type=EventType.DIRECTOR, actor="director", location=GLOBAL, content="the lights flicker"),
        Event(index=2, type=EventType.DIRECTOR, actor="director", location=GLOBAL, content="then go dark"),
    ]
    scenes = _group_scenes(events)
    assert len(scenes) == 1
    assert scenes[0].location == GLOBAL
    assert scenes[0].heading == "EVERYWHERE"
    assert len(scenes[0].lines) == 2


def test_global_event_after_a_location_change_stays_in_the_new_scene():
    # Note: a MOVEMENT event's `location` is the character's *origin* (where
    # the departure is witnessed), not the destination - so the scene only
    # actually transitions to "Lobby" once a later event is located there.
    events = [
        Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi"),
        Event(index=2, type=EventType.MOVEMENT, actor="Ana", location="Kitchen", content="Lobby"),
        Event(index=3, type=EventType.DIALOGUE, actor="Ana", location="Lobby", content="made it"),
        Event(index=4, type=EventType.DIRECTOR, actor="director", location=GLOBAL, content="a bell rings"),
    ]
    scenes = _group_scenes(events)
    assert [s.location for s in scenes] == ["Kitchen", "Lobby"]
    assert any("a bell rings" in line for line in scenes[1].lines)


def test_format_line_for_each_event_type():
    action = Event(index=1, type=EventType.ACTION, actor="Ana", location="Kitchen", content="sighs")
    movement = Event(index=2, type=EventType.MOVEMENT, actor="Ana", location="Kitchen", content="Lobby")
    global_event = Event(index=3, type=EventType.DIRECTOR, actor="director", location=GLOBAL, content="thunder")
    local_event = Event(index=4, type=EventType.DIRECTOR, actor="director", location="Kitchen", content="knock")

    assert _format_line(action) == "[Ana] sighs"
    assert _format_line(movement) == "[Ana moves to Lobby]"
    assert _format_line(global_event) == "[EVERYWHERE] thunder"
    assert _format_line(local_event) == "[EVENT] knock"


# --- _add_cues: narration/SFX/music attachment, call_structured mocked ---


def _scene(location: str = "Kitchen") -> "Scene":
    from simstars.models import Scene

    return Scene(location=location, heading=f"INT. {location.upper()}", lines=["ANA: hi"], events=[])


def test_add_cues_attaches_narration_sfx_and_music_per_scene(monkeypatch):
    def fake_call_structured(**kwargs):
        return {
            "scenes": [
                {
                    "scene_index": 0,
                    "narration": "It's the last night the shop is open.",
                    "sfx_cues": ["door chime", "footsteps on wood"],
                    "music_cue": "tense low strings",
                }
            ]
        }

    monkeypatch.setattr("simstars.screenplay.call_structured", fake_call_structured)
    scenes = _add_cues([_scene()])

    assert scenes[0].narration == "It's the last night the shop is open."
    assert scenes[0].sfx_cues == ["door chime", "footsteps on wood"]
    assert scenes[0].music_cue == "tense low strings"


def test_add_cues_leaves_a_scene_without_a_matching_index_unset(monkeypatch):
    monkeypatch.setattr("simstars.screenplay.call_structured", lambda **kwargs: {"scenes": []})
    scenes = _add_cues([_scene()])

    assert scenes[0].narration is None
    assert scenes[0].sfx_cues == []
    assert scenes[0].music_cue is None


def test_add_cues_treats_an_empty_narration_string_as_none(monkeypatch):
    monkeypatch.setattr(
        "simstars.screenplay.call_structured",
        lambda **kwargs: {"scenes": [{"scene_index": 0, "narration": "", "sfx_cues": [], "music_cue": "quiet"}]},
    )
    scenes = _add_cues([_scene()])

    assert scenes[0].narration is None
