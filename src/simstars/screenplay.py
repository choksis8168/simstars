"""Transcript -> Screenplay. Scene *structure* (who said what, in which
scene) is derived deterministically from the transcript, since production
depends on it being exactly right. SFX/music cues are the one part that
benefits from creative judgment, so those come from a single Claude call
over the whole formatted draft.
"""

from __future__ import annotations

from simstars.config import CRITIC_MODEL as SCREENPLAY_MODEL  # same tier of judgment call
from simstars.llm import call_structured
from simstars.models import Event, EventType, Scene, Screenplay
from simstars.simulation import GLOBAL


def _format_line(event: Event) -> str:
    if event.type == EventType.DIALOGUE:
        aside = f" (to {event.target})" if event.target else ""
        return f"{event.actor.upper()}{aside}: {event.content}"
    if event.type == EventType.ACTION:
        return f"[{event.actor}] {event.content}"
    if event.type == EventType.MOVEMENT:
        return f"[{event.actor} moves to {event.content}]"
    # director-injected
    where = "[EVERYWHERE]" if event.location == GLOBAL else "[EVENT]"
    return f"{where} {event.content}"


def _group_scenes(transcript: list[Event]) -> list[Scene]:
    scenes: list[Scene] = []
    current_location: str | None = None
    current_lines: list[str] = []
    current_events: list[Event] = []

    def flush():
        # current_location is guaranteed non-None by the time current_lines
        # is non-empty (see the loop condition below), so this only needs
        # to guard against an empty transcript.
        if current_lines:
            heading = "EVERYWHERE" if current_location == GLOBAL else f"INT. {current_location.upper()}"
            scenes.append(
                Scene(
                    location=current_location,
                    heading=heading,
                    lines=list(current_lines),
                    events=list(current_events),
                )
            )

    for event in transcript:
        loc = event.location
        # Start a new scene on a real location change, or when nothing has
        # started a scene yet at all - including a leading run of
        # global-only events, which would otherwise never get flushed
        # since current_location would stay None forever (a real bug this
        # guarded against: those events were being silently dropped from
        # the screenplay, and therefore from the audio).
        if current_location is None or (loc != GLOBAL and loc != current_location):
            flush()
            current_location = loc
            current_lines = []
            current_events = []
        current_lines.append(_format_line(event))
        current_events.append(event)

    flush()
    return scenes


_CUE_SYSTEM = """You add sound design and narration to a finished \
screenplay, for an AUDIO-ONLY production - the audience has no picture, \
only what gets read aloud. This is an actual audio production, not just a \
list of cues to sprinkle over the top: where a cue lands relative to the \
lines around it, and where the scene's music comes up out of its ducked \
level, changes what the scene feels like. Each scene's lines are given to \
you numbered - reference those exact numbers.

For each scene, write:

- narration: one or two spoken sentences a narrator reads at the top of \
the scene, establishing whatever a listener needs and can't see - where \
this is, roughly when, who's present, and enough situational context that \
the dialogue that follows makes sense on first listen. Do not narrate \
things the dialogue already says out loud - narration fills the gap \
between scenes, it doesn't repeat what's about to be spoken. Do not reveal \
anything a character hasn't actually said yet (no spoiling a later beat).
- sfx_cues + sfx_cue_positions: 2-4 specific, concrete sound effects for \
that scene (e.g. "glass shattering", "a chair scraping back", "rain \
against a window"), each paired with the exact numbered line it should \
land at - a door slam belongs at the line where the door slams, not at \
the top of the scene by default. Same length and order as sfx_cues.
- music_cue: one short mood description for underscore music, which plays \
ducked under the whole scene.
- music_swell_line_index: the one numbered line that's this scene's \
emotional peak, where the music should briefly swell up out of that ducked \
level - or null if the scene genuinely has no single peak (a quiet, even \
scene shouldn't get a fake one).

Keep cues short and concrete. Do not invent new plot content beyond what \
narration is allowed to add for scene-setting."""


def _add_cues(scenes: list[Scene]) -> list[Scene]:
    draft = "\n\n".join(
        f"SCENE {i}: {s.heading}\n" + "\n".join(f"[{j}] {line}" for j, line in enumerate(s.lines))
        for i, s in enumerate(scenes)
    )
    result = call_structured(
        model=SCREENPLAY_MODEL,
        system=_CUE_SYSTEM,
        user=f"Screenplay draft:\n\n{draft}\n\nAdd narration, SFX, and music cues for each scene.",
        tool_name="add_cues",
        tool_description="Attach narration, sound-effect, and music cues to each scene.",
        max_tokens=2048,
        input_schema={
            "type": "object",
            "properties": {
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_index": {"type": "integer"},
                            "narration": {"type": "string"},
                            "sfx_cues": {"type": "array", "items": {"type": "string"}},
                            "sfx_cue_positions": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Same length/order as sfx_cues - the numbered line each cue lands at.",
                            },
                            "music_cue": {"type": "string"},
                            "music_swell_line_index": {
                                "type": ["integer", "null"],
                                "description": "The numbered line marking this scene's emotional peak, or null.",
                            },
                        },
                        "required": [
                            "scene_index", "narration", "sfx_cues", "sfx_cue_positions",
                            "music_cue", "music_swell_line_index",
                        ],
                    },
                }
            },
            "required": ["scenes"],
        },
    )
    by_index = {c["scene_index"]: c for c in result["scenes"]}
    for i, scene in enumerate(scenes):
        cue = by_index.get(i)
        if cue:
            scene.narration = cue.get("narration") or None
            scene.sfx_cues = cue.get("sfx_cues", [])
            scene.sfx_cue_positions = cue.get("sfx_cue_positions", [])
            scene.music_cue = cue.get("music_cue")
            scene.music_swell_line_index = cue.get("music_swell_line_index")
    return scenes


def build_screenplay(transcript: list[Event]) -> Screenplay:
    scenes = _group_scenes(transcript)
    scenes = _add_cues(scenes)
    return Screenplay(scenes=scenes)
