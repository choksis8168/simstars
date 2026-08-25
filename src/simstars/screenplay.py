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
only what gets read aloud. For each scene, write:

- narration: one or two spoken sentences a narrator reads at the top of \
the scene, establishing whatever a listener needs and can't see - where \
this is, roughly when, who's present, and enough situational context that \
the dialogue that follows makes sense on first listen. Do not narrate \
things the dialogue already says out loud - narration fills the gap \
between scenes, it doesn't repeat what's about to be spoken. Do not reveal \
anything a character hasn't actually said yet (no spoiling a later beat).
- sfx_cues: 2-4 specific, concrete sound effects for that scene (e.g. \
"glass shattering", "a chair scraping back", "rain against a window") - \
err toward including more grounded, moment-specific cues rather than one \
generic ambience cue, so the scene has real sonic texture, not silence \
between lines.
- music_cue: one short mood description for underscore music.

Keep cues short and concrete. Do not invent new plot content beyond what \
narration is allowed to add for scene-setting."""


def _add_cues(scenes: list[Scene]) -> list[Scene]:
    draft = "\n\n".join(
        f"SCENE {i}: {s.heading}\n" + "\n".join(s.lines) for i, s in enumerate(scenes)
    )
    result = call_structured(
        model=SCREENPLAY_MODEL,
        system=_CUE_SYSTEM,
        user=f"Screenplay draft:\n\n{draft}\n\nAdd narration, SFX, and a music mood cue for each scene.",
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
                            "music_cue": {"type": "string"},
                        },
                        "required": ["scene_index", "narration", "sfx_cues", "music_cue"],
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
            scene.music_cue = cue.get("music_cue")
    return scenes


def build_screenplay(transcript: list[Event]) -> Screenplay:
    scenes = _group_scenes(transcript)
    scenes = _add_cues(scenes)
    return Screenplay(scenes=scenes)
