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
        if current_lines and current_location is not None:
            scenes.append(
                Scene(
                    location=current_location,
                    heading=f"INT. {current_location.upper()}",
                    lines=list(current_lines),
                    events=list(current_events),
                )
            )

    for event in transcript:
        loc = event.location
        if loc != GLOBAL and loc != current_location:
            flush()
            current_location = loc
            current_lines = []
            current_events = []
        current_lines.append(_format_line(event))
        current_events.append(event)

    flush()
    return scenes


_CUE_SYSTEM = """You add sound design to a finished screenplay: sound \
effects for specific moments and one music mood cue per scene. Keep cues \
short and concrete (e.g. "glass shattering", "tense low strings"). Do not \
invent new plot content."""


def _add_cues(scenes: list[Scene]) -> list[Scene]:
    draft = "\n\n".join(
        f"SCENE {i}: {s.heading}\n" + "\n".join(s.lines) for i, s in enumerate(scenes)
    )
    result = call_structured(
        model=SCREENPLAY_MODEL,
        system=_CUE_SYSTEM,
        user=f"Screenplay draft:\n\n{draft}\n\nAdd SFX and a music mood cue for each scene.",
        tool_name="add_cues",
        tool_description="Attach sound-effect and music cues to each scene.",
        max_tokens=1536,
        input_schema={
            "type": "object",
            "properties": {
                "scenes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_index": {"type": "integer"},
                            "sfx_cues": {"type": "array", "items": {"type": "string"}},
                            "music_cue": {"type": "string"},
                        },
                        "required": ["scene_index", "sfx_cues", "music_cue"],
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
            scene.sfx_cues = cue.get("sfx_cues", [])
            scene.music_cue = cue.get("music_cue")
    return scenes


def build_screenplay(transcript: list[Event]) -> Screenplay:
    scenes = _group_scenes(transcript)
    scenes = _add_cues(scenes)
    return Screenplay(scenes=scenes)
