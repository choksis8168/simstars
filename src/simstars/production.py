"""The PRODUCE phase: voice casting, TTS, SFX/music generation via
ElevenLabs, and final mixing via pydub. See docs/design.md "Production
pipeline".

This is treated as an actual audio production, not a stitch of independent
TTS calls: `Event.intensity` (set turn-by-turn by DirectorAgent - see
simulation.py) carries emotional context forward across a scene so a line
late in an escalating argument renders differently from the same
character's first line, and the inter-line pause tightens as intensity
rises; each character's own baseline stability/style (set once in
cast_voices, alongside voice_id) keeps that scaling from eroding what makes
*that* voice recognizable, since a global expressiveness setting pushed
far enough can start to blur a specific voice into something generic;
SFX/music cues are timestamped to a specific line (see screenplay.py's
sfx_cue_positions/music_swell_line_index) rather than all firing at the
top of the scene. Mixing is still scene-level, not sample-accurate - a
known simplification, not a bug.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

from simstars.config import (
    MAX_CONCURRENT_ELEVENLABS_CALLS,
    MUSIC_SWELL_BOOST_DB,
    MUSIC_SWELL_WINDOW_MS,
    PAUSE_MS_CALM,
    PAUSE_MS_INTENSE,
    TTS_SIMILARITY_BOOST,
    TTS_STABILITY,
    TTS_STABILITY_FLOOR,
    TTS_STABILITY_SWING,
    TTS_STYLE,
    TTS_STYLE_CEILING,
    TTS_STYLE_SWING,
    VOICE_CASTING_MODEL,
    require_elevenlabs_key,
)
from simstars.llm import call_structured
from simstars.models import Character, Event, EventType, Screenplay
from simstars.retry import with_retry

_client: ElevenLabs | None = None


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        _client = ElevenLabs(api_key=require_elevenlabs_key())
    return _client


def _collect(chunks) -> bytes:
    return b"".join(chunks)


# --- Voice casting (session-creation time, persisted, reused across regenerates) ---


def _infer_voice_traits(characters: list[Character]) -> dict[str, dict]:
    """One batched call inferring, per character: likely voice gender (so
    cast_voices can filter ElevenLabs' voice search by it - real bug found
    via live usage: searching on role/traits text alone gave the API no
    gender signal at all, so e.g. a character named "Travis" could just as
    easily land a female-labeled voice as a male one) and an expressiveness
    range (0.0-1.0, how far this character's delivery can swing with a
    scene's intensity before it stops sounding like *this* character - see
    _scaled_voice_settings). "neutral" gender means genuinely ambiguous or
    non-binary; cast_voices skips the gender filter for those rather than
    forcing a guess. The expressiveness range is a reasonable heuristic
    starting point from the character's traits, not something verified by
    ear - it's meant to be a sane default, not a claim of acoustic testing.
    """
    roster = "\n".join(f"- {c.name}: role={c.role}, traits={c.traits}" for c in characters)
    result = call_structured(
        model=VOICE_CASTING_MODEL,
        system=(
            "You infer two things about each fictional character, for casting "
            "and tuning a text-to-speech voice: likely gender (naming "
            "convention as the primary signal, role/traits as secondary), and "
            "an expressiveness range from 0.0 to 1.0 - how far this "
            "character's vocal delivery should be allowed to swing with a "
            "scene's emotional intensity before it risks no longer sounding "
            "like them. A volatile, hot-tempered, or dramatic character "
            "tolerates a wide range (0.7-1.0); a controlled, reserved, or "
            "deadpan character should keep a narrow one (0.1-0.3) so an "
            "intense scene doesn't distort their core voice."
        ),
        user=(
            f"Characters:\n{roster}\n\n"
            "For each character, decide 'male'/'female'/'neutral' (genuinely "
            "ambiguous or non-binary) and an expressiveness_range 0.0-1.0."
        ),
        tool_name="infer_voice_traits",
        tool_description="Infer each character's likely voice gender and expressiveness range for TTS casting/tuning.",
        input_schema={
            "type": "object",
            "properties": {
                "characters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "gender": {"type": "string", "enum": ["male", "female", "neutral"]},
                            "expressiveness_range": {"type": "number"},
                        },
                        "required": ["name", "gender", "expressiveness_range"],
                    },
                }
            },
            "required": ["characters"],
        },
    )
    return {entry["name"]: entry for entry in result["characters"]}


def cast_voices(characters: list[Character]) -> None:
    """Assigns character.voice_id in place, one ElevenLabs library voice per
    character, avoiding duplicates within the cast where possible.
    """
    client = _get_client()
    used: set[str] = set()
    fallback = [v.voice_id for v in client.voices.get_all().voices]

    try:
        traits_by_name = _infer_voice_traits(characters)
    except Exception:  # noqa: BLE001 - a nice-to-have, never blocks casting
        traits_by_name = {}

    for character in characters:
        query = f"{character.role} {character.traits}".strip()[:200]
        inferred = traits_by_name.get(character.name, {})
        gender = inferred.get("gender")
        search_gender = gender if gender in ("male", "female") else None
        # Baseline delivery + how far this character's own settings may
        # swing with intensity (see _scaled_voice_settings) - defaults to
        # the global constants/a middling range if inference failed, so
        # casting never blocks on this.
        character.voice_stability_base = TTS_STABILITY
        character.voice_style_base = TTS_STYLE
        character.voice_range = inferred.get("expressiveness_range", 0.5)

        def search(g=search_gender):
            return client.voices.search(search=query, gender=g, page_size=5)

        candidates: list[str] = []
        try:
            result = with_retry(search)
            candidates = [v.voice_id for v in result.voices]
            if not candidates and search_gender is not None:
                # gender + text search together came up empty - retry with
                # just the text search rather than falling all the way back
                # to the fully unfiltered library.
                result = with_retry(lambda: client.voices.search(search=query, page_size=5))
                candidates = [v.voice_id for v in result.voices]
        except Exception:  # noqa: BLE001 - fall through to the unfiltered library below
            candidates = []

        candidates = candidates or fallback
        choice = next((v for v in candidates if v not in used), candidates[0] if candidates else None)
        if choice is None:
            raise RuntimeError("No ElevenLabs voices available on this account.")
        character.voice_id = choice
        used.add(choice)


def cast_narrator_voice() -> str:
    """Picks one voice for scene-setting narration (see screenplay.py's
    narration field), cast once at session creation like character voices
    - reused across regenerates for the same session, same as them.
    Searches ElevenLabs' "narrative_story" use-case category rather than
    free text, since that's a real, purpose-built category in the library
    (distinct from the ad-hoc per-character search above) - a voice
    actually cast for reading narration, not just whichever came back for
    a role/traits string.
    """
    client = _get_client()
    try:
        result = with_retry(lambda: client.voices.search(use_cases="narrative_story", page_size=5))
        candidates = [v.voice_id for v in result.voices]
    except Exception:  # noqa: BLE001 - fall through to the unfiltered library below
        candidates = []
    if not candidates:
        candidates = [v.voice_id for v in with_retry(client.voices.get_all).voices]
    if not candidates:
        raise RuntimeError("No ElevenLabs voices available on this account.")
    return candidates[0]


# --- Per-line performance scaling (see Event.intensity, DirectorAgent) ---


def _scaled_voice_settings(
    stability_base: float, style_base: float, voice_range: float, intensity: float | None
) -> VoiceSettings:
    """Scales a character's own baseline delivery by this line's intensity
    and that character's own expressiveness range, so a scene's emotional
    arc actually builds across independently-generated TTS calls instead of
    every line rendering at the same flat setting - and so an intense scene
    pushes each character by *their own* tolerance, not one shared global
    ceiling that can start eroding a specific voice's recognizability. An
    absolute floor/ceiling (config.py) still applies regardless of
    character/range, so a bad inference can't send ElevenLabs a value that
    breaks synthesis entirely. intensity=None (untracked older data, or a
    non-tracked caller) renders at exactly the character's own baseline.
    """
    i = 0.0 if intensity is None else max(0.0, min(1.0, intensity))
    stability = max(TTS_STABILITY_FLOOR, stability_base - i * voice_range * TTS_STABILITY_SWING)
    style = min(TTS_STYLE_CEILING, style_base + i * voice_range * TTS_STYLE_SWING)
    return VoiceSettings(stability=stability, similarity_boost=TTS_SIMILARITY_BOOST, style=style, use_speaker_boost=True)


def _voice_settings_for_event(event: Event, voice_settings_by_name: dict[str, dict]) -> VoiceSettings:
    settings = voice_settings_by_name.get(event.actor, {})
    return _scaled_voice_settings(
        settings.get("stability_base", TTS_STABILITY),
        settings.get("style_base", TTS_STYLE),
        settings.get("range", 0.5),
        event.intensity,
    )


def _pause_after(intensity: float | None) -> int:
    """Milliseconds of silence after a line - a heated exchange should feel
    like it's talking over itself, not pause the same fixed beat a calm one
    would. intensity=None falls back to the middle of the range, matching
    the flat gap this used to always be.
    """
    i = 0.5 if intensity is None else max(0.0, min(1.0, intensity))
    return int(PAUSE_MS_CALM - i * (PAUSE_MS_CALM - PAUSE_MS_INTENSE))


# --- Per-clip generation (parallelized across a run) ---


async def _synthesize_line(
    text: str, voice_id: str, limiter: asyncio.Semaphore, voice_settings: VoiceSettings | None = None
) -> bytes:
    client = _get_client()
    # Library voices default to conservative stored settings when none are
    # passed - flat, monotonous line readings were a real complaint from
    # live usage. Callers with per-character/per-line context (produce())
    # pass a scaled VoiceSettings (see _scaled_voice_settings); anything
    # else (narration) falls back to the flat global defaults.
    settings = voice_settings or VoiceSettings(
        stability=TTS_STABILITY, similarity_boost=TTS_SIMILARITY_BOOST, style=TTS_STYLE, use_speaker_boost=True
    )

    def call():
        return _collect(
            client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                voice_settings=settings,
            )
        )

    async with limiter:
        return await asyncio.to_thread(with_retry, call)


async def _generate_sfx(cue: str, limiter: asyncio.Semaphore) -> bytes:
    client = _get_client()

    def call():
        return _collect(
            client.text_to_sound_effects.convert(
                text=cue,
                duration_seconds=3.0,
                output_format="mp3_44100_128",
            )
        )

    async with limiter:
        return await asyncio.to_thread(with_retry, call)


async def _generate_music(cue: str, length_ms: int, limiter: asyncio.Semaphore) -> bytes:
    client = _get_client()
    # Music generations can take a while and are the most likely call to be
    # unavailable on a given account/tier; degrade to silence rather than
    # failing the whole run.
    try:
        def call():
            return _collect(client.music.compose(prompt=cue, music_length_ms=length_ms))

        async with limiter:
            return await asyncio.to_thread(with_retry, call, attempts=2)
    except Exception:  # noqa: BLE001
        return b""


# --- Assembly ---


def _bytes_to_segment(data: bytes) -> AudioSegment:
    import io

    return AudioSegment.from_file(io.BytesIO(data), format="mp3") if data else AudioSegment.silent(0)


async def produce(
    screenplay: Screenplay,
    voice_by_name: dict[str, str],
    out_dir: Path,
    narrator_voice_id: str | None = None,
    voice_settings_by_name: dict[str, dict] | None = None,
) -> Path:
    # Shared across the whole call (not one per gather()) so dialogue and
    # SFX bursts within a scene both draw from the same budget - see
    # config.py's MAX_CONCURRENT_ELEVENLABS_CALLS note. Created here, not at
    # module level: asyncio.Semaphore is bound to the event loop it's
    # created in, and produce() may run under a fresh asyncio.run() each
    # call (see pipeline.play()).
    limiter = asyncio.Semaphore(MAX_CONCURRENT_ELEVENLABS_CALLS)
    voice_settings_by_name = voice_settings_by_name or {}
    scene_segments: list[AudioSegment] = []

    for scene_index, scene in enumerate(screenplay.scenes):
        # Narration, read first if this scene has one - a missing
        # narrator_voice_id (an older session predating this feature, or a
        # failed narrator cast) degrades to silently skipping it rather
        # than raising.
        narration_track = AudioSegment.silent(0)
        if scene.narration and narrator_voice_id:
            narration_bytes = await _synthesize_line(scene.narration, narrator_voice_id, limiter)
            (out_dir / f"scene{scene_index}_narration.mp3").write_bytes(narration_bytes)
            narration_track = _bytes_to_segment(narration_bytes) + AudioSegment.silent(400)

        # dialogue lines, back to back, throttled to at most
        # MAX_CONCURRENT_ELEVENLABS_CALLS in flight at once - each rendered
        # with that line's own scaled voice_settings (character baseline +
        # this beat's intensity), not one flat setting for every line.
        dialogue_events = [e for e in scene.events if e.type == EventType.DIALOGUE]
        dialogue_bytes = await asyncio.gather(
            *[
                _synthesize_line(
                    e.content,
                    voice_by_name.get(e.actor, next(iter(voice_by_name.values()))),
                    limiter,
                    _voice_settings_for_event(e, voice_settings_by_name),
                )
                for e in dialogue_events
            ]
        )

        # Walk every event (not just dialogue) so sfx_cue_positions/
        # music_swell_line_index - indexed against scene.lines/scene.events,
        # what screenplay._add_cues actually saw - land on the right
        # moment. Non-dialogue events don't advance the timeline (nothing
        # is synthesized for them); they just record "wherever we are right
        # now" so a cue attached to an action/movement/director line still
        # resolves to a sensible position.
        dialogue_track = AudioSegment.silent(300)
        line_positions_ms: list[int] = []  # one per scene.events/scene.lines entry
        dialogue_bytes_iter = iter(dialogue_bytes)
        dialogue_line_num = 0
        for event in scene.events:
            if event.type == EventType.DIALOGUE:
                data = next(dialogue_bytes_iter)
                (out_dir / f"scene{scene_index}_line{dialogue_line_num}.mp3").write_bytes(data)
                dialogue_line_num += 1
                dialogue_track += _bytes_to_segment(data)
                line_positions_ms.append(len(dialogue_track))
                dialogue_track += AudioSegment.silent(_pause_after(event.intensity))
            else:
                line_positions_ms.append(len(dialogue_track))

        narration_len = len(narration_track)
        dialogue_track = narration_track + dialogue_track
        line_positions_ms = [pos + narration_len for pos in line_positions_ms]

        # SFX, same throttling, each placed at its cued line's position
        # rather than all bunched at the top of the scene.
        sfx_bytes = await asyncio.gather(*[_generate_sfx(cue, limiter) for cue in scene.sfx_cues])
        sfx_track = AudioSegment.silent(len(dialogue_track))
        for i, data in enumerate(sfx_bytes):
            if not data:
                continue
            line_idx = scene.sfx_cue_positions[i] if i < len(scene.sfx_cue_positions) else 0
            position_ms = line_positions_ms[line_idx] if 0 <= line_idx < len(line_positions_ms) else 0
            sfx_track = sfx_track.overlay(_bytes_to_segment(data), position=position_ms)

        scene_audio = dialogue_track.overlay(sfx_track)

        # music, ducked under the whole scene, briefly swelling at the
        # scene's marked emotional peak (if any) rather than sitting at one
        # flat ducked level throughout.
        if scene.music_cue:
            music_bytes = await _generate_music(scene.music_cue, len(scene_audio), limiter)
            if music_bytes:
                music_track = (_bytes_to_segment(music_bytes) - 18)[: len(scene_audio)]  # dB, ducked under dialogue
                swell_idx = scene.music_swell_line_index
                if swell_idx is not None and 0 <= swell_idx < len(line_positions_ms):
                    center = line_positions_ms[swell_idx]
                    start = max(0, center - MUSIC_SWELL_WINDOW_MS // 2)
                    end = min(len(music_track), center + MUSIC_SWELL_WINDOW_MS // 2)
                    music_track = (
                        music_track[:start] + (music_track[start:end] + MUSIC_SWELL_BOOST_DB) + music_track[end:]
                    )
                scene_audio = scene_audio.overlay(music_track)

        scene_segments.append(scene_audio)

    final = AudioSegment.silent(0)
    for segment in scene_segments:
        final += segment + AudioSegment.silent(500)

    out_path = out_dir / "final_movie.mp3"
    final.export(out_path, format="mp3")
    return out_path
