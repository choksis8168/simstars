"""The PRODUCE phase: voice casting, TTS, SFX/music generation via
ElevenLabs, and final mixing via pydub. See docs/design.md "Production
pipeline".

Mixing is scene-level, not per-line, for v1: dialogue lines within a scene
play back to back, a scene's SFX cues play at the top of the scene, and its
music cue is ducked underneath the whole scene. That's a known
simplification (no line-level timing/alignment yet) - good enough for a
first listenable cut; tightening it is a natural v2 improvement.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

from simstars.config import (
    MAX_CONCURRENT_ELEVENLABS_CALLS,
    TTS_SIMILARITY_BOOST,
    TTS_STABILITY,
    TTS_STYLE,
    VOICE_CASTING_MODEL,
    require_elevenlabs_key,
)
from simstars.llm import call_structured
from simstars.models import Character, EventType, Screenplay
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


def _infer_genders(characters: list[Character]) -> dict[str, str]:
    """One batched call inferring each character's likely voice gender from
    name/role/traits, so cast_voices can filter ElevenLabs' voice search by
    it. Real bug found via live usage: searching on role/traits text alone
    gave the API no gender signal at all, so e.g. a character named "Travis"
    could just as easily land a female-labeled voice as a male one - the
    search had nothing to disambiguate on. "neutral" means genuinely
    ambiguous or non-binary; cast_voices skips the gender filter for those
    rather than forcing a guess.
    """
    roster = "\n".join(f"- {c.name}: role={c.role}, traits={c.traits}" for c in characters)
    result = call_structured(
        model=VOICE_CASTING_MODEL,
        system=(
            "You infer the most likely voice gender for fictional characters, "
            "for casting a text-to-speech voice - use naming convention as the "
            "primary signal, role/traits as secondary."
        ),
        user=(
            f"Characters:\n{roster}\n\n"
            "For each character, decide 'male', 'female', or 'neutral' "
            "(genuinely ambiguous or non-binary)."
        ),
        tool_name="infer_voice_genders",
        tool_description="Infer each character's likely voice gender for TTS voice casting.",
        input_schema={
            "type": "object",
            "properties": {
                "genders": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "gender": {"type": "string", "enum": ["male", "female", "neutral"]},
                        },
                        "required": ["name", "gender"],
                    },
                }
            },
            "required": ["genders"],
        },
    )
    return {entry["name"]: entry["gender"] for entry in result["genders"]}


def cast_voices(characters: list[Character]) -> None:
    """Assigns character.voice_id in place, one ElevenLabs library voice per
    character, avoiding duplicates within the cast where possible.
    """
    client = _get_client()
    used: set[str] = set()
    fallback = [v.voice_id for v in client.voices.get_all().voices]

    try:
        genders = _infer_genders(characters)
    except Exception:  # noqa: BLE001 - gender matching is a nice-to-have, never blocks casting
        genders = {}

    for character in characters:
        query = f"{character.role} {character.traits}".strip()[:200]
        gender = genders.get(character.name)
        search_gender = gender if gender in ("male", "female") else None

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


# --- Per-clip generation (parallelized across a run) ---


async def _synthesize_line(text: str, voice_id: str, limiter: asyncio.Semaphore) -> bytes:
    client = _get_client()

    def call():
        return _collect(
            client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
                # Library voices default to conservative stored settings
                # when no voice_settings are passed - flat, monotonous line
                # readings were a real complaint from live usage. See
                # config.py for what these values do.
                voice_settings=VoiceSettings(
                    stability=TTS_STABILITY,
                    similarity_boost=TTS_SIMILARITY_BOOST,
                    style=TTS_STYLE,
                    use_speaker_boost=True,
                ),
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
) -> Path:
    # Shared across the whole call (not one per gather()) so dialogue and
    # SFX bursts within a scene both draw from the same budget - see
    # config.py's MAX_CONCURRENT_ELEVENLABS_CALLS note. Created here, not at
    # module level: asyncio.Semaphore is bound to the event loop it's
    # created in, and produce() may run under a fresh asyncio.run() each
    # call (see pipeline.play()).
    limiter = asyncio.Semaphore(MAX_CONCURRENT_ELEVENLABS_CALLS)
    scene_segments: list[AudioSegment] = []

    for scene_index, scene in enumerate(screenplay.scenes):
        # dialogue lines, back to back, throttled to at most
        # MAX_CONCURRENT_ELEVENLABS_CALLS in flight at once
        dialogue_events = [e for e in scene.events if e.type == EventType.DIALOGUE]
        dialogue_bytes = await asyncio.gather(
            *[
                _synthesize_line(e.content, voice_by_name.get(e.actor, next(iter(voice_by_name.values()))), limiter)
                for e in dialogue_events
            ]
        )
        dialogue_track = AudioSegment.silent(300)
        for i, data in enumerate(dialogue_bytes):
            path = out_dir / f"scene{scene_index}_line{i}.mp3"
            path.write_bytes(data)
            dialogue_track += _bytes_to_segment(data) + AudioSegment.silent(250)

        # SFX, same throttling, laid at the top of the scene
        sfx_bytes = await asyncio.gather(*[_generate_sfx(cue, limiter) for cue in scene.sfx_cues])
        sfx_track = AudioSegment.silent(len(dialogue_track))
        for i, data in enumerate(sfx_bytes):
            if data:
                sfx_track = sfx_track.overlay(_bytes_to_segment(data), position=0)

        scene_audio = dialogue_track.overlay(sfx_track)

        # music, ducked under the whole scene
        if scene.music_cue:
            music_bytes = await _generate_music(scene.music_cue, len(scene_audio), limiter)
            if music_bytes:
                music_track = _bytes_to_segment(music_bytes) - 18  # dB, ducked under dialogue
                music_track = music_track[: len(scene_audio)]
                scene_audio = scene_audio.overlay(music_track)

        scene_segments.append(scene_audio)

    final = AudioSegment.silent(0)
    for segment in scene_segments:
        final += segment + AudioSegment.silent(500)

    out_path = out_dir / "final_movie.mp3"
    final.export(out_path, format="mp3")
    return out_path
