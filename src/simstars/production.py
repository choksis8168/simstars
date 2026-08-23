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
import time
from pathlib import Path
from typing import Callable, TypeVar

from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

from simstars.config import MAX_CONCURRENT_ELEVENLABS_CALLS, require_elevenlabs_key
from simstars.models import Character, EventType, Screenplay

T = TypeVar("T")

_client: ElevenLabs | None = None


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        _client = ElevenLabs(api_key=require_elevenlabs_key())
    return _client


def _with_retry(fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0) -> T:
    """Resilience for transient ElevenLabs API failures. The transcript/
    screenplay are already persisted by the time this runs, so a failure
    here never loses simulation work - see pipeline.py.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any transient API failure retries
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(base_delay * (2**attempt))
    assert last_error is not None
    raise last_error


def _collect(chunks) -> bytes:
    return b"".join(chunks)


# --- Voice casting (session-creation time, persisted, reused across regenerates) ---


def cast_voices(characters: list[Character]) -> None:
    """Assigns character.voice_id in place, one ElevenLabs library voice per
    character, avoiding duplicates within the cast where possible.
    """
    client = _get_client()
    used: set[str] = set()
    fallback = [v.voice_id for v in client.voices.get_all().voices]

    for character in characters:
        query = f"{character.role} {character.traits}".strip()[:200]

        def search():
            return client.voices.search(search=query, page_size=5)

        candidates: list[str] = []
        try:
            result = _with_retry(search)
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
            )
        )

    async with limiter:
        return await asyncio.to_thread(_with_retry, call)


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
        return await asyncio.to_thread(_with_retry, call)


async def _generate_music(cue: str, length_ms: int, limiter: asyncio.Semaphore) -> bytes:
    client = _get_client()
    # Music generations can take a while and are the most likely call to be
    # unavailable on a given account/tier; degrade to silence rather than
    # failing the whole run.
    try:
        def call():
            return _collect(client.music.compose(prompt=cue, music_length_ms=length_ms))

        async with limiter:
            return await asyncio.to_thread(_with_retry, call, attempts=2)
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
