"""Regression test for a real ElevenLabs 429 (concurrent_limit_exceeded)
hit live: a scene with more dialogue lines/SFX cues than the account's
concurrency limit fired that many TTS/SFX calls at once with no cap - see
config.py's MAX_CONCURRENT_ELEVENLABS_CALLS note."""

import asyncio
import threading
import time

from pydub import AudioSegment

from simstars import production
from simstars.config import MAX_CONCURRENT_ELEVENLABS_CALLS
from simstars.models import Event, EventType, Scene, Screenplay


class ConcurrencyTrackingClient:
    """Fakes the bits of the ElevenLabs client production.py touches,
    tracking how many calls are in flight simultaneously."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0
        self.max_seen = 0
        self.text_to_speech = self
        self.text_to_sound_effects = self

    def _track(self):
        with self.lock:
            self.current += 1
            self.max_seen = max(self.max_seen, self.current)
        time.sleep(0.05)  # long enough that overlapping calls actually overlap
        with self.lock:
            self.current -= 1
        return [b"fake-audio-bytes"]

    def convert(self, **kwargs):
        return self._track()


def test_produce_never_exceeds_the_concurrency_limit(tmp_path, monkeypatch):
    fake_client = ConcurrencyTrackingClient()
    monkeypatch.setattr(production, "_get_client", lambda: fake_client)
    # This test is about concurrency throttling, not real audio decoding -
    # the fake client returns placeholder bytes that aren't valid mp3 data.
    monkeypatch.setattr(production, "_bytes_to_segment", lambda data: AudioSegment.silent(100))

    # More dialogue lines and SFX cues than the limit, so the throttling
    # actually gets exercised rather than trivially passing because there
    # was never enough concurrency to violate it.
    line_count = MAX_CONCURRENT_ELEVENLABS_CALLS + 3
    events = [
        Event(index=i, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content=f"line {i}")
        for i in range(line_count)
    ]
    scene = Scene(
        location="Kitchen",
        heading="INT. KITCHEN",
        lines=[],
        events=events,
        sfx_cues=["door creak", "clock ticking", "footsteps", "kettle whistling"],
    )
    screenplay = Screenplay(scenes=[scene])

    asyncio.run(production.produce(screenplay, {"Ana": "voice-1"}, tmp_path))

    assert fake_client.max_seen <= MAX_CONCURRENT_ELEVENLABS_CALLS
    assert fake_client.max_seen > 1  # proves real concurrency happened, not accidental serialization
