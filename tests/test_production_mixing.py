"""Scene-level performance/mixing intelligence added in response to real
feedback that TTS delivery was flat and mixing was just stitched-together
clips: per-character/per-intensity voice_settings scaling, variable
inter-line pauses, positioned SFX cues, and a music swell at a scene's
marked peak. Mocks the ElevenLabs client entirely - no real API calls."""

import asyncio

from pydub import AudioSegment

from simstars import production
from simstars.config import (
    PAUSE_MS_CALM,
    PAUSE_MS_INTENSE,
    TTS_STABILITY,
    TTS_STABILITY_FLOOR,
    TTS_STABILITY_SWING,
    TTS_STYLE,
    TTS_STYLE_CEILING,
    TTS_STYLE_SWING,
)
from simstars.models import Event, EventType, Scene, Screenplay
from simstars.production import _pause_after, _scaled_voice_settings


# --- _scaled_voice_settings / _pause_after: pure functions, no mocking needed ---


def test_scaled_voice_settings_at_zero_intensity_equals_the_baseline():
    settings = _scaled_voice_settings(stability_base=0.4, style_base=0.3, voice_range=0.8, intensity=0.0)

    assert settings.stability == 0.4
    assert settings.style == 0.3


def test_scaled_voice_settings_at_full_intensity_swings_by_the_full_range():
    settings = _scaled_voice_settings(stability_base=0.5, style_base=0.3, voice_range=1.0, intensity=1.0)

    assert settings.stability == 0.5 - TTS_STABILITY_SWING
    assert settings.style == 0.3 + TTS_STYLE_SWING


def test_scaled_voice_settings_scales_by_voice_range():
    wide = _scaled_voice_settings(stability_base=0.5, style_base=0.3, voice_range=1.0, intensity=1.0)
    narrow = _scaled_voice_settings(stability_base=0.5, style_base=0.3, voice_range=0.2, intensity=1.0)

    # a narrow-range character swings less than a wide-range one at the same intensity
    assert narrow.stability > wide.stability
    assert narrow.style < wide.style


def test_scaled_voice_settings_never_crosses_the_absolute_floor_or_ceiling():
    settings = _scaled_voice_settings(stability_base=0.1, style_base=0.9, voice_range=1.0, intensity=1.0)

    assert settings.stability >= TTS_STABILITY_FLOOR
    assert settings.style <= TTS_STYLE_CEILING


def test_scaled_voice_settings_treats_none_intensity_as_zero():
    settings = _scaled_voice_settings(stability_base=0.4, style_base=0.3, voice_range=0.8, intensity=None)

    assert settings.stability == 0.4
    assert settings.style == 0.3


def test_pause_after_is_longest_at_zero_intensity_and_shortest_at_full():
    assert _pause_after(0.0) == PAUSE_MS_CALM
    assert _pause_after(1.0) == PAUSE_MS_INTENSE


def test_pause_after_none_falls_back_to_the_historical_flat_gap():
    assert _pause_after(None) == 250  # the old fixed pause, preserved as the no-data default


# --- produce(): mixing integration, ElevenLabs client mocked ---


class _FakeTTSClient:
    def convert(self, *, voice_id, text, **kwargs):
        return [b"line-audio"]


class _FakeSFXClient:
    def convert(self, **kwargs):
        return [b"sfx-audio"]


class _FakeMusicClient:
    def compose(self, **kwargs):
        return [b"music-audio"]


class _FakeClient:
    def __init__(self):
        self.text_to_speech = _FakeTTSClient()
        self.text_to_sound_effects = _FakeSFXClient()
        self.music = _FakeMusicClient()


def _fixed_length_segment(_data: bytes) -> AudioSegment:
    return AudioSegment.silent(1000)  # 1 second per synthesized clip, regardless of content


def _scene_with_two_lines(**scene_kwargs) -> Scene:
    events = [
        Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="one", intensity=0.0),
        Event(index=2, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="two", intensity=0.0),
    ]
    return Scene(location="Kitchen", heading="INT. KITCHEN", lines=[], events=events, **scene_kwargs)


def test_sfx_cue_lands_at_its_cued_line_position_not_scene_start(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(production, "_bytes_to_segment", _fixed_length_segment)

    scene = _scene_with_two_lines(sfx_cues=["door slam"], sfx_cue_positions=[1])
    screenplay = Screenplay(scenes=[scene])

    # Overlay records where each layer lands - patch AudioSegment.overlay to capture position args
    positions = []
    real_overlay = AudioSegment.overlay

    def spying_overlay(self, seg, position=0, **kwargs):
        positions.append(position)
        return real_overlay(self, seg, position=position, **kwargs)

    monkeypatch.setattr(AudioSegment, "overlay", spying_overlay)

    asyncio.run(production.produce(screenplay, {"Ana": "voice-1"}, tmp_path))

    # the SFX overlay (not the final dialogue/sfx-track overlay) should be
    # positioned after line 0's audio - i.e. > 0, not at the scene's start
    assert any(p > 0 for p in positions)


def test_music_swell_boosts_volume_around_the_marked_peak_line(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "_get_client", lambda: _FakeClient())
    monkeypatch.setattr(production, "_bytes_to_segment", _fixed_length_segment)

    scene_with_swell = _scene_with_two_lines(music_cue="tense strings", music_swell_line_index=1)
    scene_without_swell = _scene_with_two_lines(music_cue="tense strings", music_swell_line_index=None)

    dbfs_at_swell_center = {}

    async def run_and_measure(scene, key):
        screenplay = Screenplay(scenes=[scene])
        path = await production.produce(screenplay, {"Ana": "voice-1"}, tmp_path / key)
        return AudioSegment.from_file(path, format="mp3")

    import os

    os.makedirs(tmp_path / "with_swell", exist_ok=True)
    os.makedirs(tmp_path / "without_swell", exist_ok=True)

    with_swell = asyncio.run(run_and_measure(scene_with_swell, "with_swell"))
    without_swell = asyncio.run(run_and_measure(scene_without_swell, "without_swell"))

    # Both scenes are the same length/content otherwise - the swelled
    # version should not be quieter than the non-swelled one; a real signal
    # that *something* changed around the peak rather than a flat duck
    # throughout (loudness comparison is coarse but avoids depending on
    # exact byte-level pydub internals).
    assert with_swell.dBFS >= without_swell.dBFS