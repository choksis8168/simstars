"""produce()'s narration handling (see production.py) - a scene-setting
line read by a dedicated narrator voice, added in response to live feedback
that scenes were hard to follow without any context-setting. Mocks the
ElevenLabs client entirely - no real API calls.
"""

import asyncio

from pydub import AudioSegment

from simstars import production
from simstars.models import Event, EventType, Scene, Screenplay


class RecordingTTSClient:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []  # (voice_id, text), in call order

    def convert(self, *, voice_id, text, **kwargs):
        self.calls.append((voice_id, text))
        return [b"fake-audio-bytes"]


class RecordingClient:
    def __init__(self):
        self.text_to_speech = RecordingTTSClient()
        self.text_to_sound_effects = self

    def convert(self, **kwargs):  # SFX calls route here via text_to_sound_effects
        return [b"fake-sfx-bytes"]


def _dialogue_scene(narration=None) -> Scene:
    events = [Event(index=1, type=EventType.DIALOGUE, actor="Ana", location="Kitchen", content="hi")]
    return Scene(location="Kitchen", heading="INT. KITCHEN", lines=[], events=events, narration=narration)


def test_narration_is_synthesized_with_the_narrator_voice_before_dialogue(tmp_path, monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(production, "_get_client", lambda: client)
    monkeypatch.setattr(production, "_bytes_to_segment", lambda data: AudioSegment.silent(100))

    screenplay = Screenplay(scenes=[_dialogue_scene(narration="It's the last night the shop is open.")])
    asyncio.run(production.produce(screenplay, {"Ana": "voice-ana"}, tmp_path, narrator_voice_id="voice-narrator"))

    assert client.text_to_speech.calls[0] == ("voice-narrator", "It's the last night the shop is open.")
    assert client.text_to_speech.calls[1] == ("voice-ana", "hi")


def test_narration_is_skipped_without_a_narrator_voice_id(tmp_path, monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(production, "_get_client", lambda: client)
    monkeypatch.setattr(production, "_bytes_to_segment", lambda data: AudioSegment.silent(100))

    screenplay = Screenplay(scenes=[_dialogue_scene(narration="Some scene-setting line.")])
    asyncio.run(production.produce(screenplay, {"Ana": "voice-ana"}, tmp_path, narrator_voice_id=None))

    # only the dialogue line was synthesized - no narration call fired
    assert client.text_to_speech.calls == [("voice-ana", "hi")]


def test_no_narration_call_when_the_scene_has_none(tmp_path, monkeypatch):
    client = RecordingClient()
    monkeypatch.setattr(production, "_get_client", lambda: client)
    monkeypatch.setattr(production, "_bytes_to_segment", lambda data: AudioSegment.silent(100))

    screenplay = Screenplay(scenes=[_dialogue_scene(narration=None)])
    asyncio.run(production.produce(screenplay, {"Ana": "voice-ana"}, tmp_path, narrator_voice_id="voice-narrator"))

    assert client.text_to_speech.calls == [("voice-ana", "hi")]
