"""cast_voices() gender-aware search and _synthesize_line's voice_settings
(see production.py) - a real bug found via live usage: voice search on
role/traits text alone had no gender signal at all, so a character's
assigned voice could easily mismatch, and no voice_settings were passed at
all, leaving TTS delivery flat/monotonous by default. Mocks both the
ElevenLabs client and call_structured - no real API calls.
"""

import asyncio

import pytest

import simstars.production as production
from simstars.models import Character


def _character(name: str, role: str = "role", traits: str = "traits") -> Character:
    return Character(session_id="s1", name=name, role=role, traits=traits, starting_location="Kitchen")


class _FakeVoice:
    def __init__(self, voice_id: str):
        self.voice_id = voice_id


class _FakeSearchResult:
    def __init__(self, voice_ids: list[str]):
        self.voices = [_FakeVoice(v) for v in voice_ids]


class FakeVoicesClient:
    """Records every search() call's kwargs and returns scripted results
    keyed by the gender kwarg, so a test can assert exactly what was
    requested and control what comes back."""

    def __init__(self, results_by_gender: dict, fallback_ids: list[str]):
        self.calls: list[dict] = []
        self.results_by_gender = results_by_gender
        self._fallback_ids = fallback_ids

    def search(self, *, search=None, gender=None, page_size=5):
        self.calls.append({"search": search, "gender": gender})
        return self.results_by_gender.get(gender, _FakeSearchResult([]))

    def get_all(self):
        return _FakeSearchResult(self._fallback_ids)


class FakeClient:
    def __init__(self, voices_client):
        self.voices = voices_client


@pytest.fixture
def no_op_gender_inference(monkeypatch):
    """Most tests here care about search()'s behavior given a known gender
    mapping, not the LLM call that produces it - patched separately in the
    tests that do care about it."""
    monkeypatch.setattr(production, "_infer_genders", lambda characters: {})


def test_cast_voices_passes_the_inferred_gender_to_search(monkeypatch):
    monkeypatch.setattr(production, "_infer_genders", lambda characters: {"Travis": "male"})
    voices_client = FakeVoicesClient({"male": _FakeSearchResult(["voice-m1"])}, fallback_ids=["fallback-1"])
    monkeypatch.setattr(production, "_get_client", lambda: FakeClient(voices_client))

    characters = [_character("Travis")]
    production.cast_voices(characters)

    assert voices_client.calls[0]["gender"] == "male"
    assert characters[0].voice_id == "voice-m1"


def test_cast_voices_skips_the_gender_filter_for_neutral_or_unknown(no_op_gender_inference, monkeypatch):
    voices_client = FakeVoicesClient({None: _FakeSearchResult(["voice-x"])}, fallback_ids=["fallback-1"])
    monkeypatch.setattr(production, "_get_client", lambda: FakeClient(voices_client))

    characters = [_character("Ash")]
    production.cast_voices(characters)

    assert voices_client.calls[0]["gender"] is None
    assert characters[0].voice_id == "voice-x"


def test_cast_voices_retries_without_gender_when_the_gendered_search_is_empty(monkeypatch):
    monkeypatch.setattr(production, "_infer_genders", lambda characters: {"Katelyn": "female"})
    voices_client = FakeVoicesClient(
        {"female": _FakeSearchResult([]), None: _FakeSearchResult(["voice-general"])},
        fallback_ids=["fallback-1"],
    )
    monkeypatch.setattr(production, "_get_client", lambda: FakeClient(voices_client))

    characters = [_character("Katelyn")]
    production.cast_voices(characters)

    assert [c["gender"] for c in voices_client.calls] == ["female", None]
    assert characters[0].voice_id == "voice-general"


def test_cast_voices_falls_back_to_the_full_library_when_search_errors(no_op_gender_inference, monkeypatch):
    voices_client = FakeVoicesClient({}, fallback_ids=["fallback-1"])

    def boom(*, search=None, gender=None, page_size=5):
        raise RuntimeError("api down")

    voices_client.search = boom
    monkeypatch.setattr(production, "_get_client", lambda: FakeClient(voices_client))

    characters = [_character("Riley")]
    production.cast_voices(characters)

    assert characters[0].voice_id == "fallback-1"


def test_cast_voices_never_blocks_on_gender_inference_failing(monkeypatch):
    def boom(characters):
        raise RuntimeError("model call failed")

    monkeypatch.setattr(production, "_infer_genders", boom)
    voices_client = FakeVoicesClient({None: _FakeSearchResult(["voice-x"])}, fallback_ids=["fallback-1"])
    monkeypatch.setattr(production, "_get_client", lambda: FakeClient(voices_client))

    characters = [_character("Anyone")]
    production.cast_voices(characters)  # must not raise

    assert characters[0].voice_id == "voice-x"


def test_cast_voices_avoids_duplicate_voices_within_the_cast(no_op_gender_inference, monkeypatch):
    # Both characters' searches turn up the same candidate pool - the second
    # character should still get a distinct voice out of it rather than the
    # first character's already-used pick.
    voices_client = FakeVoicesClient({None: _FakeSearchResult(["voice-a", "voice-b"])}, fallback_ids=["fallback-1"])
    monkeypatch.setattr(production, "_get_client", lambda: FakeClient(voices_client))

    characters = [_character("Ana"), _character("Ben")]
    production.cast_voices(characters)

    assert characters[0].voice_id == "voice-a"
    assert characters[1].voice_id == "voice-b"


def test_synthesize_line_passes_tuned_voice_settings(monkeypatch):
    captured = {}

    class FakeTTSClient:
        def convert(self, **kwargs):
            captured.update(kwargs)
            return [b"audio-bytes"]

    class FakeClientWithTTS:
        def __init__(self):
            self.text_to_speech = FakeTTSClient()

    monkeypatch.setattr(production, "_get_client", lambda: FakeClientWithTTS())

    asyncio.run(production._synthesize_line("hello", "voice-1", asyncio.Semaphore(1)))

    settings = captured["voice_settings"]
    assert settings.stability is not None
    assert settings.style is not None
    assert 0 <= settings.stability <= 1
