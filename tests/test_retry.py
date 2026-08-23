"""with_retry is shared by llm.py (Anthropic) and production.py
(ElevenLabs) - tested once here rather than duplicated per caller."""

import pytest

from simstars.retry import with_retry


def test_returns_result_on_first_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert with_retry(fn) == "ok"
    assert len(calls) == 1


def test_retries_transient_failures_and_returns_the_eventual_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    attempts = []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return "ok"

    assert with_retry(fn, attempts=5) == "ok"
    assert len(attempts) == 3


def test_gives_up_after_exhausting_attempts_and_raises_the_last_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    attempts = []

    def fn():
        attempts.append(1)
        raise RuntimeError(f"failure {len(attempts)}")

    with pytest.raises(RuntimeError, match="failure 3"):
        with_retry(fn, attempts=3)
    assert len(attempts) == 3


def test_backs_off_exponentially_between_attempts_but_not_after_the_last_one(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda d: sleep_calls.append(d))

    def fn():
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError):
        with_retry(fn, attempts=3, base_delay=1.0)

    assert sleep_calls == [1.0, 2.0]  # 2 sleeps between 3 attempts, exponential, none after the last
