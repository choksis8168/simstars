"""call_structured retries transient failures via with_retry (retry.py) -
without this, one blip on any single call (out of the several hundred a
real run can make - see docs/design.md) used to kill the entire job."""

import pytest

from simstars import llm


class _FakeToolUseBlock:
    def __init__(self, name, input_):
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)  # each item: an Exception to raise, or a response to return

    def create(self, **kwargs):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _call(**overrides):
    kwargs = dict(
        model="m",
        system="s",
        user="u",
        tool_name="do_thing",
        tool_description="d",
        input_schema={"type": "object", "properties": {}},
    )
    kwargs.update(overrides)
    return llm.call_structured(**kwargs)


def test_retries_a_transient_failure_and_returns_the_eventual_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    success = _FakeResponse([_FakeToolUseBlock("do_thing", {"ok": True})])
    monkeypatch.setattr(llm, "_get_client", lambda: _FakeClient([RuntimeError("transient blip"), success]))

    assert _call() == {"ok": True}


def test_gives_up_after_repeated_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(
        llm, "_get_client", lambda: _FakeClient([RuntimeError("1"), RuntimeError("2"), RuntimeError("3")])
    )

    with pytest.raises(RuntimeError, match="3"):
        _call()


def test_raises_a_clear_error_if_the_model_never_returns_the_expected_tool_call(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    # a response with no matching tool_use block, e.g. the model answered
    # in plain text instead of calling the forced tool
    empty = _FakeResponse([])
    monkeypatch.setattr(llm, "_get_client", lambda: _FakeClient([empty]))

    with pytest.raises(RuntimeError, match="do_thing"):
        _call()
