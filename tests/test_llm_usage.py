"""llm.py's usage tracking/cost estimation - added so "how much did this
run cost" has a real per-Run number instead of a guess. Mocks the Anthropic
client entirely - no real API calls, no real cost."""

import threading

import pytest

from simstars import llm


class _FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeToolUseBlock:
    def __init__(self, name, input_):
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeResponse:
    def __init__(self, content, usage):
        self.content = content
        self.usage = usage


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _call(model="claude-haiku-4-5-20251001", usage=None, **overrides):
    response = _FakeResponse(
        [_FakeToolUseBlock("do_thing", {"ok": True})],
        usage or _FakeUsage(input_tokens=100, output_tokens=50),
    )
    kwargs = dict(
        model=model, system="s", user="u", tool_name="do_thing", tool_description="d",
        input_schema={"type": "object", "properties": {}},
    )
    kwargs.update(overrides)
    import simstars.llm as llm_module

    llm_module._get_client = lambda: _FakeClient([response])
    return llm.call_structured(**kwargs)


@pytest.fixture(autouse=True)
def _isolated_usage_context():
    # Each test starts its own accumulator, same as pipeline.generate()
    # does via reset_usage_tracking() - without this, tests would leak
    # into whatever context pytest happens to run them under.
    llm.reset_usage_tracking()
    yield


def test_usage_snapshot_is_empty_before_any_tracked_call():
    assert llm.usage_snapshot() == {}


def test_a_tracked_call_appears_in_the_snapshot():
    _call(usage=_FakeUsage(input_tokens=100, output_tokens=50))

    snapshot = llm.usage_snapshot()
    assert snapshot["claude-haiku-4-5-20251001"]["calls"] == 1
    assert snapshot["claude-haiku-4-5-20251001"]["input_tokens"] == 100
    assert snapshot["claude-haiku-4-5-20251001"]["output_tokens"] == 50


def test_usage_accumulates_across_multiple_calls_to_the_same_model():
    _call(usage=_FakeUsage(input_tokens=100, output_tokens=50))
    _call(usage=_FakeUsage(input_tokens=200, output_tokens=75))

    snapshot = llm.usage_snapshot()
    assert snapshot["claude-haiku-4-5-20251001"]["calls"] == 2
    assert snapshot["claude-haiku-4-5-20251001"]["input_tokens"] == 300
    assert snapshot["claude-haiku-4-5-20251001"]["output_tokens"] == 125


def test_usage_is_tracked_separately_per_model():
    _call(model="claude-haiku-4-5-20251001", usage=_FakeUsage(input_tokens=100, output_tokens=50))
    _call(model="claude-sonnet-5", usage=_FakeUsage(input_tokens=1000, output_tokens=500))

    snapshot = llm.usage_snapshot()
    assert set(snapshot.keys()) == {"claude-haiku-4-5-20251001", "claude-sonnet-5"}
    assert snapshot["claude-sonnet-5"]["input_tokens"] == 1000


def test_reset_usage_tracking_clears_prior_accumulation():
    _call(usage=_FakeUsage(input_tokens=100, output_tokens=50))
    llm.reset_usage_tracking()

    assert llm.usage_snapshot() == {}


def test_a_call_with_no_usage_attribute_is_not_tracked():
    response = _FakeResponse([_FakeToolUseBlock("do_thing", {"ok": True})], usage=None)
    llm._get_client = lambda: _FakeClient([response])

    llm.call_structured(
        model="claude-haiku-4-5-20251001", system="s", user="u", tool_name="do_thing",
        tool_description="d", input_schema={"type": "object", "properties": {}},
    )

    assert llm.usage_snapshot() == {}


def test_calls_made_before_reset_usage_tracking_is_ever_called_are_silently_untracked():
    # Simulates enrichment/voice-casting calls outside a generate() call -
    # should never raise just because tracking was never started.
    import contextvars

    ctx = contextvars.Context()
    ctx.run(lambda: _call(usage=_FakeUsage(input_tokens=100, output_tokens=50)))
    # this test's own context (autouse fixture already reset it) is unaffected
    assert llm.usage_snapshot() == {}


def test_estimated_cost_computes_from_configured_per_model_rates():
    from simstars.config import ANTHROPIC_PRICING_PER_MILLION_TOKENS

    snapshot = {
        "claude-haiku-4-5-20251001": {
            "calls": 1, "input_tokens": 1_000_000, "output_tokens": 1_000_000,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        }
    }
    rates = ANTHROPIC_PRICING_PER_MILLION_TOKENS["claude-haiku-4-5-20251001"]

    cost = llm.estimated_cost_usd(snapshot)

    assert cost == pytest.approx(rates["input"] + rates["output"])


def test_estimated_cost_applies_cache_multipliers():
    from simstars.config import ANTHROPIC_PRICING_PER_MILLION_TOKENS, CACHE_READ_PRICE_MULTIPLIER, CACHE_WRITE_PRICE_MULTIPLIER

    snapshot = {
        "claude-sonnet-5": {
            "calls": 1, "input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000,
        }
    }
    rate = ANTHROPIC_PRICING_PER_MILLION_TOKENS["claude-sonnet-5"]["input"]

    cost = llm.estimated_cost_usd(snapshot)

    assert cost == pytest.approx(rate * CACHE_WRITE_PRICE_MULTIPLIER + rate * CACHE_READ_PRICE_MULTIPLIER)


def test_estimated_cost_skips_a_model_with_no_price_entry():
    snapshot = {"some-unknown-model": {"calls": 1, "input_tokens": 1_000_000, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}

    assert llm.estimated_cost_usd(snapshot) == 0.0


def test_estimated_cost_defaults_to_the_current_snapshot_when_none_is_passed():
    _call(usage=_FakeUsage(input_tokens=1_000_000, output_tokens=0))

    assert llm.estimated_cost_usd() > 0


def test_concurrent_calls_in_different_threads_do_not_double_count_each_other():
    # Each thread gets its own accumulator via reset_usage_tracking() - a
    # bug here (e.g. a plain module-level dict instead of a ContextVar)
    # would let one thread's calls bleed into the other's total.
    results = {}

    def worker(name, n_calls):
        llm.reset_usage_tracking()
        for _ in range(n_calls):
            _call(usage=_FakeUsage(input_tokens=10, output_tokens=5))
        results[name] = llm.usage_snapshot()["claude-haiku-4-5-20251001"]["calls"]

    t1 = threading.Thread(target=worker, args=("a", 3))
    t2 = threading.Thread(target=worker, args=("b", 7))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results == {"a": 3, "b": 7}
