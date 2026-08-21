"""Control-flow tests for the branching lookahead in simulate() (see
docs/design.md follow-on plan "Story-Quality Variance Fix"). These mock
call_structured entirely - no API calls - and use small segment/branch
config so the tests stay fast. What's being verified is the *mechanics*
(branching happens, still_flat triggers a bounded re-preview with feedback
threaded through, a cut short-circuits correctly whether it happens during
a preview or during the linear segment finish) - not story quality, which
can only be judged with real API calls.
"""

import asyncio
import re
import threading
from collections import Counter

import pytest

from simstars.models import Character, EndReason, Session
from simstars.simulation import simulate


def _session() -> Session:
    return Session(world_description="test world", locations="Kitchen,Lobby", forcing_mechanic="stuck here")


def _character(name: str, location: str = "Kitchen") -> Character:
    return Character(session_id="s1", name=name, role="role", traits="traits", starting_location=location)


class FakeLLM:
    """Dispatches on tool_name; thread-safe since branch previews call this
    concurrently via asyncio.to_thread."""

    def __init__(self):
        self.lock = threading.Lock()
        self.call_count = 0
        self.tool_counts: Counter = Counter()
        self.direct_calls: list[str] = []  # user prompt text, in call order
        self.cut_from_call_n: int | None = None  # "direct" returns cut once call_count >= this
        self.compare_responses: list[dict] = []  # popped in order; falls back to a default after

    def __call__(self, *, model, system, user, tool_name, tool_description, input_schema, max_tokens=1024):
        with self.lock:
            self.call_count += 1
            n = self.call_count
            self.tool_counts[tool_name] += 1

        if tool_name == "direct":
            with self.lock:
                self.direct_calls.append(user)
            if self.cut_from_call_n is not None and n >= self.cut_from_call_n:
                return {"action": "cut", "cut_reason": "done"}
            match = re.search(r"- (\S+) \(", user)
            name = match.group(1) if match else "Unknown"
            return {"action": "character", "character_name": name}

        if tool_name == "act":
            return {"type": "dialogue", "content": f"line-{n}"}

        if tool_name == "pick_best_continuation":
            with self.lock:
                if self.compare_responses:
                    return self.compare_responses.pop(0)
            return {"best_index": 0, "still_flat": False, "reasoning": "fine"}

        if tool_name == "grade_story":
            return {
                "has_real_conflict": True,
                "has_escalation": True,
                "has_resolution": True,
                "dialogue_carries_the_story": True,
                "passes": True,
                "reasoning": "ok",
            }

        raise ValueError(f"unexpected tool_name {tool_name}")


@pytest.fixture
def small_config(monkeypatch):
    """Small enough to keep tests fast and their call counts easy to reason about."""
    monkeypatch.setattr("simstars.simulation.SEGMENT_LENGTH", 4)
    monkeypatch.setattr("simstars.simulation.PREVIEW_LENGTH", 2)
    monkeypatch.setattr("simstars.simulation.BRANCH_FACTOR", 2)
    monkeypatch.setattr("simstars.simulation.MAX_SEGMENT_ROUNDS", 2)


def _patch_llm(monkeypatch) -> FakeLLM:
    fake = FakeLLM()
    # simulate() calls call_structured directly for director/character
    # turns; simulate() also calls compare_previews() (in critic.py), which
    # has its own imported call_structured reference - both need patching.
    monkeypatch.setattr("simstars.simulation.call_structured", fake)
    monkeypatch.setattr("simstars.critic.call_structured", fake)
    return fake


def test_basic_run_respects_turn_budget_and_branches(small_config, monkeypatch):
    fake = _patch_llm(monkeypatch)
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=4))

    assert len(events) == 4
    assert end_reason == EndReason.TURN_BUDGET
    assert rounds == 0
    # preview: BRANCH_FACTOR(2) x PREVIEW_LENGTH(2) = 4 direct calls,
    # then linear finish: remaining(2) x 1 = 2 more direct calls = 6 total
    assert fake.tool_counts["direct"] == 6
    assert fake.tool_counts["pick_best_continuation"] == 1


def test_still_flat_triggers_one_bounded_repreview_round_with_feedback(small_config, monkeypatch):
    fake = _patch_llm(monkeypatch)
    fake.compare_responses = [
        {"best_index": 0, "still_flat": True, "reasoning": "too flat"},
        {"best_index": 0, "still_flat": False, "reasoning": "better now"},
    ]
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=4))

    assert rounds == 1
    assert fake.tool_counts["pick_best_continuation"] == 2
    # round 2's preview calls are direct_calls[4:8] (round 1 used [0:4]);
    # they should carry the prior round's failing feedback as guidance.
    round_two_previews = fake.direct_calls[4:8]
    assert any("too flat" in u for u in round_two_previews)


def test_cut_during_preview_ends_simulation_immediately(small_config, monkeypatch):
    fake = _patch_llm(monkeypatch)
    fake.cut_from_call_n = 1  # every direct call cuts immediately
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=4))

    assert end_reason == EndReason.RESOLVED
    assert events == []  # cut happens before any event is ever applied


def test_cut_during_linear_finish_ends_simulation_and_keeps_preview_events(small_config, monkeypatch):
    fake = _patch_llm(monkeypatch)
    fake.cut_from_call_n = 5  # preview uses calls 1-4; linear finish starts at call 5
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=4))

    assert end_reason == EndReason.RESOLVED
    assert len(events) == 2  # the winning preview's 2 turns were committed before the cut
