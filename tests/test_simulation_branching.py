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

from simstars.models import Character, EndReason, Event, EventType, Session
from simstars.simulation import (
    CharacterAgent,
    DirectorAgent,
    WorldState,
    _partition_preview_results,
    _resolve_round,
    _run_turns,
    simulate,
)


def _session() -> Session:
    return Session(world_description="test world", locations="Kitchen,Lobby", forcing_mechanic="stuck here")


def _character(name: str, location: str = "Kitchen") -> Character:
    return Character(session_id="s1", name=name, role="role", traits="traits", starting_location=location)


def _as_text(content) -> str:
    """system/user may now be a plain string or a list of content blocks
    (see llm.cached_block) - flatten to plain text for the fixture's string
    matching, regardless of which shape the caller used."""
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content)


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
        self.direct_script: list[dict] = []  # exact decisions for successive "direct" calls, popped in order

    def __call__(self, *, model, system, user, tool_name, tool_description, input_schema, max_tokens=1024):
        user_text = _as_text(user)
        with self.lock:
            self.call_count += 1
            n = self.call_count
            self.tool_counts[tool_name] += 1

        if tool_name == "direct":
            with self.lock:
                self.direct_calls.append(user_text)
                scripted = self.direct_script.pop(0) if self.direct_script else None
            if isinstance(scripted, BaseException):
                raise scripted
            if scripted is not None:
                return scripted
            if self.cut_from_call_n is not None and n >= self.cut_from_call_n:
                return {"action": "cut", "cut_reason": "done"}
            match = re.search(r"- (\S+) \(", user_text)
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


def test_multi_segment_simulation_has_no_turn_gaps_or_duplicates(monkeypatch):
    monkeypatch.setattr("simstars.simulation.SEGMENT_LENGTH", 3)
    monkeypatch.setattr("simstars.simulation.PREVIEW_LENGTH", 1)
    monkeypatch.setattr("simstars.simulation.BRANCH_FACTOR", 2)
    monkeypatch.setattr("simstars.simulation.MAX_SEGMENT_ROUNDS", 1)
    fake = _patch_llm(monkeypatch)
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    # 8 turns / segment_len 3 spans three segments (3 + 3 + 2)
    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=8))

    assert end_reason == EndReason.TURN_BUDGET
    assert len(events) == 8
    assert [e.index for e in events] == list(range(1, 9))


def test_single_turn_simulation_does_not_crash(small_config, monkeypatch):
    # max_turns(1) is smaller than PREVIEW_LENGTH(2) - preview_len must clip
    # to fit rather than overrunning the budget.
    fake = _patch_llm(monkeypatch)
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=1))

    assert len(events) == 1
    assert end_reason == EndReason.TURN_BUDGET


def test_repreview_is_bounded_by_max_segment_rounds_when_still_flat_persists(small_config, monkeypatch):
    fake = _patch_llm(monkeypatch)
    fake.compare_responses = [
        {"best_index": 0, "still_flat": True, "reasoning": "flat 1"},
        {"best_index": 0, "still_flat": True, "reasoning": "flat 2"},  # MAX_SEGMENT_ROUNDS=2: last round regardless
    ]
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=4))

    assert fake.tool_counts["pick_best_continuation"] == 2  # tried exactly 2 rounds, not more
    assert rounds == 1  # only the round *before* the final one counts as an extra re-preview
    assert len(events) == 4  # still commits and finishes despite persistent flatness, not stuck


# --- _resolve_round: pure per-round winner selection, no concurrency/mocking needed ---


def test_resolve_round_falls_back_to_first_candidate_on_out_of_range_index():
    state_a, state_b = WorldState.initial([]), WorldState.initial([])
    results = [(state_a, None), (state_b, None)]
    comparison = {"best_index": 99, "still_flat": False, "reasoning": "model drift"}

    winning_state, winning_end_reason, should_stop = _resolve_round(results, comparison)

    assert winning_state is state_a
    assert should_stop is True


def test_resolve_round_ignores_a_resolved_branch_that_was_not_chosen():
    resolved_but_unchosen, chosen_and_ongoing = WorldState.initial([]), WorldState.initial([])
    results = [(resolved_but_unchosen, EndReason.RESOLVED), (chosen_and_ongoing, None)]
    comparison = {"best_index": 1, "still_flat": False, "reasoning": "candidate 1 is more promising"}

    winning_state, winning_end_reason, should_stop = _resolve_round(results, comparison)

    assert winning_state is chosen_and_ongoing
    assert winning_end_reason is None
    assert should_stop is True  # not because it resolved - because it wasn't flagged flat


def test_resolve_round_treats_a_resolved_winner_as_authoritative_even_if_flagged_flat():
    # Nonsensical combination in principle (a comparator shouldn't call a
    # resolved ending "flat"), but resolution must win regardless - a cut
    # is never something to keep re-previewing against.
    state = WorldState.initial([])
    results = [(state, EndReason.RESOLVED)]
    comparison = {"best_index": 0, "still_flat": True, "reasoning": "..."}

    _, winning_end_reason, should_stop = _resolve_round(results, comparison)

    assert winning_end_reason == EndReason.RESOLVED
    assert should_stop is True


def test_resolve_round_continues_when_still_flat_and_not_resolved():
    state = WorldState.initial([])
    results = [(state, None)]
    comparison = {"best_index": 0, "still_flat": True, "reasoning": "too tame"}

    _, _, should_stop = _resolve_round(results, comparison)

    assert should_stop is False


# --- _partition_preview_results: resilience to a branch failing even after
# call_structured's own retries (see llm.py/retry.py) - pure, no concurrency ---


def test_partition_preview_results_drops_failures_and_keeps_successes():
    state_a, state_b = WorldState.initial([]), WorldState.initial([])
    raw = [(state_a, None), RuntimeError("branch 2 failed"), (state_b, None)]

    results = _partition_preview_results(raw)

    assert results == [(state_a, None), (state_b, None)]


def test_partition_preview_results_raises_when_every_branch_failed():
    raw = [RuntimeError("first"), RuntimeError("second")]

    with pytest.raises(RuntimeError, match="All 2 preview branches failed"):
        _partition_preview_results(raw)


def test_partition_preview_results_chains_the_first_failure_as_the_cause():
    first_error = RuntimeError("first")
    raw = [first_error, RuntimeError("second")]

    with pytest.raises(RuntimeError) as excinfo:
        _partition_preview_results(raw)

    assert excinfo.value.__cause__ is first_error


# --- Same resilience, exercised end-to-end through simulate() with real
# concurrency, not just the pure function above ---


def test_simulate_tolerates_one_failed_branch_when_others_succeed(small_config, monkeypatch):
    fake = _patch_llm(monkeypatch)
    # Exactly one "direct" call raises; with BRANCH_FACTOR=2 previews this
    # takes down at most one branch, and the round should still complete
    # using whichever branch(es) succeeded.
    fake.direct_script = [RuntimeError("simulated persistent failure")]
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    events, end_reason, rounds = asyncio.run(simulate(session, characters, max_turns=4))

    assert end_reason == EndReason.TURN_BUDGET
    assert len(events) == 4  # completed normally despite the one failed branch


def test_simulate_raises_clearly_when_every_branch_in_a_round_fails(small_config, monkeypatch):
    fake = _patch_llm(monkeypatch)
    fake.direct_script = [RuntimeError("fail 1"), RuntimeError("fail 2")]  # both BRANCH_FACTOR=2 branches' first calls
    session = _session()
    characters = [_character("Ana"), _character("Ben")]

    with pytest.raises(RuntimeError, match="All 2 preview branches failed"):
        asyncio.run(simulate(session, characters, max_turns=4))


# --- _run_turns: the mechanical reveal-enforcement backstop, tested directly
# and synchronously (no concurrency) for precise control over each turn ---


def _setup(char_locations: dict[str, str]):
    session = _session()
    characters = [_character(name, loc) for name, loc in char_locations.items()]
    director = DirectorAgent(session, characters)
    agents = {c.name: CharacterAgent(c, session) for c in characters}
    state = WorldState.initial(characters)
    return state, director, agents


def test_backstop_forces_a_witnessing_character_after_a_director_event(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("simstars.simulation.call_structured", fake)
    # Cleo is in a different room and must never be eligible for the forced
    # reaction - only Ana/Ben, who actually witnessed the Kitchen event.
    state, director, agents = _setup({"Ana": "Kitchen", "Ben": "Kitchen", "Cleo": "Lobby"})
    fake.direct_script = [
        {"action": "event", "location": "Kitchen", "content": "phone rings"},
        {"action": "event", "location": "Kitchen", "content": "phone rings again"},  # director tries another event
    ]

    end_reason = _run_turns(state, director, agents, start_turn=1, num_turns=2, max_turns=2,
                             producer_note=None, prior_feedback=None)

    assert end_reason is None
    assert len(state.events) == 2
    assert state.events[0].type == EventType.DIRECTOR
    # backstop overrode the director's second "event" choice into a
    # character reaction, restricted to witnesses of the first event
    assert state.events[1].type == EventType.DIALOGUE
    assert state.events[1].actor in ("Ana", "Ben")


def test_backstop_does_not_force_a_reaction_when_no_one_witnessed_the_event(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("simstars.simulation.call_structured", fake)
    # Nobody is in "Storage" - the injected event has no witnesses.
    state, director, agents = _setup({"Ana": "Kitchen", "Ben": "Lobby"})
    fake.direct_script = [
        {"action": "event", "location": "Storage", "content": "a crash echoes"},
        {"action": "event", "location": "Kitchen", "content": "another noise"},
    ]

    end_reason = _run_turns(state, director, agents, start_turn=1, num_turns=2, max_turns=2,
                             producer_note=None, prior_feedback=None)

    assert end_reason is None
    # backstop had nothing to force to (no witnesses), so the director's
    # second choice - another event - stands unmodified
    assert state.events[1].type == EventType.DIRECTOR


def test_backstop_treats_a_global_event_as_witnessed_by_everyone(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("simstars.simulation.call_structured", fake)
    state, director, agents = _setup({"Ana": "Kitchen", "Ben": "Lobby"})
    fake.direct_script = [
        {"action": "event", "location": "global", "content": "the power goes out"},
        {"action": "event", "location": "global", "content": "it comes back"},
    ]

    end_reason = _run_turns(state, director, agents, start_turn=1, num_turns=2, max_turns=2,
                             producer_note=None, prior_feedback=None)

    assert end_reason is None
    assert state.events[1].type == EventType.DIALOGUE
    assert state.events[1].actor in ("Ana", "Ben")  # both are eligible - global reaches everyone


def test_unrecognized_character_name_falls_back_instead_of_crashing(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr("simstars.simulation.call_structured", fake)
    state, director, agents = _setup({"Ana": "Kitchen"})
    fake.direct_script = [{"action": "character", "character_name": "NoSuchPerson"}]

    end_reason = _run_turns(state, director, agents, start_turn=1, num_turns=1, max_turns=1,
                             producer_note=None, prior_feedback=None)

    assert end_reason is None
    assert len(state.events) == 1
    assert state.events[0].actor == "Ana"  # fell back to the only real character instead of crashing
