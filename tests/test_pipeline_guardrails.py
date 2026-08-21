"""Scope guardrails (docs/design.md) should reject bad input before any
API calls are made - these raise from pure validation at the top of
new_session, so no mocking is needed."""

import pytest

from simstars.config import MAX_CHARACTERS, MAX_LOCATIONS, MIN_CHARACTERS, MIN_LOCATIONS
from simstars.pipeline import CharacterSpec, _validate_session_input, new_session


def _spec(name: str, location: str) -> CharacterSpec:
    return CharacterSpec(name, "role", "traits", location)


def test_rejects_too_few_characters():
    with pytest.raises(ValueError, match="Cast must have"):
        new_session("world", ["L1", "L2"], [_spec("A", "L1"), _spec("B", "L1")])


def test_rejects_too_many_characters():
    specs = [_spec(n, "L1") for n in "ABCDEF"]
    with pytest.raises(ValueError, match="Cast must have"):
        new_session("world", ["L1", "L2"], specs)


def test_rejects_too_few_locations():
    specs = [_spec(n, "L1") for n in "ABC"]
    with pytest.raises(ValueError, match="World must have"):
        new_session("world", ["L1"], specs)


def test_rejects_unknown_starting_location():
    specs = [_spec("A", "L1"), _spec("B", "L1"), _spec("C", "Nowhere")]
    with pytest.raises(ValueError, match="starting location"):
        new_session("world", ["L1", "L2"], specs)


# --- boundary conditions: exactly MIN/MAX must be *accepted*, not just
# one-off-either-side rejected. Classic off-by-one territory (< vs <=). ---


def test_accepts_exactly_min_characters():
    locations = [f"L{i}" for i in range(MIN_LOCATIONS)]
    specs = [_spec(f"C{i}", locations[0]) for i in range(MIN_CHARACTERS)]
    _validate_session_input("world", locations, specs)  # must not raise


def test_accepts_exactly_max_characters():
    locations = [f"L{i}" for i in range(MIN_LOCATIONS)]
    specs = [_spec(f"C{i}", locations[0]) for i in range(MAX_CHARACTERS)]
    _validate_session_input("world", locations, specs)  # must not raise


def test_accepts_exactly_min_locations():
    locations = [f"L{i}" for i in range(MIN_LOCATIONS)]
    specs = [_spec(f"C{i}", locations[0]) for i in range(MIN_CHARACTERS)]
    _validate_session_input("world", locations, specs)  # must not raise


def test_accepts_exactly_max_locations():
    locations = [f"L{i}" for i in range(MAX_LOCATIONS)]
    specs = [_spec(f"C{i}", locations[0]) for i in range(MIN_CHARACTERS)]
    _validate_session_input("world", locations, specs)  # must not raise


def test_rejects_starting_location_with_different_case_or_whitespace():
    # not just "unknown" - a near-miss (whitespace/case) should still fail
    # rather than silently matching, since Character.starting_location is
    # compared for exact membership in `locations` elsewhere in the engine.
    specs = [_spec("A", " L1"), _spec("B", "L1"), _spec("C", "l1")]
    with pytest.raises(ValueError, match="starting location"):
        _validate_session_input("world", ["L1", "L2"], specs)
