"""Scope guardrails (docs/design.md) should reject bad input before any
API calls are made - these raise from pure validation at the top of
new_session, so no mocking is needed."""

import pytest

from simstars.pipeline import CharacterSpec, new_session


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
