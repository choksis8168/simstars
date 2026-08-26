"""generate_outline() (see outline.py) - the pre-generation beat sketch fed
to DirectorAgent. Mocks call_structured entirely - no real API calls."""

from simstars.models import Character, Session
from simstars.outline import generate_outline


def _session() -> Session:
    return Session(world_description="A closing bookshop", locations="Shop Floor,Back Office", forcing_mechanic="the shop closes tonight")


def _character(name: str) -> Character:
    return Character(
        session_id="s1", name=name, role="owner", traits="stubborn", starting_location="Shop Floor",
        secret="secretly broke", wound="an old betrayal", hidden_goal="wants to be forgiven",
        relationship_seeds="raised the other character",
    )


def test_generate_outline_joins_beats_into_one_string(monkeypatch):
    monkeypatch.setattr(
        "simstars.outline.call_structured",
        lambda **kwargs: {"beats": ["Ana's debt surfaces", "Ben confronts her", "they reconcile"]},
    )

    outline = generate_outline(_session(), [_character("Ana"), _character("Ben")])

    assert outline == "- Ana's debt surfaces\n- Ben confronts her\n- they reconcile"


def test_generate_outline_includes_hidden_material_and_producer_note_in_the_prompt(monkeypatch):
    captured = {}

    def fake_call_structured(**kwargs):
        captured["user"] = kwargs["user"]
        return {"beats": ["a beat"]}

    monkeypatch.setattr("simstars.outline.call_structured", fake_call_structured)

    generate_outline(_session(), [_character("Ana")], producer_note="make it darker")

    assert "secretly broke" in captured["user"]
    assert "wants to be forgiven" in captured["user"]
    assert "make it darker" in captured["user"]


def test_generate_outline_omits_producer_note_section_when_none_given(monkeypatch):
    captured = {}

    def fake_call_structured(**kwargs):
        captured["user"] = kwargs["user"]
        return {"beats": ["a beat"]}

    monkeypatch.setattr("simstars.outline.call_structured", fake_call_structured)

    generate_outline(_session(), [_character("Ana")])

    assert "Producer's note" not in captured["user"]
