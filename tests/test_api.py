"""API route tests via FastAPI's TestClient, against an isolated temp DB.
The hidden-enrichment-boundary tests are the highest-value ones here - see
api.py's module docstring: an accidental switch back to returning ORM
objects directly would silently leak secret/wound/hidden_goal into the
browser, and that's exactly the kind of regression a fast unit test should
catch immediately rather than relying on someone noticing it in the UI."""

import pytest
from fastapi.testclient import TestClient

from simstars import jobs as jobs_module
from simstars.api import app
from simstars.db import get_session
from simstars.models import Character, EndReason, JobKind, JobStatus, Run, Session

HIDDEN_FIELDS = {"secret", "wound", "hidden_goal", "relationship_seeds"}


@pytest.fixture
def client(temp_db):
    return TestClient(app)


def _seed_session(with_secrets: bool = True) -> Session:
    session = Session(world_description="A test world", locations="Kitchen, Lobby", forcing_mechanic="stuck")
    with get_session() as db:
        db.add(session)
        db.commit()
        db.refresh(session)
        char = Character(
            session_id=session.id,
            name="Ana",
            role="tester",
            traits="curious",
            starting_location="Kitchen",
            voice_id="voice-123",
            secret="Ana is secretly the one who did it" if with_secrets else None,
            wound="an old betrayal" if with_secrets else None,
            hidden_goal="wants to expose the truth" if with_secrets else None,
            relationship_seeds="knows everyone" if with_secrets else None,
        )
        db.add(char)
        db.commit()
        db.refresh(session)
        len(session.characters)
        return session


def _assert_no_hidden_fields(obj) -> None:
    """Recursively checks a JSON-decoded response for any of the hidden
    field names, at any nesting depth/key - not just at the top level of a
    character dict, to catch a leak regardless of exactly how it happened."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert key not in HIDDEN_FIELDS, f"hidden field '{key}' leaked into the API response"
            _assert_no_hidden_fields(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_hidden_fields(item)


def test_list_sessions_never_leaks_hidden_character_fields(client):
    _seed_session(with_secrets=True)
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    _assert_no_hidden_fields(resp.json())


def test_session_detail_never_leaks_hidden_character_fields(client):
    session = _seed_session(with_secrets=True)
    resp = client.get(f"/api/sessions/{session.id}")
    assert resp.status_code == 200
    _assert_no_hidden_fields(resp.json())
    # sanity: the field really is populated in the DB, so this is a
    # meaningful assertion and not just testing an empty value
    with get_session() as db:
        char = db.get(Character, session.characters[0].id)
        assert char.secret is not None


def test_session_detail_includes_public_character_fields(client):
    session = _seed_session(with_secrets=True)
    resp = client.get(f"/api/sessions/{session.id}")
    body = resp.json()
    assert body["characters"][0]["name"] == "Ana"
    assert body["characters"][0]["voice_id"] == "voice-123"


def test_get_session_404_for_unknown_id(client):
    resp = client.get("/api/sessions/does-not-exist")
    assert resp.status_code == 404


def test_create_session_rejects_bad_input_with_400(client, monkeypatch):
    # too few characters - pipeline._validate_session_input should raise
    # ValueError, and the route should turn that into a 400, not a 500.
    body = {
        "world_description": "world",
        "locations": ["L1", "L2"],
        "characters": [{"name": "A", "role": "r", "traits": "t", "starting_location": "L1"}],
    }
    resp = client.post("/api/sessions", json=body)
    assert resp.status_code == 400


def test_generate_endpoint_404s_for_unknown_session(client):
    resp = client.post("/api/sessions/does-not-exist/generate", json={})
    assert resp.status_code == 404


def test_generate_endpoint_submits_a_job_and_returns_it(client, monkeypatch):
    session = _seed_session()
    submitted = {}

    def fake_submit_job(session_id, kind, note=None):
        submitted["args"] = (session_id, kind, note)
        from simstars.models import Job

        return Job(id="job-1", session_id=session_id, kind=kind, status=JobStatus.PENDING)

    monkeypatch.setattr(jobs_module, "submit_job", fake_submit_job)

    resp = client.post(f"/api/sessions/{session.id}/generate", json={"note": "make it tense"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "job-1"
    assert body["status"] == "pending"
    assert submitted["args"] == (session.id, JobKind.GENERATE, "make it tense")


def test_get_run_returns_parsed_screenplay_and_audio_url(client):
    session = _seed_session()
    run = Run(
        session_id=session.id,
        end_reason=EndReason.RESOLVED,
        critic_attempts=1,
        screenplay_json='{"scenes": [{"location": "Kitchen", "heading": "INT. KITCHEN", "lines": ["ANA: hi"], "events": [], "sfx_cues": [], "music_cue": null}]}',
        final_audio_path="/tmp/fake_movie.mp3",
    )
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    resp = client.get(f"/api/runs/{run.id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scenes"][0]["heading"] == "INT. KITCHEN"
    assert body["scenes"][0]["lines"] == ["ANA: hi"]
    assert body["audio_url"] == f"/api/runs/{run.id}/audio"
    assert body["end_reason"] == "resolved"


def test_get_run_audio_url_is_none_without_a_produced_file(client):
    session = _seed_session()
    run = Run(session_id=session.id, screenplay_json='{"scenes": []}')
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    resp = client.get(f"/api/runs/{run.id}")
    assert resp.json()["audio_url"] is None


def test_get_run_audio_404s_when_file_path_missing(client):
    session = _seed_session()
    run = Run(session_id=session.id)
    with get_session() as db:
        db.add(run)
        db.commit()
        db.refresh(run)

    resp = client.get(f"/api/runs/{run.id}/audio")
    assert resp.status_code == 404


def test_frontend_catchall_503s_when_not_built(client):
    # frontend/dist won't exist in the test environment - confirms the
    # helpful-error path rather than an opaque crash.
    resp = client.get("/some/deep/link")
    assert resp.status_code == 503
