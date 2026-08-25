"""Job status transitions, tested by calling the worker functions directly
(synchronously) rather than through the real ThreadPoolExecutor - avoids
timing/threading flakiness while still exercising the actual status-update
logic those workers run in the background."""

from simstars import jobs, pipeline
from simstars.models import JobKind, JobStatus, Run


def _ensure_session(session_id: str) -> None:
    """Job.session_id is a real FK (see models.py) - Postgres enforces it,
    unlike SQLite's default lax behavior these fixtures used to rely on -
    so any test using a bare session id string needs a Session row to
    actually back it."""
    from simstars.db import get_session
    from simstars.models import Session

    with get_session() as db:
        if db.get(Session, session_id) is None:
            db.add(Session(id=session_id, world_description="test world", locations="Kitchen,Lobby", forcing_mechanic="stuck"))
            db.commit()


def _create_job_row(session_id: str, kind: JobKind, created_at=None):
    from simstars.db import get_session
    from simstars.models import Job

    _ensure_session(session_id)
    job = Job(session_id=session_id, kind=kind)
    if created_at is not None:
        job.created_at = created_at
    with get_session() as db:
        db.add(job)
        db.commit()
        db.refresh(job)
    return job


def test_submit_job_creates_a_pending_row_immediately(temp_db, monkeypatch):
    # Prevent the real background thread from running during this test -
    # only the immediate row-creation behavior is under test here.
    monkeypatch.setattr(jobs, "_executor", type("Noop", (), {"submit": staticmethod(lambda *a, **k: None)})())
    _ensure_session("session-1")

    job = jobs.submit_job("session-1", JobKind.GENERATE, note="be dramatic")

    assert job.status == JobStatus.PENDING
    assert job.session_id == "session-1"
    assert job.kind == JobKind.GENERATE
    assert jobs.get_job(job.id) is not None


def test_run_generate_marks_complete_and_records_the_run_id(temp_db, monkeypatch):
    job = _create_job_row("session-1", JobKind.GENERATE)
    fake_run = Run(id="run-1", session_id="session-1")
    monkeypatch.setattr(pipeline, "generate", lambda session_id, note: (fake_run, [], None))

    jobs._run_generate(job.id, "session-1", note=None)

    updated = jobs.get_job(job.id)
    assert updated.status == JobStatus.COMPLETE
    assert updated.result_run_id == "run-1"
    assert updated.error_message is None


def test_run_play_marks_complete_and_records_the_run_id(temp_db, monkeypatch):
    job = _create_job_row("session-1", JobKind.PLAY)
    fake_run = Run(id="run-2", session_id="session-1")
    monkeypatch.setattr(pipeline, "play", lambda session_id, note: fake_run)

    jobs._run_play(job.id, "session-1", note="darker ending")

    updated = jobs.get_job(job.id)
    assert updated.status == JobStatus.COMPLETE
    assert updated.result_run_id == "run-2"


def test_run_generate_marks_failed_with_error_message_on_exception(temp_db, monkeypatch):
    job = _create_job_row("session-1", JobKind.GENERATE)

    def boom(session_id, note):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(pipeline, "generate", boom)

    jobs._run_generate(job.id, "session-1", note=None)

    updated = jobs.get_job(job.id)
    assert updated.status == JobStatus.FAILED
    assert updated.error_message == "ANTHROPIC_API_KEY is not set."
    assert updated.result_run_id is None


def test_run_play_marks_failed_with_error_message_on_exception(temp_db, monkeypatch):
    job = _create_job_row("session-1", JobKind.PLAY)

    def boom(session_id, note):
        raise RuntimeError("credit balance too low")

    monkeypatch.setattr(pipeline, "play", boom)

    jobs._run_play(job.id, "session-1", note=None)

    updated = jobs.get_job(job.id)
    assert updated.status == JobStatus.FAILED
    assert updated.error_message == "credit balance too low"


def test_status_passes_through_running_before_terminal_state(temp_db, monkeypatch):
    job = _create_job_row("session-1", JobKind.GENERATE)
    seen_statuses = []

    real_set_status = jobs._set_status

    def spy(job_id, **fields):
        if "status" in fields:
            seen_statuses.append(fields["status"])
        real_set_status(job_id, **fields)

    monkeypatch.setattr(jobs, "_set_status", spy)
    monkeypatch.setattr(pipeline, "generate", lambda session_id, note: (Run(id="r", session_id="session-1"), [], None))

    jobs._run_generate(job.id, "session-1", note=None)

    assert seen_statuses == [JobStatus.RUNNING, JobStatus.COMPLETE]


def test_list_jobs_returns_newest_first_and_scoped_to_the_session(temp_db):
    # Regression: a failed job's error was previously only ever visible to
    # whoever was watching the page at the moment it finished - nothing
    # persisted it anywhere a user could come back and see it later.
    from datetime import datetime, timedelta, timezone

    base = datetime.now(timezone.utc)
    older = _create_job_row("session-1", JobKind.GENERATE, created_at=base)
    newer = _create_job_row("session-1", JobKind.PLAY, created_at=base + timedelta(seconds=10))
    _create_job_row("other-session", JobKind.GENERATE, created_at=base + timedelta(seconds=20))

    result = jobs.list_jobs("session-1")

    assert [j.id for j in result] == [newer.id, older.id]
    assert all(j.session_id == "session-1" for j in result)


def test_list_jobs_includes_failed_status_and_error(temp_db):
    job = _create_job_row("session-1", JobKind.PLAY)
    jobs._set_status(job.id, status=JobStatus.FAILED, error_message="concurrent_limit_exceeded")

    result = jobs.list_jobs("session-1")

    assert result[0].status == JobStatus.FAILED
    assert result[0].error_message == "concurrent_limit_exceeded"


def test_submit_produce_job_pre_populates_result_run_id(temp_db, monkeypatch):
    # Unlike generate/play, a produce job's target run is known up front,
    # not just discovered on completion - see submit_produce_job()'s docstring.
    monkeypatch.setattr(jobs, "_executor", type("Noop", (), {"submit": staticmethod(lambda *a, **k: None)})())
    _ensure_session("session-1")

    job = jobs.submit_produce_job("run-1", "session-1")

    assert job.kind == JobKind.PRODUCE
    assert job.status == JobStatus.PENDING
    assert job.result_run_id == "run-1"


def test_run_produce_marks_complete_on_success(temp_db, monkeypatch):
    job = _create_job_row("session-1", JobKind.PRODUCE)
    fake_run = Run(id="run-1", session_id="session-1", final_audio_path="/tmp/movie.mp3")
    monkeypatch.setattr(pipeline, "produce_run", lambda run_id: fake_run)

    jobs._run_produce(job.id, "run-1")

    updated = jobs.get_job(job.id)
    assert updated.status == JobStatus.COMPLETE
    assert updated.result_run_id == "run-1"


def test_run_produce_marks_failed_with_error_message_on_exception(temp_db, monkeypatch):
    job = _create_job_row("session-1", JobKind.PRODUCE)

    def boom(run_id):
        raise RuntimeError("concurrent_limit_exceeded")

    monkeypatch.setattr(pipeline, "produce_run", boom)

    jobs._run_produce(job.id, "run-1")

    updated = jobs.get_job(job.id)
    assert updated.status == JobStatus.FAILED
    assert updated.error_message == "concurrent_limit_exceeded"
