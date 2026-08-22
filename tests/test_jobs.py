"""Job status transitions, tested by calling the worker functions directly
(synchronously) rather than through the real ThreadPoolExecutor - avoids
timing/threading flakiness while still exercising the actual status-update
logic those workers run in the background."""

from simstars import jobs, pipeline
from simstars.models import JobKind, JobStatus, Run


def _create_job_row(session_id: str, kind: JobKind):
    from simstars.db import get_session
    from simstars.models import Job

    job = Job(session_id=session_id, kind=kind)
    with get_session() as db:
        db.add(job)
        db.commit()
        db.refresh(job)
    return job


def test_submit_job_creates_a_pending_row_immediately(temp_db, monkeypatch):
    # Prevent the real background thread from running during this test -
    # only the immediate row-creation behavior is under test here.
    monkeypatch.setattr(jobs, "_executor", type("Noop", (), {"submit": staticmethod(lambda *a, **k: None)})())

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
