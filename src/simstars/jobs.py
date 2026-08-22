"""Background execution for generate()/play(), which can take minutes - an
HTTP request can't just block that long. Deliberately not a task queue
(Celery/RQ/Redis): this is a single-user local app, so a module-level
thread pool is enough - see docs/design.md web-app plan "Background jobs,
not a task queue".

A `Job` row is created (status=pending) before any work starts, so the
frontend has something to poll immediately; a worker thread flips it to
running, then complete (with the resulting Run's id) or failed (with the
exception message) - never left hanging or silently lost.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from simstars import pipeline
from simstars.db import get_session
from simstars.models import Job, JobKind, JobStatus

# Small and fixed: this is a single local user, not a server under load -
# a couple of concurrent jobs (e.g. generating for two sessions at once)
# is plenty; unbounded growth isn't a real risk here.
_executor = ThreadPoolExecutor(max_workers=4)


def _set_status(job_id: str, **fields) -> None:
    with get_session() as db:
        job = db.get(Job, job_id)
        if job is None:
            return  # job row was somehow removed - nothing to update
        for key, value in fields.items():
            setattr(job, key, value)
        db.add(job)
        db.commit()


def _run_generate(job_id: str, session_id: str, note: str | None) -> None:
    _set_status(job_id, status=JobStatus.RUNNING)
    try:
        run, _events, _screenplay = pipeline.generate(session_id, note)
    except Exception as exc:  # noqa: BLE001 - any failure surfaces to the poller, never crashes silently
        _set_status(job_id, status=JobStatus.FAILED, error_message=str(exc))
        return
    _set_status(job_id, status=JobStatus.COMPLETE, result_run_id=run.id)


def _run_play(job_id: str, session_id: str, note: str | None) -> None:
    _set_status(job_id, status=JobStatus.RUNNING)
    try:
        run = pipeline.play(session_id, note)
    except Exception as exc:  # noqa: BLE001
        _set_status(job_id, status=JobStatus.FAILED, error_message=str(exc))
        return
    _set_status(job_id, status=JobStatus.COMPLETE, result_run_id=run.id)


_WORKERS = {
    JobKind.GENERATE: _run_generate,
    JobKind.PLAY: _run_play,
}


def get_job(job_id: str) -> Job | None:
    with get_session() as db:
        return db.get(Job, job_id)


def submit_job(session_id: str, kind: JobKind, note: str | None = None) -> Job:
    """Creates the Job row immediately (status=pending) and hands the
    actual pipeline call to the thread pool - returns right away, before
    any real work has necessarily started, so the caller always has a job
    id to poll against.
    """
    job = Job(session_id=session_id, kind=kind)
    with get_session() as db:
        db.add(job)
        db.commit()
        db.refresh(job)

    _executor.submit(_WORKERS[kind], job.id, session_id, note)
    return job
