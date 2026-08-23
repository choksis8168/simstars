"""FastAPI app - the web app's backend. Wraps pipeline.py exactly like the
CLI does (see pipeline.py's module docstring); this file holds no engine
logic of its own, only request/response shaping and job orchestration.

Response schemas are hand-written and explicit rather than returning
SQLModel instances directly, for one hard reason: `Character.secret`,
`.wound`, `.hidden_goal`, `.relationship_seeds` must never appear in any
API response - see models.py's Character docstring on the hidden-at-
creation-time boundary. Returning an ORM object directly would serialize
every column via FastAPI's default encoder and leak them by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from simstars import jobs, pipeline
from simstars.models import Character, Job, JobKind, Run, Screenplay, Session

app = FastAPI(title="SimStars")


# --- request schemas ---


class CharacterIn(BaseModel):
    name: str
    role: str
    traits: str
    starting_location: str


class SessionCreateIn(BaseModel):
    world_description: str
    locations: list[str]
    characters: list[CharacterIn]


class NoteIn(BaseModel):
    note: Optional[str] = None


# --- response schemas (public fields only - see module docstring) ---


class CharacterOut(BaseModel):
    id: str
    name: str
    role: str
    traits: str
    starting_location: str
    voice_id: Optional[str] = None

    @classmethod
    def from_model(cls, c: Character) -> "CharacterOut":
        return cls(
            id=c.id,
            name=c.name,
            role=c.role,
            traits=c.traits,
            starting_location=c.starting_location,
            voice_id=c.voice_id,
        )


class SessionOut(BaseModel):
    id: str
    created_at: str
    world_description: str
    forcing_mechanic: Optional[str] = None
    locations: list[str]
    characters: list[CharacterOut]

    @classmethod
    def from_model(cls, s: Session, characters: list[Character]) -> "SessionOut":
        return cls(
            id=s.id,
            created_at=s.created_at.isoformat(),
            world_description=s.world_description,
            forcing_mechanic=s.forcing_mechanic,
            locations=s.location_list(),
            characters=[CharacterOut.from_model(c) for c in characters],
        )


class SceneOut(BaseModel):
    location: str
    heading: str
    lines: list[str]
    sfx_cues: list[str]
    music_cue: Optional[str] = None


class RunOut(BaseModel):
    id: str
    session_id: str
    created_at: str
    producer_note: Optional[str] = None
    end_reason: Optional[str] = None
    critic_attempts: int
    critic_reasoning: Optional[str] = None
    branch_rounds_used: int
    scenes: list[SceneOut]
    audio_url: Optional[str] = None

    @classmethod
    def from_model(cls, r: Run) -> "RunOut":
        screenplay = (
            Screenplay.model_validate_json(r.screenplay_json)
            if r.screenplay_json
            else Screenplay(scenes=[])
        )
        return cls(
            id=r.id,
            session_id=r.session_id,
            created_at=r.created_at.isoformat(),
            producer_note=r.producer_note,
            end_reason=r.end_reason.value if r.end_reason else None,
            critic_attempts=r.critic_attempts,
            critic_reasoning=r.critic_reasoning,
            branch_rounds_used=r.branch_rounds_used,
            scenes=[
                SceneOut(
                    location=sc.location,
                    heading=sc.heading,
                    lines=sc.lines,
                    sfx_cues=sc.sfx_cues,
                    music_cue=sc.music_cue,
                )
                for sc in screenplay.scenes
            ],
            audio_url=f"/api/runs/{r.id}/audio" if r.final_audio_path else None,
        )


class SessionDetailOut(SessionOut):
    runs: list[RunOut]


class JobOut(BaseModel):
    id: str
    session_id: str
    kind: str
    status: str
    error_message: Optional[str] = None
    result_run_id: Optional[str] = None
    created_at: str

    @classmethod
    def from_model(cls, j: Job) -> "JobOut":
        return cls(
            id=j.id,
            session_id=j.session_id,
            kind=j.kind.value,
            status=j.status.value,
            error_message=j.error_message,
            result_run_id=j.result_run_id,
            created_at=j.created_at.isoformat(),
        )


# --- routes ---


@app.post("/api/sessions", response_model=SessionOut)
def create_session(body: SessionCreateIn) -> SessionOut:
    specs = [pipeline.CharacterSpec(c.name, c.role, c.traits, c.starting_location) for c in body.characters]
    try:
        session = pipeline.new_session(body.world_description, body.locations, specs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SessionOut.from_model(session, session.characters)


@app.get("/api/sessions", response_model=list[SessionOut])
def list_sessions_route() -> list[SessionOut]:
    return [SessionOut.from_model(s, s.characters) for s in pipeline.list_sessions()]


@app.get("/api/sessions/{session_id}", response_model=SessionDetailOut)
def get_session_route(session_id: str) -> SessionDetailOut:
    detail = pipeline.get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session, characters, runs = detail
    base = SessionOut.from_model(session, characters)
    return SessionDetailOut(**base.model_dump(), runs=[RunOut.from_model(r) for r in runs])


@app.post("/api/sessions/{session_id}/generate", response_model=JobOut)
def start_generate(session_id: str, body: NoteIn) -> JobOut:
    if pipeline.get_session_detail(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JobOut.from_model(jobs.submit_job(session_id, JobKind.GENERATE, body.note))


@app.post("/api/sessions/{session_id}/play", response_model=JobOut)
def start_play(session_id: str, body: NoteIn) -> JobOut:
    if pipeline.get_session_detail(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JobOut.from_model(jobs.submit_job(session_id, JobKind.PLAY, body.note))


@app.get("/api/jobs/{job_id}", response_model=JobOut)
def get_job_route(job_id: str) -> JobOut:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobOut.from_model(job)


@app.get("/api/sessions/{session_id}/jobs", response_model=list[JobOut])
def list_jobs_route(session_id: str) -> list[JobOut]:
    if pipeline.get_session_detail(session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return [JobOut.from_model(j) for j in jobs.list_jobs(session_id)]


@app.get("/api/runs/{run_id}", response_model=RunOut)
def get_run_route(run_id: str) -> RunOut:
    run = pipeline.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunOut.from_model(run)


@app.get("/api/runs/{run_id}/audio")
def get_run_audio(run_id: str) -> FileResponse:
    run = pipeline.get_run(run_id)
    if run is None or not run.final_audio_path:
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(run.final_audio_path, media_type="audio/mpeg")


@app.post("/api/runs/{run_id}/produce", response_model=JobOut)
def start_produce(run_id: str) -> JobOut:
    """Retries (or runs for the first time) just the PRODUCE phase against
    this run's already-persisted screenplay - see pipeline.produce_run().
    The recovery path when production fails after simulation already
    succeeded, and also how a `generate`-only run can get audio without
    re-rolling the story.
    """
    run = pipeline.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not run.screenplay_json:
        raise HTTPException(status_code=400, detail="This run has no screenplay yet - nothing to produce.")
    return JobOut.from_model(jobs.submit_produce_job(run_id, run.session_id))


# --- static frontend ---
# A hand-written catch-all rather than StaticFiles(html=True): that option
# only serves index.html for directory requests, not for arbitrary
# unmatched paths - so a browser refresh/deep link on a client-side route
# (e.g. /sessions/abc123) would 404 instead of loading the app. Registered
# last so it never shadows the /api/* routes above (FastAPI matches routes
# in registration order).

_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@app.get("/{full_path:path}")
def serve_frontend(full_path: str) -> FileResponse:
    if not _FRONTEND_DIST.is_dir():
        raise HTTPException(
            status_code=503,
            detail="Frontend isn't built yet - run `npm install && npm run build` in frontend/.",
        )
    dist_root = _FRONTEND_DIST.resolve()
    candidate = (dist_root / full_path).resolve()
    if candidate.is_file() and dist_root in candidate.parents:
        return FileResponse(candidate)
    return FileResponse(dist_root / "index.html")
