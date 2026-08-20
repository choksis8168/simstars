"""SQLite persistence via sqlmodel.

Chosen over flat JSON files specifically so a later web app (concurrent
sessions/multiple users) is a storage swap (SQLite -> Postgres) rather than
a rewrite — see docs/design.md "Persistence".
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlmodel import Session as DBSession
from sqlmodel import SQLModel, create_engine

from moviesim.config import DATA_ROOT, DB_PATH

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{DB_PATH}")
        SQLModel.metadata.create_all(_engine)
    return _engine


@contextmanager
def get_session() -> Iterator[DBSession]:
    with DBSession(get_engine()) as db:
        yield db


def run_dir(session_id: str, run_id: str) -> Path:
    d = DATA_ROOT / session_id / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def audio_dir(session_id: str, run_id: str) -> Path:
    d = run_dir(session_id, run_id) / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d
