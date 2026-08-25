"""PostgreSQL persistence via sqlmodel.

Chosen over SQLite specifically so the single-process web app's storage
layer already speaks the protocol a later multi-user deployment would need
(concurrent connections, and pgvector for character memory retrieval - see
memory_store.py) rather than a rewrite - see docs/design.md "Persistence".
Defaults to a local Homebrew Postgres instance (no Docker required); see
config.DATABASE_URL.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import text
from sqlmodel import Session as DBSession
from sqlmodel import SQLModel, create_engine

from simstars.config import DATA_ROOT, DATABASE_URL

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(DATABASE_URL)
        with _engine.begin() as conn:
            # Needed once per database before any Vector column (see
            # memory_store.py's MemoryEmbedding table) can be created.
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # create_all() only creates missing TABLES - it never alters an
        # existing table's columns. There's no migration tooling yet (MVP),
        # so adding/renaming a model field requires either dropping the
        # local `simstars` database (loses local test data) or manually
        # `ALTER TABLE ... ADD COLUMN ...` to bring it up to date - hit this
        # for real once already, see git history.
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
