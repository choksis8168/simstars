"""Shared fixtures. `temp_db` gives tests an isolated Postgres database
(created fresh, dropped afterward) instead of touching the real local
`simstars` database - needed by anything that actually persists (jobs/API
tests), unlike the rest of the suite which stays pure-logic/mocked and
never touches the DB at all."""

import uuid

import pytest
from sqlalchemy import create_engine, text

import simstars.db as db_module
from simstars.config import DATABASE_URL


def _with_db_name(url: str, db_name: str) -> str:
    base, _, _ = url.rpartition("/")
    return f"{base}/{db_name}"


def _admin_url() -> str:
    # CREATE DATABASE/DROP DATABASE can't run inside a transaction against
    # the database being created/dropped - "postgres" is always present to
    # connect through instead.
    return _with_db_name(DATABASE_URL, "postgres")


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_name = f"simstars_test_{uuid.uuid4().hex[:12]}"
    admin_engine = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    admin_engine.dispose()

    monkeypatch.setattr(db_module, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(db_module, "DATABASE_URL", _with_db_name(DATABASE_URL, db_name))
    monkeypatch.setattr(db_module, "_engine", None)

    yield

    engine = db_module._engine
    if engine is not None:
        engine.dispose()
    monkeypatch.setattr(db_module, "_engine", None)

    admin_engine = create_engine(_admin_url(), isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
    admin_engine.dispose()
