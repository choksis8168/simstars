"""Shared fixtures. `temp_db` gives tests an isolated SQLite DB instead of
touching the real local sessions/simstars.db - needed by anything that
actually persists (jobs/API tests), unlike the rest of the suite which
stays pure-logic/mocked and never touches the DB at all."""

import pytest

import simstars.db as db_module


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db_module, "_engine", None)
    yield
    monkeypatch.setattr(db_module, "_engine", None)
