"""Comprehensive unit test suite for centralized SQLite connection factory and concurrency hygiene."""

from __future__ import annotations

import concurrent.futures
import sqlite3
import threading
from pathlib import Path
from typing import List

import pytest

from zelyr.db.connection import (
    checkpoint_and_close,
    get_db_connection,
    get_default_db_path,
    managed_transaction,
)


def test_default_db_path_resolves_to_config_dir_not_repo_root(monkeypatch, tmp_path) -> None:
    """Ensure get_default_db_path resolves to user config dir and never repo root."""
    monkeypatch.delenv("ZELYR_MEMORY_DB", raising=False)
    monkeypatch.delenv("AXIOM_MEMORY_DB", raising=False)

    fake_config = tmp_path / "custom_config"
    monkeypatch.setattr("zelyr.db.connection.get_config_dir", lambda: fake_config)

    db_path = get_default_db_path("memory.db")
    assert db_path == fake_config / "memory.db"
    assert db_path.parent.exists()

    # Verify repo root has not been polluted
    repo_root = Path(__file__).resolve().parent.parent
    assert db_path.resolve() != (repo_root / "memory.db").resolve()
    assert db_path.resolve() != (repo_root / "zelyr_memory.db").resolve()
    assert db_path.resolve() != (repo_root / "axiom_memory.db").resolve()


def test_env_var_overrides(monkeypatch, tmp_path) -> None:
    """Ensure ZELYR_MEMORY_DB and AXIOM_MEMORY_DB environment variables override default."""
    legacy_file = tmp_path / "legacy.db"
    new_file = tmp_path / "preferred.db"

    # Legacy AXIOM_MEMORY_DB fallback
    monkeypatch.delenv("ZELYR_MEMORY_DB", raising=False)
    monkeypatch.setenv("AXIOM_MEMORY_DB", str(legacy_file))
    assert get_default_db_path() == legacy_file.resolve()

    # Preferred ZELYR_MEMORY_DB overrides AXIOM
    monkeypatch.setenv("ZELYR_MEMORY_DB", str(new_file))
    assert get_default_db_path() == new_file.resolve()


def test_connection_standard_pragmas(tmp_path) -> None:
    """Verify that get_db_connection sets WAL mode, busy_timeout=5000, synchronous=NORMAL, and foreign_keys=ON."""
    db_file = tmp_path / "test_pragmas.db"
    conn = get_db_connection(db_file)
    try:
        assert conn.row_factory is sqlite3.Row

        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(journal_mode).lower() == "wal"

        busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert busy_timeout == 5000

        synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
        assert synchronous == 1  # 1 == NORMAL

        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert foreign_keys == 1  # 1 == ON
    finally:
        conn.close()


def test_managed_transaction_commit_and_rollback(tmp_path) -> None:
    """Verify that managed_transaction commits on success and rolls back on exception."""
    db_file = tmp_path / "test_tx.db"
    conn = get_db_connection(db_file)
    conn.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, balance REAL);")

    # 1. Commit on clean exit
    with managed_transaction(conn):
        conn.execute("INSERT INTO accounts (id, balance) VALUES (1, 100.0);")

    row = conn.execute("SELECT balance FROM accounts WHERE id = 1;").fetchone()
    assert row is not None
    assert row["balance"] == 100.0

    # 2. Rollback on exception
    with pytest.raises(RuntimeError, match="Transaction aborted"):
        with managed_transaction(conn):
            conn.execute("INSERT INTO accounts (id, balance) VALUES (2, 200.0);")
            raise RuntimeError("Transaction aborted")

    row_aborted = conn.execute("SELECT balance FROM accounts WHERE id = 2;").fetchone()
    assert row_aborted is None

    conn.close()


def test_multithreaded_concurrency_no_database_locked(tmp_path) -> None:
    """Spawn 10 concurrent threads rapidly executing transactions; assert 0 'database is locked' errors."""
    db_file = tmp_path / "concurrent_stress.db"

    # Initialize schema
    init_conn = get_db_connection(db_file)
    init_conn.execute("CREATE TABLE counter (thread_id INTEGER, iteration INTEGER);")
    init_conn.close()

    num_threads = 10
    iterations_per_thread = 25
    errors: List[Exception] = []
    barrier = threading.Barrier(num_threads)

    def worker(thread_idx: int) -> None:
        barrier.wait()
        for i in range(iterations_per_thread):
            try:
                conn = get_db_connection(db_file, timeout=5.0)
                with managed_transaction(conn):
                    conn.execute(
                        "INSERT INTO counter (thread_id, iteration) VALUES (?, ?);",
                        (thread_idx, i),
                    )
                # Quick read verification
                cur = conn.execute(
                    "SELECT count(*) FROM counter WHERE thread_id = ?;",
                    (thread_idx,),
                )
                assert cur.fetchone()[0] >= 1
                conn.close()
            except Exception as ex:
                errors.append(ex)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, idx) for idx in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert len(errors) == 0, f"Encountered locking errors during stress test: {errors}"

    verify_conn = get_db_connection(db_file)
    total_count = verify_conn.execute("SELECT count(*) FROM counter;").fetchone()[0]
    verify_conn.close()
    assert total_count == num_threads * iterations_per_thread


def test_checkpoint_and_close(tmp_path) -> None:
    """Verify that checkpoint_and_close runs PRAGMA wal_checkpoint and closes the handle cleanly."""
    db_file = tmp_path / "checkpoint_test.db"
    conn = get_db_connection(db_file)
    conn.execute("CREATE TABLE items (name TEXT);")
    with managed_transaction(conn):
        conn.execute("INSERT INTO items VALUES ('item1'), ('item2');")

    checkpoint_and_close(conn, mode="TRUNCATE")

    # Connection should be closed
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1;")

    # Reopening should read back all committed data
    reopened = get_db_connection(db_file)
    items = [r["name"] for r in reopened.execute("SELECT name FROM items ORDER BY name;").fetchall()]
    assert items == ["item1", "item2"]
    reopened.close()
