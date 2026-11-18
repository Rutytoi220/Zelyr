"""Standardized SQLite Connection Factory, Concurrency WAL Hygiene, and Safe Transaction Management.

Provides centralized SQLite connection lifecycle control:
- Resolves database paths via Zelyr config hierarchy (never defaulting to repo root).
- Enforces concurrency PRAGMAs (WAL mode, busy_timeout=5000, synchronous=NORMAL, foreign_keys=ON).
- Provides atomic transaction management (managed_transaction) with automatic commit/rollback.
- Provides clean teardown helpers (checkpoint_and_close) to flush WAL journals.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union

from zelyr.compat import get_env
from zelyr.config import get_config_dir

logger = logging.getLogger(__name__)


def get_default_db_path(name: str = "memory.db") -> Path:
    """Resolve the default SQLite database path.

    Precedence:
    1. Environment override: ``ZELYR_MEMORY_DB`` or legacy ``AXIOM_MEMORY_DB`` (if set).
    2. Legacy database reuse in config dir: if name is ``"memory.db"``, reuse existing
       ``zelyr_memory.db`` or ``axiom_memory.db`` if present in ``get_config_dir()``.
    3. User config directory: ``get_config_dir() / name``.

    Guarantees the parent directory exists and NEVER defaults to the repository root.
    """
    env_path = get_env("MEMORY_DB")
    if env_path:
        db_path = Path(env_path).expanduser().resolve()
    else:
        config_dir = get_config_dir()
        if name == "memory.db":
            if (config_dir / "zelyr_memory.db").exists():
                db_path = config_dir / "zelyr_memory.db"
            elif (config_dir / "axiom_memory.db").exists():
                db_path = config_dir / "axiom_memory.db"
            else:
                db_path = config_dir / name
        else:
            db_path = config_dir / name

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_db_connection(
    db_path: Optional[Union[str, Path]] = None,
    timeout: float = 5.0,
    row_factory: bool = True,
) -> sqlite3.Connection:
    """Create and return a concurrency-hardened SQLite connection.

    Enforces WAL journal mode, 5000ms busy timeout, NORMAL synchronization,
    and foreign key constraints.
    """
    if db_path is None:
        target_path = str(get_default_db_path())
    elif isinstance(db_path, Path):
        target_path = str(db_path)
    else:
        target_path = str(db_path)

    if target_path != ":memory:" and not target_path.startswith("file:"):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_path, timeout=timeout)
    if row_factory:
        conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def managed_transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Atomic transaction context manager.

    Yields the connection. Commits the transaction on clean exit,
    or rolls back on exception before re-raising.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception as rb_err:
            logger.debug("Failed to rollback SQLite transaction: %s", rb_err)
        raise


def checkpoint_and_close(conn: sqlite3.Connection, mode: str = "PASSIVE") -> None:
    """Safely flush SQLite WAL journal and close the connection.

    Mode must be one of PASSIVE, FULL, RESTART, or TRUNCATE (defaults to PASSIVE).
    Errors during checkpoint or close are safely logged rather than raised.
    """
    valid_modes = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
    norm_mode = mode.upper() if mode and mode.upper() in valid_modes else "PASSIVE"
    try:
        conn.execute(f"PRAGMA wal_checkpoint({norm_mode});")
    except Exception as exc:
        logger.debug("SQLite wal_checkpoint(%s) failed: %s", norm_mode, exc)
    finally:
        try:
            conn.close()
        except Exception as exc:
            logger.debug("SQLite connection close failed: %s", exc)


async def checkpoint_and_close_async(db: Any, mode: str = "PASSIVE") -> None:
    """Safely flush aiosqlite WAL journal and close the async connection."""
    valid_modes = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}
    norm_mode = mode.upper() if mode and mode.upper() in valid_modes else "PASSIVE"
    try:
        await db.execute(f"PRAGMA wal_checkpoint({norm_mode});")
    except Exception as exc:
        logger.debug("aiosqlite wal_checkpoint(%s) failed: %s", norm_mode, exc)
    finally:
        try:
            await db.close()
        except Exception as exc:
            logger.debug("aiosqlite connection close failed: %s", exc)


__all__ = [
    "get_default_db_path",
    "get_db_connection",
    "managed_transaction",
    "checkpoint_and_close",
    "checkpoint_and_close_async",
]
