"""Sync-over-async bridge for Zelyr.

Provides a single correct way to call async code from synchronous
contexts.  Uses a dedicated background thread with a persistent
event loop so that aiosqlite connections (which are loop-bound and
thread-bound) work correctly.

Why not ``asyncio.run()``?
    ``asyncio.run()`` raises ``RuntimeError`` when called from inside
    an already-running event loop.

Why not ``ThreadPoolExecutor`` + ``asyncio.run()``?
    aiosqlite connections are pinned to the thread and event loop
    they were created on.  Spawning a fresh loop in a worker thread
    would use a foreign loop, causing ``sqlite3.ProgrammingError``
    or silent data corruption.

This module solves both problems by running a **single** background
event loop on a **single** daemon thread.  All sync callers submit
coroutines to that loop via ``run_coroutine_threadsafe`` and block
on the resulting ``Future``.

.. warning::

   ``run_sync()`` must NOT be called from code already executing on
   the bridge loop's thread (i.e. from inside a coroutine that was
   itself submitted via ``run_sync()``).  Doing so would deadlock
   because the loop cannot process the inner coroutine while the
   outer one is still occupying it.  A ``RuntimeError`` is raised
   if this is detected.
"""

import asyncio
import threading

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Start the background loop thread exactly once (thread-safe)."""
    global _loop, _thread
    if _loop is not None and _loop.is_running():
        return _loop
    with _lock:
        # Double-check after acquiring lock.
        if _loop is not None and _loop.is_running():
            return _loop
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(
            target=_loop.run_forever,
            daemon=True,
            name="zelyr-async-bridge",
        )
        _thread.start()
    return _loop


def run_sync(coro):
    """Run an async coroutine from synchronous code.

    Submits *coro* to the dedicated background event loop and blocks
    the calling thread until it completes.  Safe to call whether or
    not an event loop is already running in the calling thread.

    Returns whatever the coroutine returns.  Propagates any exception
    the coroutine raises.

    Raises ``RuntimeError`` if called from the bridge thread itself
    (re-entrant deadlock protection).
    """
    loop = _ensure_loop()
    if threading.current_thread() is _thread:
        raise RuntimeError(
            "run_sync() called from the bridge thread — this would deadlock. "
            "Use 'await' directly instead."
        )
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def shutdown_bridge() -> None:
    """Stop the background loop.  Call during application teardown.

    Must be called *after* all consumers (e.g. ``SyncMemoryStore.close()``)
    have finished submitting work.  Calling it while a ``run_sync()`` call
    is in-flight on another thread will stop the loop mid-coroutine;
    the ``future.result()`` in that thread will raise
    ``concurrent.futures.CancelledError``.
    """
    global _loop, _thread
    if _loop is not None:
        _loop.call_soon_threadsafe(_loop.stop)
        if _thread is not None:
            _thread.join(timeout=5)
        _loop = None
        _thread = None
