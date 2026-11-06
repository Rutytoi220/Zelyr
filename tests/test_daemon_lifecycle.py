"""Unit test suite for deterministic async and synchronous daemon teardown semantics."""

import asyncio
from unittest.mock import AsyncMock
import pytest

from zelyr.memory.pruner import MemoryPrunerDaemon


class DummyStore:
    def __init__(self):
        self.db_path = ":memory:"


class TestDaemonLifecycle:
    @pytest.mark.asyncio
    async def test_shutdown_terminates_task_and_releases_resources(self):
        """await daemon.shutdown() cleanly terminates the task, joins, and releases resources."""
        store = DummyStore()
        daemon = MemoryPrunerDaemon(memory_store=store)

        daemon.start()
        assert daemon._is_running is True
        assert daemon._task is not None
        assert not daemon._task.done()

        # Attach mock resource to simulate held database connection
        mock_db = AsyncMock()
        daemon._active_db = mock_db

        await daemon.shutdown(timeout=2.0)

        assert daemon._is_running is False
        assert daemon._task is None
        assert daemon._active_db is None
        mock_db.execute.assert_awaited_with("PRAGMA wal_checkpoint(PASSIVE)")
        mock_db.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_timeout_raises_and_retains_running_task_and_resources(self):
        """Timeout during shutdown() raises TimeoutError, retains running task reference, and keeps resources unclosed."""
        store = DummyStore()
        daemon = MemoryPrunerDaemon(memory_store=store)
        daemon.start()

        # Replace task with an uncooperative coroutine that delays past the shutdown timeout
        async def uncooperative_loop():
            try:
                while True:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                await asyncio.sleep(2.0)  # Ignore immediate cancellation and block

        uncoop_task = asyncio.create_task(uncooperative_loop())
        await asyncio.sleep(0.02)
        daemon._task = uncoop_task

        mock_db = AsyncMock()
        daemon._active_db = mock_db

        try:
            with pytest.raises(TimeoutError, match="Daemon shutdown timed out after"):
                await daemon.shutdown(timeout=0.1)

            # Task reference must be retained and not cleared
            assert daemon._task is uncoop_task
            assert not uncoop_task.done()

            # Resources must NOT be closed while task is still running
            assert daemon._active_db is mock_db
            mock_db.close.assert_not_called()
        finally:
            # Clean up uncooperative task so test runner does not hang
            uncoop_task.cancel()
            try:
                await asyncio.wait_for(uncoop_task, timeout=0.2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    @pytest.mark.asyncio
    async def test_close_in_running_event_loop_raises_runtime_error(self):
        """Calling daemon.close() inside a running event loop raises RuntimeError."""
        store = DummyStore()
        daemon = MemoryPrunerDaemon(memory_store=store)
        daemon.start()

        with pytest.raises(RuntimeError, match="Cannot call synchronous close\\(\\) while an event loop is running"):
            daemon.close()

        await daemon.shutdown()

    def test_close_with_no_running_loop_completes_synchronously(self):
        """Calling daemon.close() with no running event loop completes shutdown synchronously."""
        store = DummyStore()
        daemon = MemoryPrunerDaemon(memory_store=store)

        # In a synchronous environment with no event loop, start() safely avoids creating dangling unawaited tasks
        daemon.start()
        assert daemon._is_running is False
        assert daemon._task is None

        # Synchronous close runs to completion without raising errors
        daemon.close()
        assert daemon._is_running is False
        assert daemon._task is None
