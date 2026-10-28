"""Tests for Over-The-Air (OTA) update verification, atomic rollback, and daemon worker supervision."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import sys
import time
import unittest.mock
import pytest

from zelyr.core.events import EventBus
from zelyr.server.daemon import DaemonWorkerSupervisor, WorkerState
from zelyr.services.ota_updater import (
    OTAUpdateEngine,
    compute_sha256,
    validate_release_manifest,
    verify_sha256,
)


class TestOTAUpdateIntegrityAndRollback:
    """Verifies cryptographic validation, atomic staging, and health-check rollbacks."""

    def test_release_manifest_schema_validation(self) -> None:
        """Assert release manifest requires non-empty tag_name and valid assets."""
        valid_manifest = {
            "tag_name": "v1.1.0",
            "assets": [{"name": "zelyr-1.1.0.tar.gz", "browser_download_url": "https://example.com/asset"}],
            "body": "Release v1.1.0 notes",
        }
        res = validate_release_manifest(valid_manifest)
        assert res["tag_name"] == "v1.1.0"

        with pytest.raises(ValueError, match="Release manifest must be a valid JSON dictionary"):
            validate_release_manifest("not-a-dict")

        with pytest.raises(ValueError, match="missing mandatory non-empty 'tag_name'"):
            validate_release_manifest({"tag_name": "", "assets": []})

    def test_sha256_verification_and_computation(self, tmp_path: Path) -> None:
        """Assert compute_sha256 and verify_sha256 accurately evaluate file integrity."""
        test_file = tmp_path / "artifact.bin"
        test_file.write_bytes(b"hello zelyr binary package")

        digest = compute_sha256(test_file)
        assert len(digest) == 64
        assert verify_sha256(test_file, digest) is True
        assert verify_sha256(test_file, "0" * 64) is False

    @pytest.mark.asyncio
    async def test_download_asset_with_corrupt_checksum_aborts(self, tmp_path: Path) -> None:
        """Assert download_asset deletes corrupt download and raises ValueError on sha256 mismatch."""
        engine = OTAUpdateEngine(base_dir=tmp_path / "updates")
        fake_content = b"fake release data"
        corrupt_hash = "f" * 64

        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.side_effect = [fake_content, b""]
        mock_resp.__enter__.return_value = mock_resp

        with unittest.mock.patch("zelyr.services.ota_updater.urlopen", return_value=mock_resp):
            with pytest.raises(ValueError, match="Cryptographic verification failed"):
                await engine.download_asset(
                    "https://example.com/asset.tar.gz",
                    "asset.tar.gz",
                    expected_sha256=corrupt_hash,
                )

        assert not (engine.staging_dir / "asset.tar.gz").exists()

    @pytest.mark.asyncio
    async def test_successful_update_stages_swaps_and_promotes_version(self, tmp_path: Path) -> None:
        """Assert update with valid sha256 and successful health check promotes version cleanly."""
        install_dir = tmp_path / "installed_app"
        install_dir.mkdir()
        (install_dir / "version.txt").write_text("1.0.0")

        engine = OTAUpdateEngine(
            install_dir=install_dir,
            base_dir=tmp_path / "updates",
            current_version="1.0.0",
        )

        artifact = tmp_path / "new_payload.bin"
        artifact.write_bytes(b"new binary content v1.1.0")
        artifact_hash = compute_sha256(artifact)

        staging_dir = engine.stage_update(artifact, "v1.1.0", expected_sha256=artifact_hash)
        assert (staging_dir / "new_payload.bin").exists()

        health_cmd = [sys.executable, "-c", "import sys; sys.exit(0)"]
        result = await engine.apply_update("v1.1.0", staging_dir, health_check_cmd=health_cmd)

        assert result.success is True
        assert result.rolled_back is False
        assert result.stage == "complete"
        assert engine.current_version == "1.1.0"

    @pytest.mark.asyncio
    async def test_failed_health_check_triggers_automatic_rollback(self, tmp_path: Path) -> None:
        """Assert pre-flight health check failure restores previous version backup cleanly."""
        install_dir = tmp_path / "installed_app"
        install_dir.mkdir()
        canary = install_dir / "canary.txt"
        canary.write_text("ORIGINAL_CANARY_STATE")

        engine = OTAUpdateEngine(
            install_dir=install_dir,
            base_dir=tmp_path / "updates",
            current_version="1.0.0",
        )

        artifact_dir = tmp_path / "artifact_source"
        artifact_dir.mkdir()
        (artifact_dir / "canary.txt").write_text("NEW_CRASHING_STATE")

        staging_dir = engine.stage_update(artifact_dir, "v1.1.0")

        # Health check exits with code 1
        health_cmd = [sys.executable, "-c", "import sys; sys.exit(1)"]
        result = await engine.apply_update("v1.1.0", staging_dir, health_check_cmd=health_cmd)

        assert result.success is False
        assert result.rolled_back is True
        assert result.stage == "health_check"
        assert "exit code 1" in str(result.error)

        # Confirm rollback restored original state
        assert (install_dir / "canary.txt").read_text() == "ORIGINAL_CANARY_STATE"


class TestDaemonWorkerSupervisionAndCircuitBreaker:
    """Verifies worker lifecycle management, exponential backoff, and circuit breaker tripping."""

    @pytest.mark.asyncio
    async def test_worker_restart_on_transient_crash(self) -> None:
        """Assert crashed worker is restarted with backoff delay."""
        event_bus = EventBus()
        supervisor = DaemonWorkerSupervisor(event_bus=event_bus)

        runs = 0

        async def faulty_worker():
            nonlocal runs
            runs += 1
            if runs == 1:
                raise RuntimeError("transient network crash")
            while True:
                await asyncio.sleep(0.1)

        worker = supervisor.register_worker("faulty", faulty_worker)
        await supervisor.start_worker("faulty")
        assert worker.state == WorkerState.RUNNING

        await asyncio.sleep(0.05)
        # First monitor sweep detects the crash
        await supervisor.monitor_once()
        assert worker.state == WorkerState.CRASHED
        assert worker.restart_count == 1
        assert "transient network crash" in str(worker.last_error)

        # Advance time past the 1.0s backoff delay
        worker.next_retry_at = time.time() - 1.0
        await supervisor.monitor_once()
        await asyncio.sleep(0.05)
        assert worker.state == WorkerState.RUNNING
        assert runs == 2

        await supervisor.shutdown()

    def test_exponential_backoff_intervals(self) -> None:
        """Assert backoff intervals double exponentially: 1s, 2s, 4s, 8s, 16s, up to 30s."""
        supervisor = DaemonWorkerSupervisor()
        worker = supervisor.register_worker("test", lambda: None)

        now = time.time()
        for i in range(1, 7):
            worker.record_crash(f"crash_{i}", now=now)
            delay = worker.get_backoff_delay()
            if i == 1:
                assert delay == 1.0
            elif i == 2:
                assert delay == 2.0
            elif i == 3:
                assert delay == 4.0
            elif i == 4:
                assert delay == 8.0
            elif i == 5:
                assert delay == 16.0
            elif i >= 6:
                assert delay == 30.0

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips_after_five_rapid_crashes(self) -> None:
        """Assert worker crashing 5 times in 60s trips circuit breaker and emits EventBus alert."""
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe("supervisor.circuit_breaker.tripped", lambda ev: events_received.append(ev))

        supervisor = DaemonWorkerSupervisor(event_bus=event_bus)
        worker = supervisor.register_worker("broken_agent", lambda: None)

        now = time.time()
        # Record 5 rapid crashes within 10 seconds
        for i in range(5):
            worker.record_crash(f"fatal error {i}", now=now + i)

        assert worker.should_trip_circuit_breaker(now=now + 5) is True

        # Next start attempt triggers the circuit breaker
        started = await supervisor.start_worker("broken_agent")
        assert started is False
        assert worker.state == WorkerState.CIRCUIT_BROKEN
        assert len(events_received) == 1
        assert events_received[0].data["worker"] == "broken_agent"

    @pytest.mark.asyncio
    async def test_clean_process_tree_termination_on_shutdown(self) -> None:
        """Assert supervisor cleanly terminates running child processes on shutdown."""
        supervisor = DaemonWorkerSupervisor()
        # Spawn a sleeping subprocess
        cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        worker = supervisor.register_worker("sleeper_proc", cmd, is_process=True)

        await supervisor.start_worker("sleeper_proc")
        assert worker.state == WorkerState.RUNNING
        assert worker.process is not None
        assert worker.process.returncode is None

        # Clean shutdown with 1s timeout
        await supervisor.shutdown(timeout=1.0)
        assert worker.state == WorkerState.STOPPED
        assert worker.process.returncode is not None
