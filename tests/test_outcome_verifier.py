"""Comprehensive tests for OutcomeVerifier engine and verify_outcome tool."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Any, Dict
import pytest

from zelyr.engine.verifier import (
    OutcomeVerifier,
    VerificationResult,
    verify_command_output,
    verify_file_checksum,
    verify_file_contains,
    verify_file_exists,
    verify_file_json,
    verify_outcome,
    verify_port_listening,
    verify_process_running,
)
from zelyr.tools.verify_outcome import VerifyOutcomeTool, execute as verify_execute


class TestOutcomeVerifierEngine:
    """Unit tests for the independent OutcomeVerifier engine checks."""

    def test_verify_file_exists_positive(self, tmp_path: Path):
        f = tmp_path / "test_file.txt"
        f.write_text("sample content", encoding="utf-8")
        res = verify_file_exists(f)
        assert res.success is True
        assert res.check == "file_exists"
        assert res.target == str(f)
        assert res.observed["exists"] is True
        assert res.observed["size"] > 0
        assert res.error is None

    def test_verify_file_exists_negative(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist.txt"
        res = verify_file_exists(non_existent)
        assert res.success is False
        assert res.observed["exists"] is False
        assert "File not found" in (res.error or "")

    def test_verify_file_contains_positive_literal(self, tmp_path: Path):
        f = tmp_path / "log.txt"
        f.write_text("Server running on port 8080 successfully", encoding="utf-8")
        res = verify_file_contains(f, "port 8080")
        assert res.success is True
        assert res.observed["matched"] is True
        assert res.observed["snippet"] == "port 8080"

    def test_verify_file_contains_positive_regex(self, tmp_path: Path):
        f = tmp_path / "log.txt"
        f.write_text("Worker PID: 98765 started", encoding="utf-8")
        res = verify_file_contains(f, r"PID:\s+\d+", regex=True)
        assert res.success is True
        assert res.observed["matched"] is True
        assert "PID: 98765" in res.observed["snippet"]

    def test_verify_file_contains_negative(self, tmp_path: Path):
        f = tmp_path / "log.txt"
        f.write_text("Hello world", encoding="utf-8")
        res = verify_file_contains(f, "Goodbye")
        assert res.success is False
        assert res.observed["matched"] is False
        assert "Pattern 'Goodbye' not found" in (res.error or "")

    def test_verify_file_checksum_positive(self, tmp_path: Path):
        f = tmp_path / "artifact.bin"
        content = b"deterministic binary content payload"
        f.write_bytes(content)
        expected_sha = hashlib.sha256(content).hexdigest()
        res = verify_file_checksum(f, expected_sha)
        assert res.success is True
        assert res.observed["sha256"] == expected_sha

    def test_verify_file_checksum_negative(self, tmp_path: Path):
        f = tmp_path / "artifact.bin"
        f.write_bytes(b"actual content")
        wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"
        res = verify_file_checksum(f, wrong_sha)
        assert res.success is False
        assert "Checksum mismatch" in (res.error or "")

    def test_verify_file_json_positive(self, tmp_path: Path):
        f = tmp_path / "config.json"
        data = {"app": "zelyr", "version": "1.2.0", "status": "active"}
        f.write_text(json.dumps(data), encoding="utf-8")
        res = verify_file_json(f)
        assert res.success is True
        assert res.observed["valid"] is True
        assert "app" in res.observed["keys"]

    def test_verify_file_json_with_required_keys(self, tmp_path: Path):
        f = tmp_path / "config.json"
        data = {"app": "zelyr", "version": "1.2.0"}
        f.write_text(json.dumps(data), encoding="utf-8")
        # Match
        res = verify_file_json(f, schema={"required_keys": ["app", "version"]})
        assert res.success is True
        # Mismatch
        res_fail = verify_file_json(f, schema={"required_keys": ["app", "missing_key"]})
        assert res_fail.success is False
        assert "missing required keys" in (res_fail.error or "")

    def test_verify_file_json_malformed(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("{app: invalid-json", encoding="utf-8")
        res = verify_file_json(f)
        assert res.success is False
        assert "Invalid JSON" in (res.error or "")

    def test_verify_process_running_positive(self):
        # Current pytest/python process is running
        res = verify_process_running("pytest")
        if not res.success:
            res = verify_process_running("python")
        assert res.success is True
        assert res.observed["count"] > 0

    def test_verify_process_running_negative(self):
        res = verify_process_running("unlikely_process_name_zelyr_999999")
        assert res.success is False
        assert res.observed["count"] == 0
        assert "No process matching" in (res.error or "")

    def test_verify_port_listening(self):
        # Start a local TCP server on an ephemeral port
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        stop_thread = threading.Event()

        def _accept():
            srv.settimeout(1.0)
            while not stop_thread.is_set():
                try:
                    conn, _ = srv.accept()
                    conn.close()
                except (socket.timeout, OSError):
                    pass

        t = threading.Thread(target=_accept, daemon=True)
        t.start()

        try:
            # Positive test
            res_pass = verify_port_listening(port, host="127.0.0.1", timeout=0.5)
            assert res_pass.success is True
            assert res_pass.observed["listening"] is True

            # Negative test on unlikely port
            res_fail = verify_port_listening(65530, host="127.0.0.1", timeout=0.2)
            assert res_fail.success is False
            assert res_fail.observed["listening"] is False
        finally:
            stop_thread.set()
            srv.close()
            t.join(timeout=1.0)

    def test_verify_command_output_positive(self):
        res = verify_command_output("echo 'outcome_verification_smoke_test'", "smoke_test")
        assert res.success is True
        assert res.observed["exit_code"] == 0
        assert "smoke_test" in res.observed["stdout"]

    def test_verify_command_output_negative(self):
        res = verify_command_output("echo 'hello'", "absent_pattern_xyz")
        assert res.success is False
        assert "not found in command output" in (res.error or "")

    def test_verify_command_output_timeout(self):
        res = verify_command_output("sleep 1.0", "done", timeout=0.1)
        assert res.success is False
        assert res.observed.get("timeout") is True
        assert "timed out" in (res.error or "")

    def test_verify_outcome_dispatcher(self, tmp_path: Path):
        f = tmp_path / "disp.txt"
        f.write_text("dispatcher_target_value", encoding="utf-8")

        # file_exists
        assert verify_outcome("file_exists", str(f)).success is True
        assert verify_outcome("exists", str(f)).success is True

        # file_contains
        assert verify_outcome("file_contains", str(f), "target_value").success is True
        assert verify_outcome("contains", str(f), "target_value").success is True

        # unknown check
        res_unk = verify_outcome("unknown_check", str(f))
        assert res_unk.success is False
        assert "Unknown verification check" in (res_unk.error or "")

    def test_verification_result_to_dict(self):
        res = VerificationResult(
            success=True,
            check="file_exists",
            target="/tmp/test",
            observed={"exists": True},
            error=None,
        )
        d = res.to_dict()
        assert d == {
            "success": True,
            "check": "file_exists",
            "target": "/tmp/test",
            "observed": {"exists": True},
            "error": None,
        }




class TestVerifyOutcomeToolExecution:
    """Verify canonical verify_outcome tool execution independently of registry."""

    @pytest.mark.asyncio
    async def test_verify_outcome_tool_execute_positive(self, tmp_path: Path):
        f = tmp_path / "valid.txt"
        f.write_text("sample content", encoding="utf-8")
        tool = VerifyOutcomeTool()
        result = await tool.execute(check="file_exists", target=str(f))
        assert result.success is True
        assert "[VERIFIED]" in result.output
        assert result.metadata["verification"]["success"] is True

    @pytest.mark.asyncio
    async def test_verify_outcome_tool_execute_negative(self, tmp_path: Path):
        f = tmp_path / "not_there.txt"
        tool = VerifyOutcomeTool()
        result = await tool.execute(check="file_exists", target=str(f))
        assert result.success is False
        assert "File not found" in (result.error or "")
        assert result.metadata["verification"]["success"] is False

    @pytest.mark.asyncio
    async def test_verify_outcome_module_execute(self, tmp_path: Path):
        f = tmp_path / "valid.txt"
        f.write_text("hello", encoding="utf-8")
        raw_json = await verify_execute(check="file_contains", target=str(f), expected="hello")
        data = json.loads(raw_json)
        assert data["status"] == "success"
        assert data["verified"] is True
        assert data["check"] == "file_contains"
