"""Comprehensive deterministic test battery for the Acceptance Criteria and Verification Pipeline.

Audits:
1. All supported criterion types and aliases.
2. Multiple criteria separated by commas, semicolons, and newlines.
3. Values containing punctuation, colons, Windows paths, and delimiter-like text.
4. Missing, empty, malformed, and unsupported criteria.
5. Multiple criteria with mixed pass/fail outcomes.
6. Headless CLI exit status for all-pass, partial-failure, and invalid specifications.
7. Consistency between parser, verifier, ledger state, and goal-aware replanning.
8. Regression coverage for the original two-file comma-separated specification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import tempfile
import threading
from typing import Any

import pytest

from zelyr.cli.main import _execute_headless_run
from zelyr.engine.execution_ledger import (
    TurnExecutionLedger,
    canonicalize_target,
    find_unfulfilled_task_requirements,
    parse_acceptance_criteria,
)
from zelyr.engine.verifier import OutcomeVerifier, VerificationResult


class TestSupportedCriterionTypes:
    """Category 1: Verify all supported criterion types and aliases."""

    def test_file_exists_and_alias(self, tmp_path: Path):
        f = tmp_path / "exist.txt"
        f.write_text("ok", encoding="utf-8")

        res1 = OutcomeVerifier.verify("file_exists", str(f))
        assert res1.success is True
        assert res1.observed["exists"] is True

        res2 = OutcomeVerifier.verify("exists", str(f))
        assert res2.success is True

    def test_file_contains_and_alias(self, tmp_path: Path):
        f = tmp_path / "content.txt"
        f.write_text("Hello Zelyr 1.4 World", encoding="utf-8")

        res1 = OutcomeVerifier.verify("file_contains", str(f), expected="Zelyr 1.4")
        assert res1.success is True

        res2 = OutcomeVerifier.verify("contains", str(f), expected="Zelyr 1.4")
        assert res2.success is True

    def test_file_checksum_and_aliases(self, tmp_path: Path):
        f = tmp_path / "checksum.txt"
        payload = "digest test data"
        f.write_text(payload, encoding="utf-8")
        expected_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        for alias in ("file_checksum", "checksum", "sha256"):
            res = OutcomeVerifier.verify(alias, str(f), expected=expected_sha)
            assert res.success is True
            assert res.observed["sha256"] == expected_sha

    def test_file_json_and_alias(self, tmp_path: Path):
        f = tmp_path / "data.json"
        data = {"service": "zelyr", "version": "1.4", "status": "active"}
        f.write_text(json.dumps(data), encoding="utf-8")

        res1 = OutcomeVerifier.verify("file_json", str(f))
        assert res1.success is True
        assert res1.observed["valid"] is True

        # Using comma-separated expected required keys
        res2 = OutcomeVerifier.verify("file_json", str(f), expected="service,version")
        assert res2.success is True

        # Using json schema in expected
        res3 = OutcomeVerifier.verify("json", str(f), expected='{"required_keys": ["status"]}')
        assert res3.success is True

    def test_process_running_and_aliases(self):
        # Current pytest process matches python
        for alias in ("process_running", "process", "proc"):
            res = OutcomeVerifier.verify(alias, "python")
            assert res.success is True
            assert res.observed["count"] >= 1

    def test_port_listening_and_aliases(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(128)
        port = srv.getsockname()[1]
        stop = threading.Event()

        def _accept():
            srv.settimeout(0.2)
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                    conn.close()
                except Exception:
                    pass

        t = threading.Thread(target=_accept, daemon=True)
        t.start()
        try:
            for alias in ("port_listening", "port", "listen"):
                # Port only
                res1 = OutcomeVerifier.verify(alias, str(port))
                assert res1.success is True

                # Host:port format
                res2 = OutcomeVerifier.verify(alias, f"127.0.0.1:{port}")
                assert res2.success is True

                # Split host target and port expected
                res3 = OutcomeVerifier.verify(alias, "127.0.0.1", expected=str(port))
                assert res3.success is True
        finally:
            stop.set()
            srv.close()

    def test_command_output_and_aliases(self):
        for alias in ("command_output", "command", "cmd"):
            res = OutcomeVerifier.verify(alias, "echo 'Zelyr 1.4 Verified'", expected="Zelyr 1.4")
            assert res.success is True


class TestMultipleCriteriaDelimiters:
    """Category 2: Multiple criteria separated by each supported delimiter."""

    def test_comma_delimited_criteria(self):
        spec = "file_exists:/tmp/a.txt,file_contains:/tmp/b.txt:BETA"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 2
        assert parsed[0] == {"check": "file_exists", "target": "/tmp/a.txt", "expected": None}
        assert parsed[1] == {"check": "file_contains", "target": "/tmp/b.txt", "expected": "BETA"}

    def test_semicolon_delimited_criteria(self):
        spec = "file_exists:/tmp/a.txt;file_exists:/tmp/b.txt;file_contains:/tmp/c.txt:GAMMA"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 3
        assert parsed[0]["target"] == "/tmp/a.txt"
        assert parsed[1]["target"] == "/tmp/b.txt"
        assert parsed[2]["expected"] == "GAMMA"

    def test_newline_delimited_criteria(self):
        spec = "file_exists:/tmp/a.txt\nfile_exists:/tmp/b.txt\nfile_contains:/tmp/c.txt:DELTA"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 3
        assert parsed[0]["check"] == "file_exists"
        assert parsed[1]["target"] == "/tmp/b.txt"
        assert parsed[2]["expected"] == "DELTA"

    def test_mixed_delimiters(self):
        spec = "file_exists:/tmp/a.txt; file_contains:/tmp/b.txt:BETA\nport_listening:8080,process_running:nginx"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 4
        assert parsed[0]["check"] == "file_exists"
        assert parsed[1]["expected"] == "BETA"
        assert parsed[2]["check"] == "port_listening"
        assert parsed[3]["check"] == "process_running"

    def test_list_of_strings_or_dicts(self):
        spec = [
            "file_exists:/tmp/a.txt",
            {"check": "file_contains", "target": "/tmp/b.txt", "expected": "BETA"},
        ]
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 2
        assert parsed[0]["target"] == "/tmp/a.txt"
        assert parsed[1]["expected"] == "BETA"

    def test_json_array_input(self):
        spec_json = json.dumps([
            {"check": "file_exists", "target": "/tmp/a.txt"},
            {"check": "file_contains", "target": "/tmp/b.txt", "expected": "BETA"},
        ])
        parsed = parse_acceptance_criteria(spec_json)
        assert len(parsed) == 2
        assert parsed[0]["check"] == "file_exists"
        assert parsed[1]["expected"] == "BETA"


class TestPunctuationAndDelimiterHandling:
    """Category 3: Values containing punctuation, quotes, Windows paths, and delimiter-like text."""

    def test_semicolon_inside_expected_value(self):
        spec = "file_contains:/tmp/code.py:int a = 1; int b = 2;"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 1
        assert parsed[0]["target"] == "/tmp/code.py"
        assert parsed[0]["expected"] == "int a = 1; int b = 2;"

    def test_comma_inside_expected_value(self):
        spec = "file_contains:/tmp/data.csv:id,name,role,status"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 1
        assert parsed[0]["expected"] == "id,name,role,status"

    def test_colon_inside_expected_value(self):
        spec = "file_contains:/tmp/log.txt:timestamp: 2026-11-11 12:00:00"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 1
        assert parsed[0]["expected"] == "timestamp: 2026-11-11 12:00:00"

    def test_text_resembling_another_criterion_in_expected(self):
        spec = "file_contains:/tmp/test.log:Found file_exists: here"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 1
        assert parsed[0]["expected"] == "Found file_exists: here"

    def test_quoted_values_with_delimiters(self):
        spec = 'file_contains:/tmp/test.txt:"first, second; third"\nfile_exists:/tmp/other.txt'
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 2
        assert parsed[0]["expected"] == "first, second; third"
        assert parsed[1]["target"] == "/tmp/other.txt"

    def test_quoted_target_with_spaces(self):
        spec = 'file_contains:"/tmp/path with space/file.txt":ALPHA'
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 1
        assert parsed[0]["target"] == "/tmp/path with space/file.txt"
        assert parsed[0]["expected"] == "ALPHA"

    def test_windows_paths_with_drive_letters(self):
        # Windows path for file_exists
        spec_exists = r"file_exists:C:\Users\tester\file.txt"
        parsed_exists = parse_acceptance_criteria(spec_exists)
        assert len(parsed_exists) == 1
        assert parsed_exists[0]["target"] == r"C:\Users\tester\file.txt"
        assert parsed_exists[0]["expected"] is None

        # Windows path for file_contains
        spec_contains = r"file_contains:C:\Users\tester\file.txt:WIN_CONTENT"
        parsed_contains = parse_acceptance_criteria(spec_contains)
        assert len(parsed_contains) == 1
        assert parsed_contains[0]["target"] == r"C:\Users\tester\file.txt"
        assert parsed_contains[0]["expected"] == "WIN_CONTENT"

        # Canonicalization handles Windows drive letter paths consistently
        norm = canonicalize_target(r"C:\Users\tester\file.txt")
        assert norm == r"c:\users\tester\file.txt"

    def test_unicode_and_special_characters(self, tmp_path: Path):
        f = tmp_path / "данные_файл.txt"
        content = "Тест Unicode: äöü, 你好, 🚀"
        f.write_text(content, encoding="utf-8")

        spec = f"file_contains:{f}:{content}"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 1
        assert parsed[0]["expected"] == content

        res = OutcomeVerifier.verify(parsed[0]["check"], parsed[0]["target"], expected=parsed[0]["expected"])
        assert res.success is True


class TestMissingEmptyMalformedAndUnsupportedCriteria:
    """Category 4: Missing, empty, malformed, and unsupported criteria fail-closed."""

    def test_empty_criteria_input(self):
        assert parse_acceptance_criteria("") == []
        assert parse_acceptance_criteria("   ") == []
        assert parse_acceptance_criteria(None) == []

    def test_file_exists_empty_target_fails(self):
        res = OutcomeVerifier.verify("file_exists", "")
        assert res.success is False
        assert "requires a non-empty target path" in (res.error or "")

    def test_file_exists_with_expected_content_fails(self, tmp_path: Path):
        f = tmp_path / "exist.txt"
        f.write_text("content", encoding="utf-8")
        res = OutcomeVerifier.verify("file_exists", str(f), expected="content")
        assert res.success is False
        assert "Incompatible verification specification" in (res.error or "")

    def test_file_contains_missing_expected_fails(self, tmp_path: Path):
        f = tmp_path / "exist.txt"
        f.write_text("content", encoding="utf-8")
        res1 = OutcomeVerifier.verify("file_contains", str(f), expected=None)
        assert res1.success is False
        assert "requires expected pattern or content" in (res1.error or "")

        res2 = OutcomeVerifier.verify("file_contains", str(f), expected="")
        assert res2.success is False
        assert "requires expected pattern or content" in (res2.error or "")

    def test_file_contains_empty_target_fails(self):
        res = OutcomeVerifier.verify("file_contains", "", expected="some_content")
        assert res.success is False
        assert "requires a non-empty target path" in (res.error or "")

    def test_file_checksum_empty_target_or_expected_fails(self, tmp_path: Path):
        f = tmp_path / "exist.txt"
        f.write_text("content", encoding="utf-8")
        res1 = OutcomeVerifier.verify("file_checksum", "")
        assert res1.success is False

        res2 = OutcomeVerifier.verify("file_checksum", str(f), expected="")
        assert res2.success is False
        assert "requires an expected SHA-256 digest" in (res2.error or "")

    def test_file_json_empty_target_fails(self):
        res = OutcomeVerifier.verify("file_json", "")
        assert res.success is False
        assert "requires a non-empty target path" in (res.error or "")

    def test_process_running_empty_pattern_fails(self):
        res = OutcomeVerifier.verify("process_running", "")
        assert res.success is False
        assert "requires a non-empty process pattern" in (res.error or "")

    def test_process_running_with_expected_fails(self):
        res = OutcomeVerifier.verify("process_running", "python", expected="unexpected")
        assert res.success is False
        assert "Incompatible verification specification" in (res.error or "")

    def test_port_listening_empty_port_fails(self):
        res = OutcomeVerifier.verify("port_listening", "")
        assert res.success is False
        assert "requires a non-empty port number" in (res.error or "")

    def test_command_output_empty_command_or_expected_fails(self):
        res1 = OutcomeVerifier.verify("command_output", "")
        assert res1.success is False

        res2 = OutcomeVerifier.verify("command_output", "echo hi", expected="")
        assert res2.success is False

    def test_unsupported_check_fails(self):
        res = OutcomeVerifier.verify("nonexistent_check", "/tmp/a")
        assert res.success is False
        assert "Unknown verification check 'nonexistent_check'" in (res.error or "")


class TestMultipleCriteriaMixedOutcomes:
    """Category 5: Multiple criteria with mixed pass/fail outcomes."""

    def test_one_pass_one_fail_overall_failure(self, tmp_path: Path):
        f1 = tmp_path / "pass.txt"
        f1.write_text("ALPHA", encoding="utf-8")
        f2 = tmp_path / "fail.txt"
        # f2 does not exist

        spec = f"file_contains:{f1}:ALPHA,file_exists:{f2}"
        parsed = parse_acceptance_criteria(spec)
        assert len(parsed) == 2

        results = [
            OutcomeVerifier.verify(c["check"], c["target"], expected=c.get("expected"))
            for c in parsed
        ]
        assert results[0].success is True
        assert results[1].success is False
        assert all(r.success for r in results) is False

    def test_three_criteria_middle_fails(self, tmp_path: Path):
        f1 = tmp_path / "1.txt"
        f1.write_text("ONE", encoding="utf-8")
        f2 = tmp_path / "2.txt"
        f2.write_text("WRONG", encoding="utf-8")
        f3 = tmp_path / "3.txt"
        f3.write_text("THREE", encoding="utf-8")

        spec = f"file_contains:{f1}:ONE;file_contains:{f2}:TWO;file_contains:{f3}:THREE"
        parsed = parse_acceptance_criteria(spec)
        results = [
            OutcomeVerifier.verify(c["check"], c["target"], expected=c.get("expected"))
            for c in parsed
        ]
        assert [r.success for r in results] == [True, False, True]
        assert all(r.success for r in results) is False


class TestCLIExitStatusSemantics:
    """Category 6: CLI exit status for all-pass, partial-failure, and invalid specifications."""

    @pytest.mark.asyncio
    async def test_cli_all_pass_exit_0(self, tmp_path: Path, monkeypatch, capsys):
        f1 = tmp_path / "a.txt"
        f1.write_text("A_OK", encoding="utf-8")
        f2 = tmp_path / "b.txt"
        f2.write_text("B_OK", encoding="utf-8")

        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed."}}]}

        from zelyr.agents.native_orchestrator import NativeOrchestrator
        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            verify_spec=f"file_contains:{f1}:A_OK,file_contains:{f2}:B_OK",
        )
        assert code == 0
        captured = capsys.readouterr()
        assert f"[verified] file_contains on {f1}: PASS" in captured.out
        assert f"[verified] file_contains on {f2}: PASS" in captured.out

    @pytest.mark.asyncio
    async def test_cli_partial_failure_exit_2(self, tmp_path: Path, monkeypatch, capsys):
        f1 = tmp_path / "a.txt"
        f1.write_text("A_OK", encoding="utf-8")
        f2 = tmp_path / "missing.txt"

        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed."}}]}

        from zelyr.agents.native_orchestrator import NativeOrchestrator
        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            verify_spec=f"file_contains:{f1}:A_OK,file_exists:{f2}",
        )
        assert code == 2
        captured = capsys.readouterr()
        assert f"[verified] file_contains on {f1}: PASS" in captured.out
        assert f"[verification failed] file_exists on {f2}: FAIL" in captured.out

    @pytest.mark.asyncio
    async def test_cli_empty_verify_spec_exit_2(self, monkeypatch, capsys):
        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed."}}]}

        from zelyr.agents.native_orchestrator import NativeOrchestrator
        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            verify_spec="   ",
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "[verification failed] Empty verification specification provided." in captured.err

    @pytest.mark.asyncio
    async def test_cli_incomplete_criterion_exit_2(self, monkeypatch, capsys):
        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed."}}]}

        from zelyr.agents.native_orchestrator import NativeOrchestrator
        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            verify_spec="file_exists:",
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "[verification failed] file_exists on : FAIL" in captured.out

    @pytest.mark.asyncio
    async def test_cli_json_mode_reports_failure_exit_2(self, tmp_path: Path, monkeypatch, capsys):
        f1 = tmp_path / "ok.txt"
        f1.write_text("OK", encoding="utf-8")
        f2 = tmp_path / "absent.txt"

        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task done."}}]}

        from zelyr.agents.native_orchestrator import NativeOrchestrator
        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            output_json=True,
            verify_spec=f"file_exists:{f1},file_exists:{f2}",
        )
        assert code == 2
        captured = capsys.readouterr()
        data = json.loads(captured.out.strip())
        assert "verification" in data
        assert data["verification"]["success"] is False
        assert len(data["verification"]["checks"]) == 2


class TestPipelineConsistency:
    """Category 7: Consistency between parser output, OutcomeVerifier, ledger, and replanning."""

    def test_parser_verifier_ledger_replanning_alignment(self, tmp_path: Path):
        f_verified = tmp_path / "done.txt"
        f_verified.write_text("DONE_CONTENT", encoding="utf-8")
        f_pending = tmp_path / "pending.txt"

        spec = f"file_contains:{f_verified}:DONE_CONTENT,file_contains:{f_pending}:PENDING_CONTENT"
        ledger = TurnExecutionLedger()

        # Call find_unfulfilled_task_requirements
        verified_outcomes, unfulfilled_outcomes = find_unfulfilled_task_requirements(
            user_prompt=f"Create {f_verified} and {f_pending}",
            ledger=ledger,
            acceptance_spec=spec,
        )

        # 1. Verification results accurately separated
        assert any(str(f_verified) in v and "verified" in v for v in verified_outcomes)
        assert any(str(f_pending) in u and "PENDING_CONTENT" in u for u in unfulfilled_outcomes)

        # 2. Ledger recorded f_verified as verified target
        assert ledger.is_target_verified(str(f_verified)) is True
        assert ledger.is_target_verified(str(f_pending)) is False


class TestOriginalTwoFileRegression:
    """Category 8: Regression coverage for original two-file comma-separated specification."""

    def test_original_two_file_comma_separated_workflow(self, tmp_path: Path):
        f_a = tmp_path / "file_alpha.txt"
        f_b = tmp_path / "file_beta.txt"

        spec = f"file_contains:{f_a}:ALPHA_101,file_contains:{f_b}:BETA_202"
        parsed = parse_acceptance_criteria(spec)

        assert len(parsed) == 2
        assert parsed[0] == {"check": "file_contains", "target": str(f_a), "expected": "ALPHA_101"}
        assert parsed[1] == {"check": "file_contains", "target": str(f_b), "expected": "BETA_202"}

        # Case A: Neither exists -> Both fail
        res_a = [OutcomeVerifier.verify(c["check"], c["target"], expected=c["expected"]) for c in parsed]
        assert [r.success for r in res_a] == [False, False]

        # Case B: Only alpha written -> Alpha passes, beta fails
        f_a.write_text("ALPHA_101", encoding="utf-8")
        res_b = [OutcomeVerifier.verify(c["check"], c["target"], expected=c["expected"]) for c in parsed]
        assert [r.success for r in res_b] == [True, False]

        # Case C: Both written -> Both pass
        f_b.write_text("BETA_202", encoding="utf-8")
        res_c = [OutcomeVerifier.verify(c["check"], c["target"], expected=c["expected"]) for c in parsed]
        assert [r.success for r in res_c] == [True, True]
        assert all(r.success for r in res_c) is True
