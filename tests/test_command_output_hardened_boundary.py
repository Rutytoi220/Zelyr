"""Comprehensive regression test suite for the hardened command_output verification boundary.

Boundary Scope Notice:
    This boundary enforces deterministic application-level restrictions (command allowlist,
    shell-syntax and dangerous-flag rejection, trusted executable resolution, minimal
    environment allowlisting, process-group timeout termination, and bounded stream reads).
    It is NOT a complete OS-level sandbox (no kernel namespaces, chroot, seccomp, or cgroup
    isolation are claimed or provided).

Verifies:
1. Permitted harmless commands.
2. Commands outside the allowlist.
3. Shell chaining and pipelines.
4. Redirection and command substitution.
5. Interpreter or script invocation.
6. Environment variable handling and strict minimal allowlist.
7. Timeout behavior.
8. Streaming output limits and exact boundary behavior (at limit and above limit).
9. Parent and child process-group cleanup after timeout.
10. Git external helper/pager disarming.
11. Executable resolution independent of untrusted PATH.
12. Python test-runner workspace trust boundary and target path enforcement.
13. Invalid and malformed criteria.
14. CLI behavior when verification is rejected.
15. Consistency between verifier results and final task status.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zelyr.engine.verifier import (
    MAX_COMMAND_OUTPUT_BYTES,
    OutcomeVerifier,
    _get_sanitized_environment,
    _read_stream_bounded,
    _resolve_executable,
    _validate_and_parse_command,
    verify_command_output,
)


class TestHardenedCommandOutputBoundary:
    """Rigorous tests asserting that command_output cannot be abused as an arbitrary shell vector."""

    def test_1_permitted_harmless_command(self):
        """Allowlisted commands (echo, sleep, git status, python -m pytest) execute safely."""
        # Echo
        res_echo = OutcomeVerifier.verify("command_output", "echo 'PARITY_SMOKE_TEST'", expected="SMOKE_TEST")
        assert res_echo.success is True
        assert res_echo.observed.get("exit_code") == 0
        assert res_echo.observed.get("matched") is True

        # Git read-only
        repo_root = Path(__file__).resolve().parent.parent
        res_git = OutcomeVerifier.verify("command_output", "git status", expected="On branch", cwd=repo_root)
        assert res_git.observed.get("exit_code") == 0
        assert res_git.observed.get("safety_blocked") is not True

        # Python test runner module
        res_py = OutcomeVerifier.verify("command_output", f"{sys.executable} -m pytest --version", expected="pytest")
        assert res_py.success is True
        assert res_py.observed.get("exit_code") == 0

    def test_2_command_outside_allowlist(self):
        """Commands not in the explicit allowlist are immediately rejected without execution."""
        disallowed = [
            ("whoami", "root"),
            ("curl https://example.com", "example"),
            ("wget https://example.com", "example"),
            ("cat /etc/passwd", "root"),
            ("rm -f test.txt", "test"),
            ("sudo id", "root"),
            ("chmod 777 test.txt", "test"),
            ("nc -l 8080", "listen"),
        ]
        for cmd, exp in disallowed:
            res = OutcomeVerifier.verify("command_output", cmd, expected=exp)
            assert res.success is False
            assert res.observed.get("safety_blocked") is True
            assert "not permit-listed" in (res.error or "")

    def test_3_shell_chaining_and_pipelines(self):
        """Shell operators (;, &&, ||, |, &) are rejected before execution."""
        chaining_cmds = [
            "echo harmless; whoami",
            "echo harmless && whoami",
            "echo harmless || whoami",
            "echo harmless | grep harmless",
            "echo harmless &",
        ]
        for cmd in chaining_cmds:
            res = OutcomeVerifier.verify("command_output", cmd, expected="harmless")
            assert res.success is False
            assert res.observed.get("safety_blocked") is True
            assert "shell operators" in (res.error or "")

    def test_4_redirection_and_command_substitution(self):
        """Redirection (>, >>, <) and command substitution ($(..), `..`) are blocked."""
        injection_cmds = [
            "echo safe > output.txt",
            "echo safe >> output.txt",
            "cat < input.txt",
            "echo $(whoami)",
            "echo `whoami`",
        ]
        for cmd in injection_cmds:
            res = OutcomeVerifier.verify("command_output", cmd, expected="safe")
            assert res.success is False
            assert res.observed.get("safety_blocked") is True
            assert ("shell operators" in (res.error or "") or "not permit-listed" in (res.error or ""))

    def test_5_interpreter_or_script_invocation(self):
        """Shell interpreters and arbitrary Python script/-c invocations are forbidden."""
        # Shell interpreters
        interpreters = [
            "sh -c 'echo test'",
            "bash -c 'echo test'",
            "zsh -c 'echo test'",
            "dash -c 'echo test'",
        ]
        for sh_cmd in interpreters:
            res = OutcomeVerifier.verify("command_output", sh_cmd, expected="test")
            assert res.success is False
            assert res.observed.get("safety_blocked") is True
            assert "shell interpreter" in (res.error or "")

        # Python arbitrary inline code
        res_c = OutcomeVerifier.verify("command_output", f"{sys.executable} -c 'print(123)'", expected="123")
        assert res_c.success is False
        assert res_c.observed.get("safety_blocked") is True
        assert "'python -c' is not permitted" in (res_c.error or "")

        # Python arbitrary script path
        res_script = OutcomeVerifier.verify("command_output", f"{sys.executable} /tmp/evil_test_script.py", expected="123")
        assert res_script.success is False
        assert res_script.observed.get("safety_blocked") is True
        assert "must use '-m <module>'" in (res_script.error or "")

        # Python non-allowlisted module
        res_mod = OutcomeVerifier.verify("command_output", f"{sys.executable} -m os", expected="os")
        assert res_mod.success is False
        assert res_mod.observed.get("safety_blocked") is True
        assert "not permitted for verification" in (res_mod.error or "")

    def test_6_environment_variable_handling(self):
        """Environment variables are rejected in command strings and sanitized in subprocesses."""
        # Variable expansion in command string is rejected
        res_env = OutcomeVerifier.verify("command_output", "echo $SECRET_KEY", expected="val")
        assert res_env.success is False
        assert res_env.observed.get("safety_blocked") is True
        assert "environment variable expansion is prohibited" in (res_env.error or "")

        # Strict minimal environment allowlist drops arbitrary secrets
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-secret-12345",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-xyz",
            "GITHUB_TOKEN": "ghp_secrettoken",
            "DATABASE_URL": "postgres://user:pass@host/db",
            "MY_CUSTOM_SECRET": "critical_value",
        }):
            sanitized = _get_sanitized_environment(Path.cwd())
            assert "OPENAI_API_KEY" not in sanitized
            assert "AWS_SECRET_ACCESS_KEY" not in sanitized
            assert "GITHUB_TOKEN" not in sanitized
            assert "DATABASE_URL" not in sanitized
            assert "MY_CUSTOM_SECRET" not in sanitized
            assert "PATH" in sanitized
            assert sanitized.get("GIT_PAGER") == "cat"
            assert sanitized.get("PYTHONNOUSERSITE") == "1"

    def test_7_timeout_behavior(self):
        """Commands exceeding timeout are killed and reported with timeout=True."""
        res = verify_command_output("sleep 5.0", expected_pattern="done", timeout=0.1)
        assert res.success is False
        assert res.observed.get("timeout") is True
        assert res.observed.get("exit_code") is None
        assert "timed out after" in (res.error or "")

    def test_8_streaming_output_limits_exact_boundary(self):
        """Streaming reader bounds memory usage and tests exact limit boundary."""
        import threading

        # 1. Exact boundary: stream with exactly MAX_COMMAND_OUTPUT_BYTES
        exact_data = b"A" * MAX_COMMAND_OUTPUT_BYTES
        stream_exact = io.BytesIO(exact_data)
        buf_exact = []
        counter_exact = [0]
        lock = threading.Lock()
        stop_event = threading.Event()

        _read_stream_bounded(
            stream_exact,
            buf_exact,
            counter_exact,
            lock,
            stop_event,
            max_bytes=MAX_COMMAND_OUTPUT_BYTES,
        )
        assert not stop_event.is_set()
        assert counter_exact[0] == MAX_COMMAND_OUTPUT_BYTES
        assert len(b"".join(buf_exact)) == MAX_COMMAND_OUTPUT_BYTES

        # 2. Above boundary: stream with MAX_COMMAND_OUTPUT_BYTES + 1
        over_data = b"B" * (MAX_COMMAND_OUTPUT_BYTES + 1)
        stream_over = io.BytesIO(over_data)
        buf_over = []
        counter_over = [0]
        stop_event_over = threading.Event()

        _read_stream_bounded(
            stream_over,
            buf_over,
            counter_over,
            lock,
            stop_event_over,
            max_bytes=MAX_COMMAND_OUTPUT_BYTES,
        )
        assert stop_event_over.is_set()
        assert counter_over[0] > MAX_COMMAND_OUTPUT_BYTES
        # Buffer is capped at MAX_COMMAND_OUTPUT_BYTES
        assert len(b"".join(buf_over)) <= MAX_COMMAND_OUTPUT_BYTES

    def test_9_parent_and_child_process_cleanup_after_timeout(self):
        """On POSIX, session killpg ensures child and grandchild processes are terminated."""
        if sys.platform == "win32":
            pytest.skip("Process group testing with SIGKILL is POSIX-specific")

        # Create a harmless script that spawns a background child process that sleeps
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            script_file = tmp_path / "test_runner_proc.py"
            child_pid_file = tmp_path / "child.pid"
            script_file.write_text(f"""
import subprocess, sys, time
proc = subprocess.Popen(['sleep', '30'])
with open('{child_pid_file}', 'w') as f:
    f.write(str(proc.pid))
time.sleep(30)
""", encoding="utf-8")

            # Run via allowlisted python module unittest/pytest running this script as a test
            test_file = tmp_path / "test_timeout_tree.py"
            test_file.write_text(f"""
import subprocess, time
def test_long():
    proc = subprocess.Popen(['sleep', '30'])
    with open('{child_pid_file}', 'w') as f:
        f.write(str(proc.pid))
    time.sleep(30)
""", encoding="utf-8")

            # Execute with short timeout (0.3s)
            res = verify_command_output(
                f"{sys.executable} -m pytest {test_file.name}",
                expected_pattern="passed",
                timeout=0.3,
                cwd=tmp_path,
            )
            assert res.success is False
            assert res.observed.get("timeout") is True

            # Check if child process was terminated
            if child_pid_file.exists():
                c_pid = int(child_pid_file.read_text().strip())
                time.sleep(0.1)
                try:
                    os.kill(c_pid, 0)
                    is_alive = True
                except (ProcessLookupError, PermissionError):
                    is_alive = False
                assert not is_alive, f"Child process {c_pid} outlived the timeout!"

    def test_10_git_external_helper_and_pager_disarmed(self):
        """Git commands cannot trigger external diff drivers, pagers, or hooks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo = Path(tmp_dir)
            subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, check=True)

            f = repo / "sample.txt"
            f.write_text("initial content", encoding="utf-8")
            subprocess.run(["git", "add", "sample.txt"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True)

            # Modify file
            f.write_text("modified content", encoding="utf-8")

            # Create a canary file that would be touched if a pager or external diff ran
            pager_marker = repo / "pager_ran.canary"
            fake_pager = repo / "fake_pager.py"
            fake_pager.write_text(f"""
from pathlib import Path
Path('{pager_marker}').touch()
""", encoding="utf-8")

            # Configure repo-level git config to use this pager and external diff
            subprocess.run(
                ["git", "config", "core.pager", f"{sys.executable} {fake_pager}"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                ["git", "config", "diff.external", f"{sys.executable} {fake_pager}"],
                cwd=repo,
                check=True,
            )

            # Verify diff through hardened verify_command_output
            res = verify_command_output("git diff", expected_pattern="modified content", cwd=repo)
            assert res.success is True
            # Assert that the fake pager/diff was NEVER executed
            assert not pager_marker.exists(), "Git invoked an external pager or helper!"

    def test_11_executable_resolution_independent_of_untrusted_path(self):
        """Executable resolution ignores fake binaries in an untrusted PATH."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_bin_dir = Path(tmp_dir) / "fake_bin"
            fake_bin_dir.mkdir()
            fake_git = fake_bin_dir / "git"
            fake_git.write_text("#!/bin/sh\necho 'MALICIOUS_GIT'\n", encoding="utf-8")
            fake_git.chmod(0o755)

            # Prepend fake_bin to PATH
            with patch.dict(os.environ, {"PATH": f"{fake_bin_dir}:{os.environ.get('PATH', '')}"}):
                ok, resolved, err = _resolve_executable("git", "git")
                assert ok is True
                assert str(fake_bin_dir) not in resolved
                assert resolved in ("/usr/bin/git", "/bin/git", "/usr/local/bin/git")

    def test_12_python_test_runner_workspace_path_constraint(self):
        """Python test runner rejects positional target paths outside the workspace."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            outside_file = "/etc/passwd"

            res = verify_command_output(
                f"{sys.executable} -m pytest {outside_file}",
                expected_pattern="test",
                cwd=workspace,
            )
            assert res.success is False
            assert res.observed.get("safety_blocked") is True
            assert "must reside within the verified workspace" in (res.error or "")

    def test_13_invalid_and_malformed_criteria(self):
        """Malformed, empty, or unparseable commands fail cleanly."""
        assert OutcomeVerifier.verify("command_output", "").success is False
        assert OutcomeVerifier.verify("command_output", "   ").success is False
        assert OutcomeVerifier.verify("command_output", "echo test", expected="").success is False

        # Unbalanced quotes
        res_quote = OutcomeVerifier.verify("command_output", "echo 'unclosed", expected="test")
        assert res_quote.success is False
        assert res_quote.observed.get("safety_blocked") is True

        # Git without subcommand
        res_git = OutcomeVerifier.verify("command_output", "git", expected="test")
        assert res_git.success is False

        # Git mutating subcommands
        mutating = ["git commit -m 'test'", "git push", "git reset --hard", "git clean -fdx"]
        for m in mutating:
            r = OutcomeVerifier.verify("command_output", m, expected="test")
            assert r.success is False
            assert r.observed.get("safety_blocked") is True
            assert "Only read-only inspection subcommands are allowed" in (r.error or "")

        # Git config injection
        res_cfg = OutcomeVerifier.verify("command_output", "git -c core.pager=evil status", expected="test")
        assert res_cfg.success is False
        assert "git flag '-c' is prohibited" in (res_cfg.error or "")

    @pytest.mark.asyncio
    async def test_14_cli_behavior_when_verification_is_rejected(self, monkeypatch, capsys):
        """CLI --verify rejects blocked commands and exits with code 2."""
        from zelyr.agents.native_orchestrator import NativeOrchestrator
        from zelyr.cli.main import _execute_headless_run

        async def _mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed successfully."}}]}

        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", _mock_stream)

        code = await _execute_headless_run(
            prompt="smoke test prompt",
            orchestrator=orch,
            verify_spec="command_output:whoami:root",
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "[verification failed] command_output on whoami: FAIL" in captured.out

    @pytest.mark.asyncio
    async def test_15_consistency_between_verifier_results_and_final_task_status(self, monkeypatch, capsys):
        """Final task status strictly reflects outcome verifier results."""
        from zelyr.agents.native_orchestrator import NativeOrchestrator
        from zelyr.cli.main import _execute_headless_run

        async def _mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed successfully."}}]}

        # 1. Allowlisted command that succeeds -> exit 0
        orch1 = NativeOrchestrator()
        monkeypatch.setattr(orch1, "generate_stream", _mock_stream)
        code_pass = await _execute_headless_run(
            prompt="smoke test prompt",
            orchestrator=orch1,
            verify_spec="command_output:echo 'SUCCESS_TOKEN':SUCCESS_TOKEN",
        )
        assert code_pass == 0
        captured1 = capsys.readouterr()
        assert "command_output on echo 'SUCCESS_TOKEN': PASS" in captured1.out

        # 2. Allowlisted command whose pattern is absent -> exit 2
        orch2 = NativeOrchestrator()
        monkeypatch.setattr(orch2, "generate_stream", _mock_stream)
        code_fail = await _execute_headless_run(
            prompt="smoke test prompt",
            orchestrator=orch2,
            verify_spec="command_output:echo 'WRONG_TOKEN':EXPECTED_TOKEN",
        )
        assert code_fail == 2
        captured2 = capsys.readouterr()
        assert "command_output on echo 'WRONG_TOKEN': FAIL" in captured2.out
