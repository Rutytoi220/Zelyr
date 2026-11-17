"""Comprehensive test suite for Terminal & Shell Execution Hardening:
- Non-Interactive Environment Guards (PAGER, TERM, DEBIAN_FRONTEND, CI, PYTHONUNBUFFERED)
- DEVNULL stdin isolation (prevents blocking on unhandled input prompts)
- Deterministic ANSI Escape Sequence Stripping (ECMA-48 regex)
- Pseudo-Terminal (PTY) Mode execution (isatty == True)
- PTY Timeout handling & Process-Group Reaping
- Strict Hash Provenance Upgrade and automated timestamped backup
"""

from __future__ import annotations

import os
import subprocess
import uuid
import pytest

from zelyr.core.plugins import (
    _EXECUTE_COMMAND_TEMPLATE,
    HISTORICAL_TEMPLATE_HASHES,
    execute_tool,
    load_plugins,
    safe_provision_tool,
    strip_ansi_codes,
)


@pytest.fixture(autouse=True)
def ensure_plugins_loaded():
    """Ensure dynamic tools are loaded and up to date for all tests."""
    load_plugins()


class TestNonInteractiveEnvironmentGuards:
    """Tests verifying non-interactive defaults in execute_command."""

    @pytest.mark.asyncio
    async def test_non_interactive_env_variables_injected(self, monkeypatch):
        """Execute printenv and assert PAGER, TERM, DEBIAN_FRONTEND, CI, and PYTHONUNBUFFERED are set."""
        # Ensure CI is absent from inherited environment so default CI=1 is injected
        monkeypatch.delenv("CI", raising=False)
        cmd = (
            "python3 -c '"
            "import os; "
            "print(\"PAGER=\" + str(os.environ.get(\"PAGER\"))); "
            "print(\"TERM=\" + str(os.environ.get(\"TERM\"))); "
            "print(\"DEBIAN_FRONTEND=\" + str(os.environ.get(\"DEBIAN_FRONTEND\"))); "
            "print(\"CI=\" + str(os.environ.get(\"CI\"))); "
            "print(\"PYTHONUNBUFFERED=\" + str(os.environ.get(\"PYTHONUNBUFFERED\")))"
            "'"
        )
        result = await execute_tool("execute_command", command=cmd)

        assert "Exit code 0" in result
        assert "PAGER=cat" in result
        assert "TERM=dumb" in result
        assert "DEBIAN_FRONTEND=noninteractive" in result
        assert "CI=1" in result
        assert "PYTHONUNBUFFERED=1" in result

    @pytest.mark.asyncio
    async def test_existing_ci_env_variable_is_preserved(self, monkeypatch):
        """When CI already exists in the environment (e.g. CI=true in GitHub Actions), its value is preserved."""
        cmd = "python3 -c 'import os; print(\"CI=\" + str(os.environ.get(\"CI\")))'"

        monkeypatch.setenv("CI", "true")
        result_true = await execute_tool("execute_command", command=cmd)
        assert "Exit code 0" in result_true
        assert "CI=true" in result_true

        monkeypatch.setenv("CI", "custom_ci_runner")
        result_custom = await execute_tool("execute_command", command=cmd)
        assert "Exit code 0" in result_custom
        assert "CI=custom_ci_runner" in result_custom


class TestStdinIsolation:
    """Tests verifying DEVNULL stdin prevents hanging on interactive prompts."""

    @pytest.mark.asyncio
    async def test_read_command_receives_immediate_eof(self):
        """Standard bash 'read line' receives immediate EOF and does not block."""
        result = await execute_tool("execute_command", command="read line", timeout=3.0)

        # In bash, read returns 1 on EOF without data
        assert "Exit code 1" in result or "Exit code 0" in result
        assert "timed out after" not in result


class TestANSICodeStripping:
    """Tests verifying ECMA-48 ANSI escape sequences are completely stripped from outputs."""

    def test_strip_ansi_codes_helper(self):
        """Unit test for strip_ansi_codes function on various escape sequences."""
        raw_text = "\x1b[31;1mERROR:\x1b[0m \x1b[32mOperation successful\x1b[0m\n\x1b[2J\x1b[HDetails:\tIndent"
        clean_text = strip_ansi_codes(raw_text)

        assert "\x1b" not in clean_text
        assert "ERROR: Operation successful\nDetails:\tIndent" in clean_text

    @pytest.mark.asyncio
    async def test_execute_command_strips_ansi_color_sequences(self):
        """Command emitting ANSI color codes returns clean output without escape sequences."""
        cmd = 'printf "\\033[31;1mERROR\\033[0m\\n"'
        result = await execute_tool("execute_command", command=cmd)

        assert "Exit code 0" in result
        assert "ERROR" in result
        assert "\x1b" not in result
        assert "\\033" not in result


class TestPTYExecutionMode:
    """Tests for pseudo-terminal (PTY) execution mode."""

    @pytest.mark.asyncio
    async def test_pty_true_reports_isatty_true(self):
        """When pty=True is passed, stdout.isatty() is True."""
        cmd = "python3 -c 'import sys; print(\"ISATTY=\" + str(sys.stdout.isatty()))'"
        result = await execute_tool("execute_command", command=cmd, pty=True)

        assert "Exit code 0" in result
        assert "ISATTY=True" in result

    @pytest.mark.asyncio
    async def test_pty_false_reports_isatty_false(self):
        """When pty=False is passed, stdout.isatty() is False."""
        cmd = "python3 -c 'import sys; print(\"ISATTY=\" + str(sys.stdout.isatty()))'"
        result = await execute_tool("execute_command", command=cmd, pty=False)

        assert "Exit code 0" in result
        assert "ISATTY=False" in result

    @pytest.mark.asyncio
    async def test_pty_timeout_and_process_group_reaping(self):
        """Command in PTY mode times out cleanly, returns exit code 124, and reaps all child processes."""
        marker = "pty_reap_probe_" + uuid.uuid4().hex[:8]
        cmd = f'python3 -c "import time, os; os.fork(); time.sleep(30)" # {marker}'

        result = await execute_tool("execute_command", command=cmd, timeout=1.5, pty=True)

        assert "Exit code 124" in result
        assert "timed out after" in result

        # Check process table strictly for our marker to ensure no surviving zombies/orphans
        ps = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True)
        current_pid = os.getpid()
        surviving = [
            line for line in ps.stdout.splitlines()
            if marker in line and str(current_pid) not in line and "ps -eo" not in line
        ]

        # In case any survived, clean up
        for line in surviving:
            parts = line.strip().split()
            if parts:
                try:
                    os.kill(int(parts[0]), 9)
                except OSError:
                    pass

        assert len(surviving) == 0, f"Found surviving processes: {surviving}"


class TestSafeHashProvenanceUpgrade:
    """Tests verifying automated hash upgrade from Day 2 template."""

    def test_day2_template_upgrades_cleanly_with_backup(self, tmp_path):
        """A disk tool matching the Day 2 hash upgrades to current hardened template and creates .bak."""
        target = tmp_path / "execute_command.py"

        # Day 2 v2 template code
        day2_code = (
            "import asyncio\n"
            "import os\n"
            "import signal\n\n"
            "IS_CORE = True\n"
            "TIER = 0\n"
            "TOOL_SCHEMA = {\n"
            '    "type": "function",\n'
            '    "function": {\n'
            '        "name": "execute_command",\n'
            '        "description": "Executes a shell command via bash and returns stdout and stderr.",\n'
            '        "parameters": {\n'
            '            "type": "object",\n'
            '            "properties": {\n'
            '                "command": {"type": "string", "description": "Command to execute"}\n'
            '            },\n'
            '            "required": ["command"]\n'
            '        }\n'
            '    }\n'
            "}\n"
            'TUI_HINT = "⚙️ Executing system command..."\n'
            "REQUIRED_RING = 0\n\n"
            "async def execute(command: str = '', timeout: float = 60.0, **kwargs) -> str:\n"
            "    start_new_session = hasattr(os, 'setsid')\n"
            "    try:\n"
            "        proc = await asyncio.create_subprocess_exec(\n"
            "            'bash', '-c', command,\n"
            "            stdout=asyncio.subprocess.PIPE,\n"
            "            stderr=asyncio.subprocess.PIPE,\n"
            "            start_new_session=start_new_session,\n"
            "        )\n"
            "    except Exception as e:\n"
            '        return f"Error executing command: {e}"\n\n'
            "    max_bytes = 2 * 1024 * 1024\n\n"
            "    async def _read_stream(stream):\n"
            "        buf = bytearray()\n"
            "        try:\n"
            "            while True:\n"
            "                chunk = await stream.read(65536)\n"
            "                if not chunk:\n"
            "                    break\n"
            "                if len(buf) < max_bytes:\n"
            "                    buf.extend(chunk[:max_bytes - len(buf)])\n"
            "        except (asyncio.CancelledError, Exception):\n"
            "            pass\n"
            "        return bytes(buf)\n\n"
            "    async def _cleanup_proc():\n"
            "        has_pg = hasattr(os, 'killpg') and hasattr(os, 'getpgid')\n"
            "        try:\n"
            "            if has_pg:\n"
            "                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)\n"
            "            else:\n"
            "                proc.terminate()\n"
            "        except (ProcessLookupError, OSError):\n"
            "            pass\n\n"
            "        try:\n"
            "            await asyncio.wait_for(proc.wait(), timeout=1.5)\n"
            "        except (asyncio.TimeoutError, TimeoutError):\n"
            "            try:\n"
            "                if has_pg:\n"
            "                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)\n"
            "                else:\n"
            "                    proc.kill()\n"
            "            except (ProcessLookupError, OSError):\n"
            "                pass\n"
            "            try:\n"
            "                await proc.wait()\n"
            "            except Exception:\n"
            "                pass\n\n"
            "    try:\n"
            "        stdout_bytes, stderr_bytes = await asyncio.wait_for(\n"
            "            asyncio.gather(_read_stream(proc.stdout), _read_stream(proc.stderr)),\n"
            "            timeout=timeout,\n"
            "        )\n"
            "        await proc.wait()\n"
            "        out = stdout_bytes.decode('utf-8', errors='replace')\n"
            "        err = stderr_bytes.decode('utf-8', errors='replace')\n"
            "        body = (out or '') + (err or '')\n"
            '        return f"Exit code {proc.returncode}\\n{body}".strip() if body else f"Exit code {proc.returncode}"\n'
            "    except (asyncio.TimeoutError, TimeoutError):\n"
            "        await _cleanup_proc()\n"
            '        return f"Error executing command: Command \'{command}\' timed out after {timeout} seconds"\n'
            "    except (asyncio.CancelledError, BaseException):\n"
            "        await _cleanup_proc()\n"
            "        raise\n"
        )

        import hashlib
        day2_hash = hashlib.sha256(day2_code.encode("utf-8")).hexdigest()
        assert day2_hash in HISTORICAL_TEMPLATE_HASHES["execute_command"]

        # Write day2 code to target file
        target.write_text(day2_code, encoding="utf-8")

        # Upgrade using safe_provision_tool with canonical template
        res = safe_provision_tool(target, _EXECUTE_COMMAND_TEMPLATE)
        assert res is True

        # Target now contains the hardened template
        assert target.read_text(encoding="utf-8") == _EXECUTE_COMMAND_TEMPLATE

        # A timestamped .bak file was created containing the day2 code
        bak_files = list(tmp_path.glob("execute_command.py.bak.*"))
        assert len(bak_files) == 1
        assert bak_files[0].read_text(encoding="utf-8") == day2_code
