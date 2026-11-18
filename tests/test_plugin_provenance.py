"""Unit test suite for strict hash-based plugin provenance, atomic upgrade, and rollback."""

import os
import stat
from unittest.mock import patch
import pytest

from zelyr.core.plugins import HISTORICAL_TEMPLATE_HASHES, safe_provision_tool

V1_BASELINE_CODE = (
    "import subprocess\n\n"
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
    "        }\n"
    "    }\n"
    "}\n"
    'TUI_HINT = "⚙️ Executing system command..."\n'
    "REQUIRED_RING = 0\n\n"
    "async def execute(command: str = '', **kwargs) -> str:\n"
    "    try:\n"
    "        proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)\n"
    "        body = (proc.stdout or '') + (proc.stderr or '')\n"
    '        return f"Exit code {proc.returncode}\\n{body}".strip() if body else f"Exit code {proc.returncode}"\n'
    "    except Exception as e:\n"
    '        return f"Error executing command: {e}"\n'
)

V2_CANONICAL_CODE = (
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
    "        }\n"
    "    }\n"
    "}\n"
    'TUI_HINT = "⚙️ Executing system command..."\n'
    "REQUIRED_RING = 0\n\n"
    "async def execute(command: str = '', timeout: float = 60.0, **kwargs) -> str:\n"
    "    return 'ok'\n"
)


class TestPluginProvenance:
    def test_new_file_provisioning_success(self, tmp_path):
        """If target file does not exist, provision canonical template atomically."""
        target = tmp_path / "execute_command.py"
        assert not target.exists()

        result = safe_provision_tool(target, V2_CANONICAL_CODE)
        assert result is True
        assert target.exists()
        assert target.read_text(encoding="utf-8") == V2_CANONICAL_CODE

    def test_exact_historical_hash_triggers_safe_upgrade_and_backup(self, tmp_path):
        """Historical generated template match triggers safe upgrade and creates a valid .bak file."""
        target = tmp_path / "execute_command.py"
        target.write_text(V1_BASELINE_CODE, encoding="utf-8")

        result = safe_provision_tool(target, V2_CANONICAL_CODE)
        assert result is True
        assert target.read_text(encoding="utf-8") == V2_CANONICAL_CODE

        # Verify a .bak.<timestamp> file was created containing the original v1 baseline
        bak_files = list(tmp_path.glob("execute_command.py.bak.*"))
        assert len(bak_files) == 1
        assert bak_files[0].read_text(encoding="utf-8") == V1_BASELINE_CODE

    def test_modified_file_unknown_hash_strictly_preserved(self, tmp_path):
        """Modified file with unknown hash is strictly preserved without modification."""
        target = tmp_path / "execute_command.py"
        user_custom_code = "# User custom modification\ndef custom_logic(): pass\n"
        target.write_text(user_custom_code, encoding="utf-8")

        result = safe_provision_tool(target, V2_CANONICAL_CODE)
        assert result is False
        assert target.read_text(encoding="utf-8") == user_custom_code

        # No backup should have been created
        bak_files = list(tmp_path.glob("execute_command.py.bak.*"))
        assert len(bak_files) == 0

    def test_symlinks_and_non_regular_files_detected_and_untouched(self, tmp_path):
        """Symlinks and non-regular files are detected and untouched."""
        real_file = tmp_path / "external_tool.py"
        real_file.write_text(V1_BASELINE_CODE, encoding="utf-8")

        symlink_target = tmp_path / "execute_command.py"
        symlink_target.symlink_to(real_file)
        assert symlink_target.is_symlink()

        result = safe_provision_tool(symlink_target, V2_CANONICAL_CODE)
        assert result is False
        assert symlink_target.is_symlink()
        assert symlink_target.resolve() == real_file
        assert real_file.read_text(encoding="utf-8") == V1_BASELINE_CODE

    def test_posix_permissions_preserved_across_upgrades(self, tmp_path):
        """POSIX permissions are preserved when upgrading a recognized historical template."""
        target = tmp_path / "execute_command.py"
        target.write_text(V1_BASELINE_CODE, encoding="utf-8")
        custom_mode = 0o750
        os.chmod(target, custom_mode)

        result = safe_provision_tool(target, V2_CANONICAL_CODE)
        assert result is True

        new_mode = stat.S_IMODE(os.stat(target).st_mode)
        assert new_mode == custom_mode

    def test_simulated_write_failure_triggers_clean_rollback(self, tmp_path):
        """Simulated replacement failure retains backup, rolls back file, and raises RuntimeError."""
        target = tmp_path / "execute_command.py"
        target.write_text(V1_BASELINE_CODE, encoding="utf-8")

        with patch("os.replace", side_effect=OSError("Disk write error")):
            with pytest.raises(RuntimeError, match="Atomic replacement failed"):
                safe_provision_tool(target, V2_CANONICAL_CODE)

        # Original file should still be preserved with v1 content
        assert target.exists()
        assert target.read_text(encoding="utf-8") == V1_BASELINE_CODE

        # Backup file must be retained
        bak_files = list(tmp_path.glob("execute_command.py.bak.*"))
        assert len(bak_files) == 1
        assert bak_files[0].read_text(encoding="utf-8") == V1_BASELINE_CODE
