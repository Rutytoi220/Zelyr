"""End-to-end developer workflow integration test suite for Zelyr.

Tests live multi-turn autonomous execution and self-repair scenarios strictly inside
isolated temporary workspaces:
1. test_git_merge_conflict_resolution: Resolving git conflict markers and committing clean state.
2. test_python_test_diagnostic_and_self_repair: Diagnostic on failing test, code patching, and re-verification.
3. test_atomic_mutation_and_rollback: Atomic file write with automatic backup and state restoration.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import AsyncMock

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.observation_framing import ObservationFramingEngine
from zelyr.engine.verifier import OutcomeVerifier, verify_outcome
from zelyr.tools.workspace_file import PRIMARY_BACKUP_DIR, ManageWorkspaceFileTool


def _run_git_cmd(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Helper to run git commands in a specific test directory."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.asyncio
async def test_git_merge_conflict_resolution(monkeypatch):
    """Scenario 1: Autonomous Git Merge Conflict Resolution.

    1. Induces real git merge conflict on 'app.py' across two branches.
    2. Model observes conflict via git status.
    3. Reads conflict markers via manage_workspace_file(read).
    4. Writes clean resolution via manage_workspace_file(write).
    5. Commits resolution via execute_command.
    6. Verifies clean working tree via verify_outcome(command_output).
    """
    with tempfile.TemporaryDirectory(prefix="zelyr_git_workflow_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str).resolve()

        # Step 0: Setup Git repo and induce real merge conflict
        _run_git_cmd(tmp_dir, "init", "-b", "main")
        _run_git_cmd(tmp_dir, "config", "user.name", "Zelyr Agent")
        _run_git_cmd(tmp_dir, "config", "user.email", "agent@zelyr.internal")

        app_file = tmp_dir / "app.py"
        app_file.write_text("def calculate(x):\n    return x * 1\n", encoding="utf-8")
        _run_git_cmd(tmp_dir, "add", "app.py")
        _run_git_cmd(tmp_dir, "commit", "-m", "Initial baseline calculate")

        # Feature branch edit
        _run_git_cmd(tmp_dir, "checkout", "-b", "feature-a")
        app_file.write_text("def calculate(x):\n    return x * 10\n", encoding="utf-8")
        _run_git_cmd(tmp_dir, "commit", "-am", "Feature A multiplier")

        # Main branch conflicting edit
        _run_git_cmd(tmp_dir, "checkout", "main")
        app_file.write_text("def calculate(x):\n    return x * 100\n", encoding="utf-8")
        _run_git_cmd(tmp_dir, "commit", "-am", "Main branch multiplier")

        # Induce merge conflict
        proc_merge = subprocess.run(
            ["git", "merge", "feature-a"],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
        )
        assert proc_merge.returncode != 0
        raw_conflict = app_file.read_text(encoding="utf-8")
        assert "<<<<<<<" in raw_conflict
        assert "=======" in raw_conflict
        assert ">>>>>>>" in raw_conflict

        # Direct the autonomous agent across multi-turn ReAct execution
        step = 0
        recorded_observations = []

        async def deterministic_stream_provider(payload):
            nonlocal step
            step += 1

            if step == 1:
                # Step 1: Check git status
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_git_status",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({
                                        "command": "git status --short",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 2:
                # Step 2: Read conflict markers from app.py
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_read_conflict",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "read",
                                        "path": "app.py",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 3:
                # Step 3: Write resolved file content
                resolved_content = "def calculate(x):\n    return x * 1000\n"
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_resolution",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": "app.py",
                                        "content": resolved_content,
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 4:
                # Step 4: Add and commit the resolved conflict
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_commit_resolution",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({
                                        "command": "git add app.py && git commit -m 'resolve merge conflict in calculate'",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 5:
                # Step 5: Verify clean working tree outcome
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_verify_clean",
                                "type": "function",
                                "function": {
                                    "name": "verify_outcome",
                                    "arguments": json.dumps({
                                        "check": "command_output",
                                        "target": "git status",
                                        "expected": "nothing to commit, working tree clean",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                # Final synthesis turn
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Merge conflict in app.py successfully resolved, committed, and verified."
                        }
                    }]
                }

        orch = NativeOrchestrator(stream_provider=deterministic_stream_provider)
        monkeypatch.setattr(orch, "get_active_window_context", AsyncMock(return_value=None))

        payload = {
            "messages": [
                {"role": "user", "content": f"Resolve the git merge conflict in {tmp_dir}"}
            ]
        }

        agent_output = []
        async for chunk in orch.generate_stream(payload, max_depth=10):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                agent_output.append(delta["content"])

        # Collect tool observation records from messages payload
        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 5

        # Verification Assertions
        # 1. Clean working tree verified by OutcomeVerifier
        verifier_res = OutcomeVerifier.verify(
            check="command_output",
            target="git status",
            expected="nothing to commit, working tree clean",
            cwd=tmp_dir,
        )
        assert verifier_res.success is True
        assert verifier_res.observed.get("exit_code") == 0

        # 2. File content on disk contains resolved multiplier
        content_res = OutcomeVerifier.verify(
            check="file_contains",
            target="app.py",
            expected="return x * 1000",
            cwd=tmp_dir,
        )
        assert content_res.success is True

        # 3. No conflict markers remain on disk
        conflict_res = OutcomeVerifier.verify(
            check="file_contains",
            target="app.py",
            expected="<<<<<<<",
            cwd=tmp_dir,
        )
        assert conflict_res.success is False

        # 4. Git log confirms merge commit
        log_proc = _run_git_cmd(tmp_dir, "log", "--oneline", "-n", "1")
        assert "resolve merge conflict" in log_proc.stdout


@pytest.mark.asyncio
async def test_python_test_diagnostic_and_self_repair(monkeypatch):
    """Scenario 2: Broken Python Test Diagnostic & Self-Repair.

    1. Creates math_lib.py with an intentional subtraction bug and test_math_lib.py.
    2. Runs test suite via execute_command -> fails with exit code 1.
    3. Model inspects error and file via manage_workspace_file(read).
    4. Model surgically patches bug via manage_workspace_file(patch).
    5. Re-runs test suite via execute_command -> passes (1 passed).
    6. Confirms passing state via verify_outcome(command_output).
    """
    with tempfile.TemporaryDirectory(prefix="zelyr_repair_workflow_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str).resolve()

        # Buggy implementation: subtraction instead of addition
        math_file = tmp_dir / "math_lib.py"
        math_file.write_text("def add(a, b):\n    return a - b  # BUG: subtraction\n", encoding="utf-8")

        test_file = tmp_dir / "test_math_lib.py"
        test_file.write_text(
            "from math_lib import add\n\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n",
            encoding="utf-8",
        )

        test_cmd = f"{sys.executable} -B -m pytest -o cache_dir={tmp_dir}/.pytest_cache test_math_lib.py"

        step = 0
        async def repair_stream_provider(payload):
            nonlocal step
            step += 1

            if step == 1:
                # Step 1: Run pytest (which will fail)
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_initial_test",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({
                                        "command": test_cmd,
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 2:
                # Step 2: Read buggy source file
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_read_math_lib",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "read",
                                        "path": "math_lib.py",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 3:
                # Step 3: Patch bug surgically
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_patch_math_lib",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "patch",
                                        "path": "math_lib.py",
                                        "search_text": "return a - b",
                                        "replace_text": "return a + b",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 4:
                # Step 4: Re-run test suite
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_rerun_test",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({
                                        "command": test_cmd,
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 5:
                # Step 5: Deterministically verify test outcome
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_verify_test",
                                "type": "function",
                                "function": {
                                    "name": "verify_outcome",
                                    "arguments": json.dumps({
                                        "check": "command_output",
                                        "target": test_cmd,
                                        "expected": "1 passed",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Bug successfully diagnosed, patched, and verified with 1 passed test."
                        }
                    }]
                }

        orch = NativeOrchestrator(stream_provider=repair_stream_provider)
        monkeypatch.setattr(orch, "get_active_window_context", AsyncMock(return_value=None))

        payload = {
            "messages": [
                {"role": "user", "content": f"Fix the failing test in {tmp_dir}"}
            ]
        }

        async for _ in orch.generate_stream(payload, max_depth=10):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 5

        from zelyr.engine.observation_framing import ObservationFramingEngine

        # Check turn 1 observed failure
        obs1 = ObservationFramingEngine.parse_observation(tool_messages[0]["content"])
        assert obs1.get("status") == "error"
        assert obs1.get("exit_code") != 0

        # Check turn 4 observed success
        obs4 = ObservationFramingEngine.parse_observation(tool_messages[3]["content"])
        assert obs4.get("status") == "success"
        assert obs4.get("exit_code") == 0

        # Verification Assertions
        verifier_res = OutcomeVerifier.verify(
            check="command_output",
            target=test_cmd,
            expected="1 passed",
            cwd=tmp_dir,
        )
        assert verifier_res.success is True

        # Check patched file on disk
        patched_code = math_file.read_text(encoding="utf-8")
        assert "return a + b" in patched_code
        assert "return a - b" not in patched_code


@pytest.mark.asyncio
async def test_atomic_mutation_and_rollback(monkeypatch):
    """Scenario 3: Atomic File Mutation & Automated Rollback.

    1. Creates production config.toml.
    2. Modifies config.toml via manage_workspace_file(write).
    3. Confirms automatic backup creation in PRIMARY_BACKUP_DIR.
    4. Rolls back file via manage_workspace_file(rollback).
    5. Confirms state restoration via verify_outcome(file_contains).
    """
    with tempfile.TemporaryDirectory(prefix="zelyr_rollback_workflow_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str).resolve()

        config_file = tmp_dir / "config.toml"
        orig_content = 'mode = "production"\nworkers = 4\n'
        config_file.write_text(orig_content, encoding="utf-8")

        backup_file_path = None
        step = 0

        async def rollback_stream_provider(payload):
            nonlocal step, backup_file_path
            step += 1

            if step == 1:
                # Step 1: Overwrite config file (which creates backup)
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_write_mutation",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": "config.toml",
                                        "content": 'mode = "broken"\nworkers = 0\n',
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 2:
                # Step 2: Extract backup path from previous observation & verify backup exists
                last_tool_msg = next(m for m in reversed(payload["messages"]) if m.get("role") == "tool")
                obs = ObservationFramingEngine.parse_observation(last_tool_msg["content"])
                out_field = obs.get("output", {})
                if isinstance(out_field, str):
                    try:
                        out_field = json.loads(out_field)
                    except Exception:
                        pass
                if isinstance(out_field, dict) and "backup" in out_field:
                    backup_file_path = out_field["backup"]
                elif isinstance(obs.get("result"), dict) and "backup" in obs["result"]:
                    backup_file_path = obs["result"]["backup"]
                else:
                    backup_file_path = str(list(PRIMARY_BACKUP_DIR.glob("config.toml.*.bak"))[0])

                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_verify_backup_exists",
                                "type": "function",
                                "function": {
                                    "name": "verify_outcome",
                                    "arguments": json.dumps({
                                        "check": "file_exists",
                                        "target": backup_file_path,
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 3:
                # Step 3: Trigger rollback
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_rollback",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "rollback",
                                        "path": "config.toml",
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif step == 4:
                # Step 4: Verify production content restored
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_verify_restored",
                                "type": "function",
                                "function": {
                                    "name": "verify_outcome",
                                    "arguments": json.dumps({
                                        "check": "file_contains",
                                        "target": "config.toml",
                                        "expected": 'mode = "production"',
                                        "cwd": str(tmp_dir),
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Configuration successfully restored to production state."
                        }
                    }]
                }

        orch = NativeOrchestrator(stream_provider=rollback_stream_provider)
        monkeypatch.setattr(orch, "get_active_window_context", AsyncMock(return_value=None))

        payload = {
            "messages": [
                {"role": "user", "content": f"Mutate and rollback {config_file}"}
            ]
        }

        async for _ in orch.generate_stream(payload, max_depth=10):
            pass

        tool_messages = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 4

        # Verification Assertions
        # 1. Target file contains original content exactly
        assert config_file.read_text(encoding="utf-8") == orig_content

        # 2. OutcomeVerifier confirms file_contains
        verify_res = OutcomeVerifier.verify(
            check="file_contains",
            target="config.toml",
            expected='mode = "production"',
            cwd=tmp_dir,
        )
        assert verify_res.success is True

        # 3. OutcomeVerifier confirms file_contains for workers count
        verify_workers = OutcomeVerifier.verify(
            check="file_contains",
            target="config.toml",
            expected="workers = 4",
            cwd=tmp_dir,
        )
        assert verify_workers.success is True
