"""Integration tests for OutcomeVerifier and ErrorClassifier with NativeOrchestrator, ToolRegistry, and CLI."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import AsyncGenerator
import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import _execute_headless_run
from zelyr.db.memory import MemoryStore, init_db
from zelyr.engine.verifier import OutcomeVerifier
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.verify_outcome import VerifyOutcomeTool


@pytest.fixture
def isolated_db():
    """Create an isolated temporary SQLite database for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_memory_hydration.db"
        init_db(str(db_file))
        yield str(db_file)


class TestVerifyOutcomeToolRegistration:
    """Verify registry wiring and tool schemas in ToolRegistry."""

    def test_verify_outcome_in_builtin_tools(self):
        registry = ToolRegistry()
        builtins = registry.get_builtin_tools()
        assert "verify_outcome" in builtins

        tool = registry.get_tool("verify_outcome")
        assert tool is not None
        assert isinstance(tool, VerifyOutcomeTool)
        assert tool.tool_id == "verify_outcome"
        assert tool.REQUIRED_RING == 1
        assert tool.tier == 1


class TestNativeOrchestratorVerificationAliases:
    """Verify alias mapping and argument canonicalization in NativeOrchestrator."""

    def test_tool_name_normalization_for_verify_aliases(self):
        assert NativeOrchestrator.normalize_tool_name("assert_outcome") == "verify_outcome"
        assert NativeOrchestrator.normalize_tool_name("check_state") == "verify_outcome"
        assert NativeOrchestrator.normalize_tool_name("verify_state") == "verify_outcome"
        assert NativeOrchestrator.normalize_tool_name("verify") == "verify_outcome"
        assert NativeOrchestrator.normalize_tool_name("tool.verify_outcome") == "verify_outcome"
        assert NativeOrchestrator.normalize_tool_name("functions.check_state") == "verify_outcome"

    def test_tool_argument_normalization_for_verify_outcome(self):
        raw_args = {
            "assertion": "file_contains",
            "file": "/tmp/output.log",
            "content": "SUCCESS",
        }
        norm = NativeOrchestrator.normalize_tool_arguments("verify_outcome", raw_args)
        assert norm["check"] == "file_contains"
        assert norm["target"] == "/tmp/output.log"
        assert norm["expected"] == "SUCCESS"

    @pytest.mark.asyncio
    async def test_orchestrator_react_with_verify_tool(self, monkeypatch, tmp_path: Path):
        """Simulate an agent generating a tool call for verify_outcome."""
        orch = NativeOrchestrator()
        test_file = tmp_path / "react_target.txt"
        test_file.write_text("system initialized", encoding="utf-8")

        mock_stream_calls = [
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "check_state",
                                "arguments": json.dumps({"check": "file_exists", "target": str(test_file)}),
                            },
                        }]
                    }
                }]
            },
            {
                "choices": [{
                    "delta": {
                        "content": "The system state has been verified successfully.",
                    }
                }]
            },
        ]

        async def mock_generate_stream(payload):
            if len(payload["messages"]) == 1:
                yield mock_stream_calls[0]
            else:
                yield mock_stream_calls[1]

        monkeypatch.setattr(orch, "generate_stream", mock_generate_stream)

        async def mock_execute_tool(name: str, **kwargs):
            clean = NativeOrchestrator.normalize_tool_name(name)
            assert clean == "verify_outcome"
            res = OutcomeVerifier.verify(**kwargs)
            return json.dumps(res.to_dict())

        import zelyr.agents.native_orchestrator as nat_orch_mod
        monkeypatch.setattr(nat_orch_mod, "execute_tool", mock_execute_tool)

        exit_code = await _execute_headless_run(
            prompt="Verify system state",
            orchestrator=orch,
        )
        assert exit_code == 0


class TestCLIOutcomeVerification:
    """Verify CLI --verify flag in text and JSON modes."""

    @pytest.mark.asyncio
    async def test_headless_run_with_verify_positive(self, monkeypatch, tmp_path: Path, capsys):
        test_file = tmp_path / "cli_verify.txt"
        test_file.write_text("ok", encoding="utf-8")

        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed successfully."}}]}

        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        exit_code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            verify_spec=f"file_exists:{test_file}",
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert f"[verified] file_exists on {test_file}: PASS" in captured.out

    @pytest.mark.asyncio
    async def test_headless_run_with_verify_negative_exit_2(self, monkeypatch, tmp_path: Path, capsys):
        missing_file = tmp_path / "absent.txt"

        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task completed successfully."}}]}

        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        exit_code = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            verify_spec=f"file_exists:{missing_file}",
        )
        captured = capsys.readouterr()
        assert exit_code == 2
        assert f"[verification failed] file_exists on {missing_file}: FAIL" in captured.out

    @pytest.mark.asyncio
    async def test_headless_run_with_verify_json_mode(self, monkeypatch, tmp_path: Path, capsys):
        test_file = tmp_path / "cli_json.txt"
        test_file.write_text("payload", encoding="utf-8")

        async def mock_stream(payload):
            yield {"choices": [{"delta": {"content": "Task done."}}]}

        orch = NativeOrchestrator()
        monkeypatch.setattr(orch, "generate_stream", mock_stream)

        exit_code_pass = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            output_json=True,
            verify_spec=f"file_exists:{test_file}",
        )
        captured_pass = capsys.readouterr()
        assert exit_code_pass == 0
        data_pass = json.loads(captured_pass.out.strip())
        assert "verification" in data_pass
        assert data_pass["verification"]["success"] is True

        missing_file = tmp_path / "absent_json.txt"
        exit_code_fail = await _execute_headless_run(
            prompt="Do task",
            orchestrator=orch,
            output_json=True,
            verify_spec=f"file_exists:{missing_file}",
        )
        captured_fail = capsys.readouterr()
        assert exit_code_fail == 2
        data_fail = json.loads(captured_fail.out.strip())
        assert "verification" in data_fail
        assert data_fail["verification"]["success"] is False


class TestOrchestratorErrorObservationFormatting:
    """Verify NativeOrchestrator tool observation formatting with ErrorClassifier metadata."""

    def test_format_tool_observation_includes_classification_metadata(self):
        raw_output = "Exit code 127\nbash: non_existent_tool: command not found"
        formatted = NativeOrchestrator.format_tool_observation("execute_command", raw_output)
        from zelyr.engine.observation_framing import ObservationFramingEngine

        data = ObservationFramingEngine.parse_observation(formatted)
        assert data["status"] == "error"
        assert data["error_type"] == "COMMAND_NOT_FOUND"
        assert data["exit_code"] == 127
        assert "Command not found in PATH" in data["remedy_hint"]
        assert "execute_command" in data["allowed_actions"]
        assert "Recovery Hint:" in formatted


# ==============================================================================
# Commit 3 Actuator Integration Tests (Hyprland, CDP, Memory, Tool Crafter, Routine)
# ==============================================================================

class TestDesktopHyprlandOrchestratorIntegration:
    """Verify NativeOrchestrator resolves desktop aliases."""

    def test_native_orchestrator_desktop_alias_resolution(self):
        assert NativeOrchestrator.normalize_tool_name("switch_workspace") == "manage_desktop_window"
        assert NativeOrchestrator.normalize_tool_name("focus_window") == "manage_desktop_window"
        assert NativeOrchestrator.normalize_tool_name("tool.desktop_window") == "manage_desktop_window"
        assert NativeOrchestrator.normalize_tool_name("functions.manage_desktop_window") == "manage_desktop_window"


class TestBrowserCDPOrchestratorIntegration:
    """Verify NativeOrchestrator resolves browser aliases and normalizes arguments."""

    def test_native_orchestrator_browser_aliases(self):
        orch = NativeOrchestrator()
        aliases = ["browser", "cdp", "web_browser", "chrome", "chromium"]
        for alias in aliases:
            canonical_name = orch.normalize_tool_name(alias)
            assert canonical_name == "interact_with_browser"

        args_snap = orch.normalize_tool_arguments("interact_with_browser", {"action": "snapshot"}, original_name="browser")
        assert args_snap["action"] == "get_page_snapshot"

        args_click = orch.normalize_tool_arguments("interact_with_browser", {"action": "click", "element_id": 5}, original_name="cdp")
        assert args_click["action"] == "click_element"
        assert args_click["element_id"] == 5

        args_nav = orch.normalize_tool_arguments("interact_with_browser", {"action": "navigate", "url": "https://example.com"}, original_name="web_browser")
        assert args_nav["action"] == "navigate_url"
        assert args_nav["url"] == "https://example.com"

        args_tabs = orch.normalize_tool_arguments("interact_with_browser", {"action": "tabs"}, original_name="chrome")
        assert args_tabs["action"] == "list_tabs"


class TestMemoryHydrationOrchestratorIntegration:
    """Verify NativeOrchestrator resolves memory aliases and normalizes 'fact' -> 'content'."""

    def test_native_orchestrator_alias_and_argument_normalization(self):
        orch = NativeOrchestrator()
        aliases = ["remember", "remember_fact", "search_memory", "save_memory", "recall_memory"]
        for alias in aliases:
            assert orch.normalize_tool_name(alias) == "manage_memory"

        args = orch.normalize_tool_arguments("manage_memory", {"fact": "Tokyo Night theme", "category": "preference"})
        assert args["content"] == "Tokyo Night theme"
        assert "fact" not in args
        assert args["action"] == "store"

        args_query = orch.normalize_tool_arguments("manage_memory", {"query": "editor"})
        assert args_query["action"] == "search"


class TestToolCrafterOrchestratorIntegration:
    """Tests for ToolCrafter NativeOrchestrator alias mapping and argument normalization."""

    def test_native_orchestrator_alias_mapping(self):
        assert NativeOrchestrator.normalize_tool_name("create_tool") == "craft_tool"
        assert NativeOrchestrator.normalize_tool_name("write_tool") == "craft_tool"
        assert NativeOrchestrator.normalize_tool_name("hotload_tool") == "craft_tool"

    def test_native_orchestrator_argument_normalization(self):
        normalized = NativeOrchestrator.normalize_tool_arguments(
            "craft_tool",
            {"tool_name": "calc_tool", "source_code": "def execute(): pass"},
        )
        assert normalized.get("name") == "calc_tool"
        assert normalized.get("code") == "def execute(): pass"


class TestRoutineToolOrchestratorAndRegistryIntegration:
    """Tests for Routine tool integration with ToolRegistry and NativeOrchestrator."""

    def test_manage_routine_tool_discovered_in_registry(self):
        registry = ToolRegistry()
        tool = registry.get_tool("manage_routine")
        assert tool is not None
        assert tool.name == "manage_routine"
        assert tool.REQUIRED_RING == 1
        assert tool.tier == 1

        schemas = registry.get_tool_schemas()
        names = {s.get("function", {}).get("name") for s in schemas}
        assert "manage_routine" in names

    @pytest.mark.asyncio
    async def test_routine_lifecycle_create_list_run_toggle_delete(self, tmp_path: Path):
        from zelyr.tools.manage_routine import execute as routine_execute
        db_file = str(tmp_path / "routines_lifecycle.db")

        # 1. CREATE routine with interval
        create_res = await routine_execute(
            action="create",
            name="heartbeat_monitor",
            schedule="every 60s",
            routine_type="command",
            action_payload="echo 'heartbeat_alive'",
            description="System heartbeat monitor",
            db_path=db_file,
        )
        create_data = json.loads(create_res)
        assert create_data["status"] == "success"
        assert create_data["action"] == "create"
        assert create_data["name"] == "heartbeat_monitor"
        assert create_data["schedule"] == "60"
        assert create_data["schedule_type"] == "interval"
        assert create_data["enabled"] is True

        # 2. LIST routines
        list_res = await routine_execute(action="list", db_path=db_file)
        list_data = json.loads(list_res)
        assert list_data["status"] == "success"
        assert list_data["count"] == 1
        r_item = list_data["routines"][0]
        assert r_item["name"] == "heartbeat_monitor"
        assert r_item["schedule"] == "60"
        assert r_item["action_type"] == "command"
        assert r_item["enabled"] is True

        # 3. RUN routine (single-shot execution)
        run_res = await routine_execute(action="run", name="heartbeat_monitor", db_path=db_file)
        run_data = json.loads(run_res)
        assert run_data["status"] == "success"
        assert run_data["action"] == "run"
        assert run_data["name"] == "heartbeat_monitor"
        assert run_data["execution_status"] == "success"
        assert "heartbeat_alive" in run_data["output"]

        # 4. HISTORY inspection
        hist_res = await routine_execute(action="history", name="heartbeat_monitor", limit=5, db_path=db_file)
        hist_data = json.loads(hist_res)
        assert hist_data["status"] == "success"
        assert hist_data["count"] >= 1
        h_item = hist_data["history"][0]
        assert h_item["routine_name"] == "heartbeat_monitor"
        assert h_item["status"] == "success"
        assert "heartbeat_alive" in h_item["output"]

        # 5. TOGGLE routine (disable then enable)
        toggle_res1 = await routine_execute(action="toggle", name="heartbeat_monitor", enabled=False, db_path=db_file)
        toggle_data1 = json.loads(toggle_res1)
        assert toggle_data1["status"] == "success"
        assert toggle_data1["enabled"] is False

        toggle_res2 = await routine_execute(action="toggle", name="heartbeat_monitor", enabled=True, db_path=db_file)
        toggle_data2 = json.loads(toggle_res2)
        assert toggle_data2["status"] == "success"
        assert toggle_data2["enabled"] is True

        # 6. DELETE routine
        del_res = await routine_execute(action="delete", name="heartbeat_monitor", db_path=db_file)
        del_data = json.loads(del_res)
        assert del_data["status"] == "success"
        assert del_data["name"] == "heartbeat_monitor"

        # Confirm list is now empty
        post_del_list = json.loads(await routine_execute(action="list", db_path=db_file))
        assert post_del_list["count"] == 0

    @pytest.mark.asyncio
    async def test_tool_registry_execute_invocation(self, tmp_path: Path):
        from zelyr.tools.manage_routine import ManageRoutineTool
        db_file = str(tmp_path / "routines_tool.db")
        tool = ManageRoutineTool(db_path=db_file)

        create_result = await tool.execute_async(
            action="create",
            name="periodic_cleanup",
            schedule="300",
            action_payload="echo 'cleanup_done'",
        )
        assert create_result.success is True
        parsed_create = json.loads(create_result.output)
        assert parsed_create["status"] == "success"

        list_result = await tool.execute_async(action="list")
        assert list_result.success is True
        parsed_list = json.loads(list_result.output)
        assert parsed_list["count"] == 1

        run_result = await tool.execute_async(action="run", name="periodic_cleanup")
        assert run_result.success is True
        parsed_run = json.loads(run_result.output)
        assert parsed_run["execution_status"] == "success"
        assert "cleanup_done" in parsed_run["output"]

    def test_native_orchestrator_alias_and_argument_normalization(self):
        for alias in ("schedule_task", "create_routine", "schedule_routine", "cron", "cron_job"):
            assert NativeOrchestrator.normalize_tool_name(alias) == "manage_routine"

        normalized1 = NativeOrchestrator.normalize_tool_arguments(
            "manage_routine",
            {"name": "job1", "cron_expression": "*/5 * * * *", "command": "uptime"},
        )
        assert normalized1["schedule"] == "*/5 * * * *"
        assert normalized1["action_payload"] == "uptime"
        assert normalized1["action"] == "create"

        normalized2 = NativeOrchestrator.normalize_tool_arguments(
            "manage_routine",
            {"name": "job2", "interval": "60", "instruction": "check disk"},
        )
        assert normalized2["schedule"] == "60"
        assert normalized2["action_payload"] == "check disk"
        assert normalized2["action"] == "create"

@pytest.mark.asyncio
async def test_native_orchestrator_proactive_hydration_depth0(isolated_db):
    """Verify Orchestrator injects recalled memories on depth 0 and skips on retry or recovery."""
    store = MemoryStore(db_path=isolated_db)
    store.store(content="zsh with starship prompt", category="preference", key="user.shell")

    orch = NativeOrchestrator(db_path=isolated_db)

    async def mock_stream_provider(payload: dict) -> AsyncGenerator[dict, None]:
        yield {"choices": [{"delta": {"content": "Understood."}}]}

    # Case 1: depth == 0 turn -> Memory block MUST be injected
    payload_depth0 = {
        "messages": [{"role": "user", "content": "What is my user.shell?"}],
        "tools": [],
    }
    async for _ in orch.generate_stream(payload_depth0, depth=0, is_retry=False, stream_provider=mock_stream_provider):
        break

    sys_msg = payload_depth0["messages"][0]["content"]
    assert "[Contextual Memory]:" in sys_msg
    assert "<recalled_memory>" in sys_msg
    assert "user.shell: zsh with starship prompt" in sys_msg

    # Case 2: is_retry == True -> Must NOT inject
    payload_retry = {
        "messages": [{"role": "user", "content": "What is my user.shell?"}],
        "tools": [],
    }
    async for _ in orch.generate_stream(payload_retry, depth=0, is_retry=True, stream_provider=mock_stream_provider):
        break
    sys_retry = payload_retry["messages"][0]["content"]
    assert "[Contextual Memory]:" not in sys_retry

    # Case 3: Recovery turn (following a failed tool) -> Must NOT inject
    payload_recovery = {
        "messages": [
            {"role": "user", "content": "Run tests"},
            {"role": "assistant", "content": "Running command", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "execute_command"}}]},
            {"role": "tool", "content": "[Tool Error]: Command failed. Recovery Hint: check permissions"}
        ],
        "tools": [],
    }
    async for _ in orch.generate_stream(payload_recovery, depth=0, is_retry=False, stream_provider=mock_stream_provider):
        break
    sys_recovery = payload_recovery["messages"][0]["content"]
    assert "[Contextual Memory]:" not in sys_recovery


class TestMidTurnAcceptanceCriteriaFailureFeedback:
    """Phase 1 regression tests: mid-turn acceptance-criteria failure feedback."""

    @pytest.mark.asyncio
    async def test_1_successful_write_with_incorrect_content_produces_failed_verification_observation(self, tmp_path: Path):
        """1. A successful file write with incorrect content produces a failed-verification observation containing the actual mismatch."""
        orch = NativeOrchestrator()
        target_b = tmp_path / "beta.txt"

        async def real_write_executor(name: str, **kwargs):
            p = Path(kwargs["path"])
            p.write_text(kwargs["content"], encoding="utf-8")
            return f"Successfully wrote {len(kwargs['content'])} characters to {p}"

        turn = 0
        observed_tool_messages = []

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            for m in payload.get("messages", []):
                if m.get("role") == "tool" and m not in observed_tool_messages:
                    observed_tool_messages.append(dict(m))
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_wrong_beta",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_b),
                                        "content": "BETA_102",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Stopping here."}}]}

        from zelyr.engine.execution_ledger import TurnExecutionLedger
        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": f"Create {target_b} containing exactly BETA_202."}],
            "acceptance_criteria": f"file_contains:{target_b}:BETA_202",
        }
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=real_write_executor,
            ledger=ledger,
            max_replanning_attempts=0,
        ):
            pass

        assert len(observed_tool_messages) == 1
        obs_content = observed_tool_messages[0]["content"]
        # Tool itself executed successfully (not claimed as tool execution error)
        assert "- Status: SUCCESS" in obs_content
        assert observed_tool_messages[0]["status"] == "success"
        assert "Successfully wrote 8 characters" in obs_content
        # Outcome is explicitly reported as NOT verified with actual failed criterion and reason
        assert "- Verified: FALSE" in obs_content
        assert "NOT verified" in obs_content or "[Outcome Verification Failed]:" in obs_content
        assert str(target_b) in obs_content
        assert "file_contains" in obs_content
        assert "BETA_202" in obs_content
        assert f"Pattern 'BETA_202' not found in {target_b}" in obs_content

    @pytest.mark.asyncio
    async def test_2_successful_write_with_correct_content_reported_as_verified(self, tmp_path: Path):
        """2. A successful write with correct content is reported as verified."""
        orch = NativeOrchestrator()
        target_a = tmp_path / "alpha.txt"

        async def real_write_executor(name: str, **kwargs):
            p = Path(kwargs["path"])
            p.write_text(kwargs["content"], encoding="utf-8")
            return f"Successfully wrote {len(kwargs['content'])} characters to {p}"

        turn = 0
        observed_tool_messages = []

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            for m in payload.get("messages", []):
                if m.get("role") == "tool" and m not in observed_tool_messages:
                    observed_tool_messages.append(dict(m))
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_ok_alpha",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_a),
                                        "content": "ALPHA_101",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Completed."}}]}

        from zelyr.engine.execution_ledger import TurnExecutionLedger
        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": f"Create {target_a} containing exactly ALPHA_101."}],
            "acceptance_criteria": f"file_contains:{target_a}:ALPHA_101",
        }
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=real_write_executor,
            ledger=ledger,
        ):
            pass

        assert len(observed_tool_messages) == 1
        obs_content = observed_tool_messages[0]["content"]
        assert "- Status: SUCCESS" in obs_content
        assert "- Verified: TRUE" in obs_content
        assert "[Machine State Verified]:" in obs_content
        assert ledger.is_target_verified(str(target_a)) is True
        assert ledger._records["call_ok_alpha"].verified is True

    @pytest.mark.asyncio
    async def test_3_tool_execution_error_distinguished_from_acceptance_criteria_mismatch(self, tmp_path: Path):
        """3. A tool execution error is still distinguished from an acceptance-criteria mismatch."""
        orch = NativeOrchestrator()
        target_err = tmp_path / "readonly.txt"
        target_mismatch = tmp_path / "mismatch.txt"

        async def executor(name: str, **kwargs):
            p = Path(kwargs["path"])
            if p == target_err:
                raise PermissionError(f"Permission denied: '{p}'")
            p.write_text(kwargs["content"], encoding="utf-8")
            return f"Successfully wrote {len(kwargs['content'])} characters to {p}"

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_exec_err",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_err),
                                        "content": "EXPECTED",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif turn == 2:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_crit_mismatch",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_mismatch),
                                        "content": "WRONG_VALUE",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Unable to complete due to error."}}]}

        from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger
        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": f"Write {target_err} and {target_mismatch}."}],
            "acceptance_criteria": f"file_contains:{target_err}:EXPECTED,file_contains:{target_mismatch}:EXPECTED",
        }
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=executor,
            ledger=ledger,
            max_replanning_attempts=0,
        ):
            pass

        tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

        # First tool call: actual tool execution error
        assert tool_msgs[0]["status"] == "error"
        assert "- Status: ERROR" in tool_msgs[0]["content"]
        assert "[Error Diagnostic:" in tool_msgs[0]["content"]
        assert ledger._records["call_exec_err"].state == ActionState.FAILED

        # Second tool call: tool execution succeeded, acceptance criterion mismatched
        assert tool_msgs[1]["status"] == "success"
        assert "- Status: SUCCESS" in tool_msgs[1]["content"]
        assert "- Verified: FALSE" in tool_msgs[1]["content"]
        assert "[Outcome Verification Failed]:" in tool_msgs[1]["content"]
        assert ledger._records["call_crit_mismatch"].state == ActionState.SUCCESS
        assert ledger._records["call_crit_mismatch"].verified is False

    @pytest.mark.asyncio
    async def test_4_failed_criterion_never_marks_target_verified_and_preserves_ledger_failure_detail(self, tmp_path: Path):
        """4. A failed criterion never marks its target verified and preserves the verification failure in the ledger."""
        orch = NativeOrchestrator()
        target_b = tmp_path / "beta.txt"

        async def executor(name: str, **kwargs):
            p = Path(kwargs["path"])
            p.write_text(kwargs["content"], encoding="utf-8")
            return f"Wrote to {p}"

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_b1",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_b),
                                        "content": "BETA_102",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Stopped."}}]}

        from zelyr.engine.execution_ledger import TurnExecutionLedger
        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": f"Create {target_b} containing BETA_202."}],
            "acceptance_criteria": f"file_contains:{target_b}:BETA_202",
        }
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=executor,
            ledger=ledger,
            max_replanning_attempts=0,
        ):
            pass

        assert ledger.is_target_verified(str(target_b)) is False
        assert ledger.get_verified_targets() == []
        rec = ledger._records["call_b1"]
        assert rec.verified is False
        assert rec in ledger.unverified_actions()
        assert rec.verification_detail is not None
        assert "Verification failed" in rec.verification_detail
        assert "BETA_202" in rec.verification_detail

    @pytest.mark.asyncio
    async def test_5_verification_failure_does_not_trigger_automatic_repetition_and_blocks_identical_retry(self, tmp_path: Path):
        """5. The model cannot trigger automatic repetition merely because verification failed."""
        orch = NativeOrchestrator()
        target_b = tmp_path / "beta.txt"
        executed_calls = []

        async def executor(name: str, **kwargs):
            executed_calls.append(dict(kwargs))
            p = Path(kwargs["path"])
            p.write_text(kwargs["content"], encoding="utf-8")
            return f"Wrote to {p}"

        turn = 0

        async def mock_stream(payload: dict):
            nonlocal turn
            turn += 1
            if turn in (1, 2):
                # Model tries to emit the exact same wrong content twice
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": f"call_dup_{turn}",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_b),
                                        "content": "BETA_102",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Stopped."}}]}

        from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger
        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": f"Create {target_b} containing BETA_202."}],
            "acceptance_criteria": f"file_contains:{target_b}:BETA_202",
        }
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=executor,
            ledger=ledger,
            max_replanning_attempts=0,
        ):
            pass

        # Executor was called only ONCE; no automatic retry occurred, and identical parameters in Turn 2 were blocked
        assert len(executed_calls) == 1
        assert ledger._records["call_dup_2"].state == ActionState.FAILED
        assert "redundant_action" in str(ledger._records["call_dup_2"].error)

    @pytest.mark.asyncio
    async def test_6_subsequent_corrected_tool_call_succeeds_and_verifies(self, tmp_path: Path):
        """6. A subsequent corrected tool call can succeed through the normal validation and execution path."""
        orch = NativeOrchestrator()
        target_a = tmp_path / "alpha.txt"
        target_b = tmp_path / "beta.txt"
        executed_writes = []

        async def executor(name: str, **kwargs):
            executed_writes.append((kwargs["path"], kwargs["content"]))
            p = Path(kwargs["path"])
            p.write_text(kwargs["content"], encoding="utf-8")
            return f"Successfully wrote {len(kwargs['content'])} characters to {p}"

        turn = 0
        observed_beta_failure_msg = ""

        async def mock_stream(payload: dict):
            nonlocal turn, observed_beta_failure_msg
            turn += 1
            if turn == 1:
                # Proposes two writes (S2 reproduction)
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({
                                            "action": "write",
                                            "path": str(target_a),
                                            "content": "ALPHA_101",
                                        }),
                                    },
                                },
                                {
                                    "index": 1,
                                    "id": "call_b_pending",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({
                                            "action": "write",
                                            "path": str(target_b),
                                            "content": "BETA_202",
                                        }),
                                    },
                                },
                            ]
                        }
                    }]
                }
            elif turn == 2:
                # Model emits second write with wrong content BETA_102
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_b_wrong",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_b),
                                        "content": "BETA_102",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            elif turn == 3:
                # Model sees the failed-verification observation and emits corrected tool call with BETA_202
                tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
                observed_beta_failure_msg = tool_msgs[-1]["content"]
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_b_fixed",
                                "type": "function",
                                "function": {
                                    "name": "manage_workspace_file",
                                    "arguments": json.dumps({
                                        "action": "write",
                                        "path": str(target_b),
                                        "content": "BETA_202",
                                    }),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {"choices": [{"delta": {"content": "Both files created and verified."}}]}

        from zelyr.engine.execution_ledger import ActionState, TurnExecutionLedger
        ledger = TurnExecutionLedger()
        payload = {
            "messages": [{"role": "user", "content": f"Create {target_a} with ALPHA_101 and {target_b} with BETA_202."}],
            "acceptance_criteria": f"file_contains:{target_a}:ALPHA_101,file_contains:{target_b}:BETA_202",
        }
        async for _ in orch.generate_stream(
            payload,
            stream_provider=mock_stream,
            tool_executor=executor,
            ledger=ledger,
        ):
            pass

        assert "- Verified: FALSE" in observed_beta_failure_msg
        assert "BETA_202" in observed_beta_failure_msg
        assert target_a.read_text(encoding="utf-8") == "ALPHA_101"
        assert target_b.read_text(encoding="utf-8") == "BETA_202"
        assert ledger.is_target_verified(str(target_a)) is True
        assert ledger.is_target_verified(str(target_b)) is True
        assert ledger._records["call_b_fixed"].state == ActionState.VERIFIED
        assert ledger._records["call_b_fixed"].verified is True
