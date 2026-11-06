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
        data = json.loads(formatted)
        assert data["status"] == "error"
        assert data["error_type"] == "COMMAND_NOT_FOUND"
        assert data["exit_code"] == 127
        assert "Command not found in PATH" in data["remedy_hint"]
        assert "execute_command" in data["allowed_actions"]
        assert "Recovery Hint:" in data["error"]


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
