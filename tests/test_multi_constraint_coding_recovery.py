"""Regression tests for multi-constraint coding recovery, failure framing, and context budgeting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.budget_manager import ContextBudgetEngine
from zelyr.engine.execution_ledger import TurnExecutionLedger
from zelyr.engine.observation_framing import ObservationFramingEngine
from zelyr.engine.verifier import OutcomeVerifier
from zelyr.tools.workspace_file import ManageWorkspaceFileTool


@pytest.mark.asyncio
async def test_patch_missing_path_fails_as_missing_parameter_not_dir_not_found(tmp_path: Path, monkeypatch) -> None:
    """Omitting 'path' on patch must report missing parameter 'path', not PATH_NOT_FOUND on cwd."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "versioning.py").write_text("x = 1\n", encoding="utf-8")

    tool = ManageWorkspaceFileTool()
    res = await tool.execute(action="patch", search_text="x = 1", replace_text="x = 2")
    assert res.success is False
    assert "Missing required parameter 'path'" in (res.error or "")

    # Also verify compact schema preserves clear path guidance within 60-char condensed property description
    from zelyr.engine.schema_compact import CompactSchemaGenerator

    wf_openai_schema = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        },
    }
    compacted = CompactSchemaGenerator.compact_tool_schema(wf_openai_schema)
    path_desc = compacted["function"]["parameters"]["properties"]["path"]["description"]
    assert "required" in path_desc.lower() or "patch" in path_desc.lower()


def test_command_failure_snippet_preserves_top_failing_tests_and_assertions(tmp_path: Path, monkeypatch) -> None:
    """verify_command_output must preserve both early and late failing tests even when verbose assertions exceed 360 chars."""
    monkeypatch.chdir(tmp_path)
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "import pytest\n"
        "def test_alpha_very_long_regression_name_for_first_failure_case():\n"
        "    assert (1, 2, 3, '', '') == (1, 2, 3)\n"
        "def test_beta_very_long_regression_name_for_second_failure_case():\n"
        "    assert (2, 4, 0, '', '') == (2, 4, 0)\n"
        "def test_gamma_very_long_regression_name_for_third_failure_case():\n"
        "    assert 'expected-prerelease-and-build-metadata-tuple' == 'broken-prerelease-value'\n"
        "def test_delta_very_long_regression_name_for_fourth_failure_case():\n"
        "    with pytest.raises(ValueError):\n"
        "        int('42')\n"
        "def test_epsilon_very_long_regression_name_for_fifth_failure_case():\n"
        "    assert False, 'fifth failing check at bottom of summary output'\n",
        encoding="utf-8",
    )

    vr = OutcomeVerifier.verify(
        check="command_output",
        target=f"{Path(__file__).resolve().parent.parent / '.venv' / 'bin' / 'pytest'} -v {test_file.name}",
        expected="5 passed",
    )
    assert vr.success is False
    assert vr.error is not None
    assert "Expected pattern '5 passed' not found in command output" in vr.error
    assert "test_alpha_very_long_regression_name_for_first_failure_case" in vr.error
    assert "test_delta_very_long_regression_name_for_fourth_failure_case" in vr.error
    assert "test_epsilon_very_long_regression_name_for_fifth_failure_case" in vr.error


def test_observation_multiline_error_and_verifier_reason_preserve_all_failures() -> None:
    """_summarize_multiline_error and frame_observation retain distinct test failures without losing early ones."""
    pytest_err = "\n".join([
        "============================= test session starts ==============================",
        "platform linux -- Python 3.11.17",
        "collected 4 items",
        "tests/test_v.py::test_standard FAILED",
        "E   AssertionError: assert (1, 2, 3, '', '') == (1, 2, 3)",
        "tests/test_v.py::test_two_part FAILED",
        "E   AssertionError: assert (2, 4, 0, '', '') == (2, 4, 0)",
        "FAILED tests/test_v.py::test_prerelease_and_build - ValueError: invalid literal for int() with base 10: '0+exp'",
        "FAILED tests/test_v.py::test_alpha_build_meta - AssertionError: metadata lost in parse_version output tuple",
        "FAILED tests/test_v.py::test_trailing_dot_case - Failed: DID NOT RAISE <class 'ValueError'>",
        "FAILED tests/test_v.py::test_invalid - Failed: DID NOT RAISE <class 'ValueError'>",
    ])
    summary = ObservationFramingEngine._summarize_multiline_error(pytest_err)
    assert "test_standard" in summary
    assert "test_two_part" in summary
    assert "test_prerelease_and_build" in summary
    assert "test_invalid" in summary

    long_verifier_reason = (
        "Expected pattern '4 passed' not found in command output (exit 1) | output: "
        "FAILED tests/test_versioning.py::test_prerelease_and_build - AssertionError: assert (1, 2, 0, 'rc1', 'build.4') == (1, 2, 0, 'rc1', '') | "
        + ("context_padding_token " * 18)
        + " | FAILED tests/test_versioning.py::test_invalid - Failed: DID NOT RAISE <class 'ValueError'>"
    )
    framed = ObservationFramingEngine.frame_observation(
        "manage_workspace_file",
        json.dumps({
            "path": "versioning.py",
            "status": "success",
            "verification_failed": True,
            "check": "command_output",
            "target": "pytest -q",
            "failed_criterion": "command_output on pytest -q (expected '4 passed')",
            "verifier_reason": long_verifier_reason,
        }),
    )
    assert "test_prerelease_and_build" in framed
    assert "test_invalid" in framed


def test_stage2_consolidates_summaries_without_stacking_and_strips_echoed_writes() -> None:
    """Repeated Stage 2 pruning must consolidate to a single summary message and strip redundant write echoes."""
    engine = ContextBudgetEngine()
    sys_msg = {"role": "system", "content": "Root directives."}
    prior_sum = {
        "role": "system",
        "content": "[Previous conversation summary: Tools executed: execute_command | Preserved failures/verification errors: old]",
    }
    user_root = {"role": "user", "content": "Fix versioning.py and pass all tests."}

    long_code = "\n".join(f"line_{i} = {i}" for i in range(40))
    write_obs = (
        "[TOOL OBSERVATION: manage_workspace_file]\n"
        "- Status: SUCCESS\n"
        "- Exit Code: N/A\n"
        "- Verified: FALSE\n"
        "- Summary: Tool 'manage_workspace_file' executed successfully, but outcome is NOT verified.\n\n"
        "[Output]:\n"
        + json.dumps({
            "path": "/tmp/ws/versioning.py",
            "bytes_written": len(long_code),
            "backup": "/tmp/zelyr_backups/versioning.py.123.bak",
            "message": "Successfully wrote versioning.py.",
            "line_count": 40,
            "written_content": long_code * 2,
        })
        + "\n\n[Outcome Verification Failed]:\n"
        "- Target: pytest -q\n"
        "- Failed Criterion: command_output on pytest -q (expected '4 passed')\n"
        "- Verifier Reason: FAILED tests/test_versioning.py::test_invalid - DID NOT RAISE ValueError"
    )
    compacted = engine.compact_tool_output(write_obs, token_limit=200)
    assert "written_content" not in compacted or "omitted" in compacted
    assert "[Outcome Verification Failed]:" in compacted
    assert "test_invalid" in compacted

    msgs: List[Dict[str, Any]] = [
        sys_msg,
        prior_sum,
        user_root,
        {"role": "assistant", "content": "Try 1", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"action": "write", "path": "versioning.py"}}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "manage_workspace_file", "content": write_obs},
        {"role": "assistant", "content": "Try 2", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "execute_command", "arguments": {"command": "pytest -v"}}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "execute_command", "content": "FAILED test_invalid " * 120},
        {"role": "assistant", "content": "Try 3", "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"action": "read", "path": "tests/test_versioning.py"}}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "manage_workspace_file", "content": "def test_invalid(): pass"},
    ]
    pruned, evicted = engine.stage2_prune_turns(msgs, target_budget=350)
    assert evicted > 0
    summary_msgs = [m for m in pruned if "[Previous conversation summary:" in str(m.get("content", ""))]
    assert len(summary_msgs) == 1
    assert pruned[1] == summary_msgs[0]


@pytest.mark.asyncio
async def test_identical_failed_verification_edit_is_blocked_and_cycle_test_steers_edit(tmp_path: Path) -> None:
    """Reverting back to an identical edit that already failed --verify must be blocked; pytest retry cycle gives clear edit guidance."""
    target = tmp_path / "versioning.py"
    target.write_text("def parse_version(raw): return (0, 0, 0)\n", encoding="utf-8")
    out_flag = tmp_path / "flag.txt"
    out_flag.write_text("INITIAL", encoding="utf-8")

    # 1. Edit A fails verification -> Edit B fails verification -> Re-submitting Edit A must be blocked
    orch = NativeOrchestrator()
    orch.max_steps = 6
    turn = 0
    seen_tools: List[str] = []

    async def mock_stream(payload: Dict[str, Any]):
        nonlocal turn
        turn += 1
        seen_tools.extend(m.get("content", "") for m in payload["messages"] if m.get("role") == "tool")
        if turn == 1:
            tc = {"index": 0, "id": "c_a1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": str(target), "content": "def parse_version(raw): return (1, 0, 0)\n"})}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "tool_calls"}]}
        elif turn == 2:
            tc = {"index": 0, "id": "c_b1", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": str(target), "content": "def parse_version(raw): return (2, 0, 0)\n"})}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "tool_calls"}]}
        elif turn == 3:
            # Repeat Edit A (identical args & content that previously failed verification!)
            tc = {"index": 0, "id": "c_a2", "type": "function", "function": {"name": "manage_workspace_file", "arguments": json.dumps({"action": "write", "path": str(target), "content": "def parse_version(raw): return (1, 0, 0)\n"})}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Stopped."}, "finish_reason": "stop"}]}

    exec_calls: List[str] = []

    async def mock_exec(name: str, **kwargs):
        exec_calls.append(kwargs.get("content", ""))
        Path(kwargs["path"]).write_text(kwargs["content"], encoding="utf-8")
        return json.dumps({"status": "success", "path": kwargs["path"]})

    ledger = TurnExecutionLedger()
    payload = {
        "messages": [{"role": "user", "content": f"Update {target} cleanly."}],
        "acceptance_criteria": f"file_contains:{out_flag}:DONE",
    }
    async for _ in orch.generate_stream(
        payload,
        stream_provider=mock_stream,
        tool_executor=mock_exec,
        ledger=ledger,
        max_replanning_attempts=0,
    ):
        pass

    assert len(exec_calls) == 2
    tool_msgs = [m for m in payload["messages"] if m.get("role") == "tool"]
    assert json.loads(tool_msgs[-1]["content"])["reason"] == "redundant_action"

    # 2. Failed pytest retry cycle guidance tells the model to inspect tests / edit code instead of re-running pytest
    orch2 = NativeOrchestrator()
    orch2.max_steps = 4
    t2 = 0

    async def mock_stream_cycle(payload: Dict[str, Any]):
        nonlocal t2
        t2 += 1
        if t2 in (1, 2):
            tc = {"index": 0, "id": f"pt_{t2}", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "pytest -v"})}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Will inspect code."}, "finish_reason": "stop"}]}

    async def mock_fail_exec(name: str, **kwargs):
        return json.dumps({"status": "error", "exit_code": 1, "output": "FAILED tests/test_versioning.py::test_invalid"})

    p2 = {"messages": [{"role": "user", "content": "Run pytest -v and fix versioning.py"}]}
    async for _ in orch2.generate_stream(
        p2,
        stream_provider=mock_stream_cycle,
        tool_executor=mock_fail_exec,
        max_replanning_attempts=0,
    ):
        pass

    rej_msg = json.loads([m for m in p2["messages"] if m.get("role") == "tool"][-1]["content"])
    assert rej_msg["reason"] == "cycle_detected"
    assert "manage_workspace_file" in rej_msg["detail"]


@pytest.mark.asyncio
async def test_native_orchestrator_prunes_using_final_scoped_tools_and_safe_headroom(tmp_path: Path) -> None:
    """Prune step must account for final scoped tools and keep comfortable headroom under 8192 num_ctx."""
    captured_payloads: List[Dict[str, Any]] = []

    async def mock_stream(payload: Dict[str, Any]):
        captured_payloads.append({"messages": list(payload.get("messages", [])), "tools": list(payload.get("tools", []))})
        if len(captured_payloads) == 1:
            tc = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "execute_command", "arguments": json.dumps({"command": "pytest -v"})}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content": "Analyzed."}, "finish_reason": "stop"}]}

    async def mock_exec(name: str, **kwargs):
        return json.dumps({"status": "error", "exit_code": 1, "output": "FAILED " + ("detail " * 900)})

    orch = NativeOrchestrator(stream_provider=mock_stream, tool_executor=mock_exec)
    orch.max_context_tokens = 8192
    # Add compact turns (<250 tokens each so Stage 1 cannot trim them) totaling ~6300 est tokens
    # while staying below checkpoint proposal truncation threshold.
    chunk = " ".join(f"item{i}" for i in range(180))
    hist: List[Dict[str, Any]] = [{"role": "user", "content": "Run pytest -v and inspect failure"}]
    for idx in range(1, 21):
        cid = f"old_{idx}"
        hist.append({
            "role": "assistant",
            "content": "Checking step " + str(idx),
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"action": "read", "path": f"file_{idx}.py"}}}],
        })
        hist.append({
            "role": "tool",
            "tool_call_id": cid,
            "name": "manage_workspace_file",
            "content": f"[TOOL OBSERVATION: manage_workspace_file]\n- Status: SUCCESS\n\n[Output]:\n{chunk}",
        })

    async for _ in orch.generate_stream({"messages": hist, "model": "qwen3.5:4b"}):
        pass

    from zelyr.engine.budget_manager import estimate_messages_tokens, estimate_tools_tokens

    assert captured_payloads
    last = captured_payloads[-1]
    est_total = estimate_messages_tokens(last["messages"]) + estimate_tools_tokens(last["tools"])
    # Under 8192 num_ctx, estimated messages + tools must respect >=25% reserve (<= 6144) for Qwen3.5 template & reasoning
    assert est_total <= 6144
