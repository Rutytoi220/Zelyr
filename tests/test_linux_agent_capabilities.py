"""Regression tests for Linux agent capabilities, CLI visibility, and Chromium Native Messaging."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zelyr.agents.actuator_arbiter import ActuatorArbiter
from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.cli.main import (
    _build_parser,
    _execute_headless_run,
    _redact_cli_args,
    _summarize_cli_tool_result,
)
from zelyr.engine.execution_ledger import (
    TurnExecutionLedger,
    extract_global_mutation_prohibition,
    has_prospective_action_prose,
    is_diagnostic_test_command,
    is_mutating_tool_call,
    is_truthful_failure_report,
)
from zelyr.engine.execution_policy import (
    ExecutionPolicyConfig,
    ExecutionPolicyEngine,
    PolicyDecisionOutcome,
)
from zelyr.tools.browser_cdp import InteractWithBrowserTool
from zelyr.tools.browser_native_host import (
    BrowserNativeHost,
    MAX_NATIVE_MESSAGE_BYTES,
    NativeBrowserClient,
    compute_unpacked_extension_id,
    install_native_messaging_host,
    read_native_message,
    validate_browser_command,
    validate_extension_origin,
    write_native_message,
)
from zelyr.tools.workspace_file import ManageWorkspaceFileTool


def test_arbiter_coding_repo_vs_browser_routing() -> None:
    assert ActuatorArbiter.arbitrate(task="Inspect this repository and evaluate why pytest fails", bridge_connected=True) != "tier2_browser"
    assert ActuatorArbiter.arbitrate(task="Fix the bug in pkg/parser.py and run tests", bridge_connected=True) == "tier1_ipc"
    assert ActuatorArbiter.arbitrate(task="Inspect the browser tab and evaluate_script on the active web page", bridge_connected=True) == "tier2_browser"


@pytest.mark.asyncio
async def test_workspace_file_mkdir_dir_read_unescape_syntax_and_action_inference(tmp_path: Path) -> None:
    tool = ManageWorkspaceFileTool()

    res_mkdir = await tool.execute({"action": "mkdir", "path": "capability_test/sub", "cwd": str(tmp_path)})
    assert res_mkdir.success
    assert (tmp_path / "capability_test" / "sub").is_dir()

    res_dir_read = await tool.execute({"action": "read", "path": ".", "cwd": str(tmp_path)})
    assert res_dir_read.success
    assert "entries" in res_dir_read.output
    assert "directory" in res_dir_read.output.get("note", "").lower()

    # Bare '/' path outside '/' fails cleanly instead of walking root filesystem
    res_root_guard = await tool.execute({"action": "list_tree", "path": "/"})
    assert not res_root_guard.success
    assert "root filesystem" in (res_root_guard.error or "").lower()

    # Large directory trees are bounded cleanly instead of blowing context window
    big_dir = tmp_path / "big"
    big_dir.mkdir()
    for idx in range(150):
        (big_dir / f"f_{idx:03d}.txt").write_text("x", encoding="utf-8")
    res_bounded = await tool.execute({"action": "list_tree", "path": str(big_dir)})
    assert res_bounded.success
    assert res_bounded.output["truncated"] is True
    assert res_bounded.output["count"] <= 120

    # Double-escaped newlines from smaller local models get decoded cleanly
    py_path = tmp_path / "mod.py"
    res_write = await tool.execute({
        "action": "write",
        "path": str(py_path),
        "content": "def add(a, b):\\n    return a + b\\n",
    })
    assert res_write.success
    assert py_path.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"

    # Syntax errors on patch/write abort before modifying the file on disk
    bad_patch = await tool.execute({
        "action": "patch",
        "path": str(py_path),
        "search_text": "    return a + b",
        "replace_text": "return a + b",
    })
    assert not bad_patch.success
    assert "SyntaxError" in (bad_patch.error or "") or "IndentationError" in (bad_patch.error or "")
    assert py_path.read_text(encoding="utf-8") == "def add(a, b):\n    return a + b\n"

    # Omitted action infers patch when search_text/replace_text present
    inferred_patch = await tool.execute({
        "path": str(py_path),
        "search_text": "    return a + b",
        "replace_text": "    return a + b + 0",
    })
    assert inferred_patch.success
    assert "a + b + 0" in py_path.read_text(encoding="utf-8")

    norm_args = NativeOrchestrator.normalize_tool_arguments(
        "manage_workspace_file",
        {"path": "mod.py", "search_text": "x", "replace_text": "y"},
    )
    assert norm_args["action"] == "patch"


def test_ledger_global_prohibitions_and_diagnostic_helpers() -> None:
    p_ro = extract_global_mutation_prohibition("Inspect app.py, do NOT modify any files or run any mutating commands.")
    assert p_ro["no_file_modifications"] is True
    assert p_ro["no_mutating_commands"] is True

    assert is_diagnostic_test_command("execute_command", {"command": "pytest -v"}) is True
    assert is_mutating_tool_call("execute_command", {"command": "pytest -v"}, allow_read_only_tests=True)[0] is False
    assert is_mutating_tool_call("execute_command", {"command": "rm -rf tmp"}, allow_read_only_tests=True)[0] is True
    assert is_mutating_tool_call("manage_workspace_file", {"action": "write", "path": "a.py"})[0] is True

    assert has_prospective_action_prose("Let's fix this bug now:\n```python\nx = 1\n```") is True
    assert is_truthful_failure_report("Let's fix this bug cleanly in app.py") is False
    assert is_truthful_failure_report("The test failed because permission was denied.") is True

    from zelyr.engine.execution_ledger import is_read_only_diagnostic_failures_only
    diag_ledger = TurnExecutionLedger(execution_id="exec_diag_only")
    diag_ledger.record_proposed("d1", "manage_workspace_file", {"action": "read", "path": "calc.py"})
    diag_ledger.record_validated("d1")
    diag_ledger.record_dispatched("d1")
    diag_ledger.record_observation("d1", status="success", output="def add(a, b): return a - b")
    diag_ledger.record_proposed("d2", "execute_command", {"command": "pytest -v"})
    diag_ledger.record_validated("d2")
    diag_ledger.record_dispatched("d2")
    diag_ledger.record_observation("d2", output="1 failed", status="error", exit_code=1, error="1 failed")
    assert is_read_only_diagnostic_failures_only(
        "Inspect this project and run pytest -v. Explain why tests fail without modifying any files.",
        diag_ledger,
        "1 test failed in test_calc.py because add() returns a - b instead of a + b.",
    ) is True

    ledger = TurnExecutionLedger(execution_id="exec_compat_1")
    ledger.record_proposed("c1", "manage_workspace_file", {"action": "patch", "path": "app.py"})
    ledger.record_validated("c1")
    ledger.record_dispatched("c1")
    ledger.record_observation("c1", output="", status="error", error="search_text appeared 0 times")

    ledger.record_proposed("c2", "manage_workspace_file", {"action": "write", "path": "app.py"})
    ledger.record_validated("c2")
    ledger.record_dispatched("c2")
    ledger.record_observation("c2", status="success", output="ok")
    assert ledger.unresolved_failures() == []

    # Superseded reconfirmed_and_dispatched cancellation is treated as resolved
    ledger.record_proposed("c3", "manage_workspace_file", {"action": "write", "path": "b.py"})
    ledger.cancel_pending_proposal("c3", reason="reconfirmed_and_dispatched")
    assert ledger.unresolved_failures() == []


@pytest.mark.asyncio
async def test_orchestrator_readonly_guard_cycle_reset_and_thinking_retry() -> None:
    # 1. Global prohibition blocks mutating write
    turn_idx = 0

    @contextlib.asynccontextmanager
    async def mock_ro_stream(method, url, **kwargs):
        nonlocal turn_idx
        turn_idx += 1
        resp = AsyncMock()
        resp.status_code = 200
        if turn_idx == 1:
            tc = {
                "id": "call_ro_1",
                "type": "function",
                "function": {
                    "name": "manage_workspace_file",
                    "arguments": {"action": "write", "path": "app.py", "content": "x = 1"},
                },
            }
            lines = [f'data: {json.dumps({"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]})}', "data: [DONE]"]
        else:
            lines = ['data: {"choices": [{"delta": {"content": "Blocked as requested; explained bug without modifying files."}, "finish_reason": "stop"}]}', "data: [DONE]"]

        async def aiter_lines():
            for ln in lines:
                yield ln

        resp.aiter_lines = aiter_lines
        yield resp

    orch = NativeOrchestrator()
    executed = []

    async def fake_exec(name, **kw):
        executed.append((name, kw))
        return {"status": "ok"}

    payload = {
        "messages": [{"role": "user", "content": "Explain app.py bug, do NOT modify any files or run any mutating commands."}],
        "tools": [{"type": "function", "function": {"name": "manage_workspace_file", "parameters": {"type": "object"}}}],
    }
    with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=fake_exec), \
         patch.object(orch, "get_active_window_context", return_value=None), \
         patch("httpx.AsyncClient") as mock_http:
        mock_http.return_value.__aenter__.return_value.stream = mock_ro_stream
        out_chunks = [c async for c in orch.generate_stream(payload)]
    assert executed == []
    assert any("Blocked as requested" in str(c) for c in out_chunks)

    # 2. Cycle detection allows re-running pytest after intervening file edit
    seq = 0

    async def edit_then_retest_stream(p):
        nonlocal seq
        seq += 1
        if seq == 1:
            tc = {"id": "t1", "index": 0, "type": "function", "function": {"name": "execute_command", "arguments": {"command": "pytest -v"}}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]}
        elif seq == 2:
            tc = {"id": "t2", "index": 0, "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"action": "write", "path": "mod.py", "content": "x = 1"}}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]}
        elif seq == 3:
            tc = {"id": "t3", "index": 0, "type": "function", "function": {"name": "execute_command", "arguments": {"command": "pytest -v"}}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]}
        else:
            yield {"choices": [{"delta": {"content": "All tests pass now."}, "finish_reason": "stop"}]}

    cmds = []

    async def mock_runner(name, args):
        if name == "execute_command":
            cmds.append(args["command"])
            if len(cmds) == 1:
                return json.dumps({"status": "error", "exit_code": 1, "output": "1 failed"})
            return json.dumps({"status": "ok", "exit_code": 0, "output": "1 passed"})
        return json.dumps({"status": "ok"})

    orch2 = NativeOrchestrator(stream_provider=edit_then_retest_stream, tool_executor=mock_runner)
    orch2.max_steps = 8
    tools_list = [
        {"type": "function", "function": {"name": "execute_command", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "manage_workspace_file", "parameters": {"type": "object"}}},
    ]
    res_chunks = [c async for c in orch2.generate_stream({"messages": [{"role": "user", "content": "Fix and re-run tests"}], "tools": tools_list})]
    assert cmds == ["pytest -v", "pytest -v"]
    assert any("All tests pass now." in str(c) for c in res_chunks)

    # 3. Thinking-only output with empty visible content retries once cleanly
    think_turn = 0

    async def think_only_stream(p):
        nonlocal think_turn
        think_turn += 1
        if think_turn == 1:
            yield {"choices": [{"delta": {"reasoning_content": "Internal reasoning only"}, "finish_reason": "stop"}]}
        else:
            yield {"choices": [{"delta": {"content": "Visible final explanation."}, "finish_reason": "stop"}]}

    orch3 = NativeOrchestrator(stream_provider=think_only_stream)
    think_chunks = [c async for c in orch3.generate_stream({"messages": [{"role": "user", "content": "Explain"}]})]
    joined = "".join(c.get("choices", [{}])[0].get("delta", {}).get("content", "") for c in think_chunks)
    assert think_turn == 2
    assert "Visible final explanation." in joined

    # 4. Mid-task thinking truncation after prior tool calls retries once concisely
    len_turn = 0

    async def mid_task_len_stream(p):
        nonlocal len_turn
        len_turn += 1
        if len_turn == 1:
            tc = {"id": "lt1", "index": 0, "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"action": "read", "path": "a.py"}}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]}
        elif len_turn == 2:
            yield {"choices": [{"delta": {"reasoning_content": "Long unfinished thought..."}, "finish_reason": "length"}]}
        else:
            yield {"choices": [{"delta": {"content": "Recovered concisely after length limit."}, "finish_reason": "stop"}]}

    async def read_mock_runner(name, args):
        return json.dumps({"status": "ok", "content": "x = 1"})

    orch4 = NativeOrchestrator(
        stream_provider=mid_task_len_stream,
        tool_executor=read_mock_runner,
    )
    orch4.max_steps = 5
    len_chunks = [c async for c in orch4.generate_stream({"messages": [{"role": "user", "content": "Read a.py"}], "tools": tools_list})]
    joined_len = "".join(c.get("choices", [{}])[0].get("delta", {}).get("content", "") for c in len_chunks)
    assert len_turn == 3
    assert "Recovered concisely after length limit." in joined_len

    # 5. Final turn at depth == max_depth - 1 after actions retries synthesis if only <think> was emitted
    late_turn = 0

    async def late_think_only_stream(p):
        nonlocal late_turn
        late_turn += 1
        if late_turn == 1:
            tc = {"id": "lt2", "index": 0, "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"action": "read", "path": "a.py"}}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]}
        elif late_turn == 2:
            yield {"choices": [{"delta": {"reasoning_content": "All done inside think only."}, "finish_reason": "stop"}]}
        else:
            assert not p.get("tools"), "Synthesis retry at depth ceiling must disable tools"
            yield {"choices": [{"delta": {"content": "Final grounded summary visible."}, "finish_reason": "stop"}]}

    orch5 = NativeOrchestrator(stream_provider=late_think_only_stream, tool_executor=read_mock_runner)
    orch5.max_steps = 2
    orch5.max_depth = 2
    late_chunks = [c async for c in orch5.generate_stream({"messages": [{"role": "user", "content": "Read a.py"}], "tools": tools_list})]
    late_joined = "".join(c.get("choices", [{}])[0].get("delta", {}).get("content", "") for c in late_chunks)
    assert late_turn == 3
    assert "Final grounded summary visible." in late_joined


@pytest.mark.asyncio
async def test_cli_run_visibility_redaction_and_exit_codes(capsys) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["run", "--max-steps", "9", "--verbose", "hello"])
    assert parsed.max_steps == 9
    assert parsed.verbose is True

    red = _redact_cli_args({"api_key": "sk-secret", "token": "tok123", "content": "a" * 200})
    assert "sk-secret" not in red and "tok123" not in red and "[REDACTED]" in red and "200 chars" in red

    st, code, summary = _summarize_cli_tool_result(
        json.dumps({"success": False, "output": {"exit_code": 1, "output": "FAILED test_add\nline 2"}})
    )
    assert st == "error" and code == 1 and "FAILED test_add" in summary

    # Step limit reached -> exit code 2
    async def loop_stream(p):
        tc = {"id": "call_s1", "index": 0, "type": "function", "function": {"name": "manage_workspace_file", "arguments": {"action": "list_tree", "path": "."}}}
        yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]}

    orch_lim = NativeOrchestrator(stream_provider=loop_stream, tool_executor=AsyncMock(return_value=json.dumps({"success": True, "entries": []})))
    orch_lim.max_steps = 1
    rc_lim = await _execute_headless_run("Inspect", orchestrator=orch_lim, verbose=False)
    assert rc_lim == 2
    err_lim = capsys.readouterr().err
    assert "[task]" in err_lim and "step_limit_exceeded" in err_lim

    # Read-only diagnostic test run exit 1 -> CLI returns 0 with DIAGNOSTIC_TEST_FAILED banner
    diag_turn = 0

    async def diag_stream(p):
        nonlocal diag_turn
        diag_turn += 1
        if diag_turn == 1:
            tc = {"id": "call_pt", "index": 0, "type": "function", "function": {"name": "execute_command", "arguments": {"command": "pytest -v"}}}
            yield {"choices": [{"delta": {"tool_calls": [tc]}, "finish_reason": "stop"}]}
        else:
            yield {"choices": [{"delta": {"content": "Tests fail due to project bug in calc.py."}, "finish_reason": "stop"}]}

    orch_diag = NativeOrchestrator(
        stream_provider=diag_stream,
        tool_executor=AsyncMock(return_value=json.dumps({"status": "error", "exit_code": 1, "output": "1 failed"})),
    )
    rc_diag = await _execute_headless_run(
        "Run pytest -v and explain why tests fail without modifying any files.",
        orchestrator=orch_diag,
    )
    assert rc_diag == 0
    captured = capsys.readouterr()
    assert "Tests fail due to project bug in calc.py." in captured.out
    assert "DIAGNOSTIC_TEST_FAILED" in captured.err


@pytest.mark.asyncio
async def test_native_host_framing_origin_url_policy_and_cli(tmp_path: Path) -> None:
    ext_dir = Path("/home/rutytoi/Documents/zelyr-recovery/worktree-spotless/zelyr/tools/extension")
    ext_id = compute_unpacked_extension_id(ext_dir)
    assert len(ext_id) == 32 and all("a" <= c <= "p" for c in ext_id)

    buf = io.BytesIO()
    write_native_message(buf, {"id": "m1", "action": "ping"})
    buf.seek(0)
    assert read_native_message(buf) == {"id": "m1", "action": "ping"}

    oversize = io.BytesIO(b"\xff\xff\xff\x7f")
    with pytest.raises(ValueError, match="Invalid Native Messaging frame size"):
        read_native_message(oversize, max_bytes=MAX_NATIVE_MESSAGE_BYTES)

    ok_orig, _ = validate_extension_origin(
        f"chrome-extension://{ext_id}/",
        allowed_origins=[f"chrome-extension://{ext_id}/"],
    )
    assert ok_orig
    bad_orig, _ = validate_extension_origin(
        "chrome-extension://bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/",
        allowed_origins=[f"chrome-extension://{ext_id}/"],
    )
    assert not bad_orig

    assert validate_browser_command({"id": "1", "action": "get_active_tab"})[0] is True
    assert validate_browser_command({"id": "2", "action": "evaluate_script", "script": "1+1"})[0] is False
    assert validate_browser_command({"id": "3", "action": "open_tab", "url": "file:///etc/passwd"})[0] is False
    assert validate_browser_command({"id": "4", "action": "open_tab", "url": "http://169.254.169.254/latest"})[0] is False
    assert validate_browser_command({"id": "5", "action": "open_tab", "url": "https://example.org"}, allowed_hosts=["127.0.0.1"])[0] is False
    assert validate_browser_command({"id": "6", "action": "open_tab", "url": "http://127.0.0.1:8080/ok"}, allowed_hosts=["127.0.0.1"])[0] is True

    # ExecutionPolicyEngine enforces URL scheme and allowed_hosts
    pol = ExecutionPolicyEngine(
        ExecutionPolicyConfig(
            policy_mode="autopilot",
            resource_restrictions={"allowed_hosts": ["127.0.0.1"], "denied_hosts": ["blocked.local"]},
        )
    )
    assert pol.evaluate("interact_with_browser", {"action": "open_tab", "url": "file:///etc/passwd"}, required_ring=0).outcome == PolicyDecisionOutcome.DENY
    assert pol.evaluate("interact_with_browser", {"action": "open_tab", "url": "https://other.example"}, required_ring=0).reason_code == "host_not_in_allowlist"
    assert pol.evaluate("interact_with_browser", {"action": "open_tab", "url": "http://127.0.0.1:9000/test"}, required_ring=0).outcome == PolicyDecisionOutcome.ALLOW

    # Install-host writes user-scoped manifest & wrapper safely into isolated temp dir
    prof_dir = tmp_path / "chr_prof"
    wrapper = tmp_path / "bin" / "zelyr-browser-host"
    res = install_native_messaging_host(
        user_data_dir=prof_dir,
        wrapper_path=wrapper,
        allowed_hosts=["127.0.0.1"],
    )
    assert Path(res["manifest_path"]).is_file()
    assert Path(res["launcher_path"]).is_file()
    assert os.access(res["launcher_path"], os.X_OK)

    # Full AF_UNIX + stdio Native Messaging pipe roundtrip with InteractWithBrowserTool
    r_in, w_in = os.pipe()
    r_out, w_out = os.pipe()
    stdin_bin = os.fdopen(r_in, "rb", buffering=0)
    to_host = os.fdopen(w_in, "wb", buffering=0)
    from_host = os.fdopen(r_out, "rb", buffering=0)
    stdout_bin = os.fdopen(w_out, "wb", buffering=0)

    sock_path = tmp_path / "native.sock"
    host = BrowserNativeHost(
        origin=f"chrome-extension://{ext_id}/",
        allowed_origins=[f"chrome-extension://{ext_id}/"],
        socket_path=sock_path,
        allowed_hosts=["127.0.0.1"],
        stdin_stream=stdin_bin,
        stdout_stream=stdout_bin,
    )

    import threading

    def fake_browser_responder() -> None:
        while True:
            try:
                req = read_native_message(from_host)
            except Exception:
                break
            if req is None:
                break
            write_native_message(to_host, {"id": req["id"], "success": True, "action": req["action"], "title": "Mock Page"})

    t_resp = threading.Thread(target=fake_browser_responder, daemon=True)
    t_resp.start()
    host_task = asyncio.create_task(host.run())
    try:
        for _ in range(40):
            if sock_path.exists():
                break
            await asyncio.sleep(0.05)
        assert sock_path.exists()

        old_sock = os.environ.get("ZELYR_BROWSER_NATIVE_SOCKET")
        os.environ["ZELYR_BROWSER_NATIVE_SOCKET"] = str(sock_path)
        try:
            mock_bridge = MagicMock()
            mock_bridge.server = True
            mock_bridge.is_connected.return_value = False
            with patch("zelyr.tools.browser_extension.get_bridge", return_value=mock_bridge):
                tool = InteractWithBrowserTool()
                t_res = await tool.execute({"action": "get_active_tab"})
                assert t_res.success is True
                assert t_res.output["title"] == "Mock Page"
        finally:
            if old_sock is None:
                os.environ.pop("ZELYR_BROWSER_NATIVE_SOCKET", None)
            else:
                os.environ["ZELYR_BROWSER_NATIVE_SOCKET"] = old_sock
    finally:
        for stream in (to_host, from_host, stdout_bin, stdin_bin):
            try:
                stream.close()
            except Exception:
                pass
        await asyncio.wait_for(host_task, timeout=3.0)
        t_resp.join(timeout=1.0)


@pytest.mark.asyncio
async def test_multiline_error_framing_write_preview_and_verifier_snippet(tmp_path: Path, monkeypatch) -> None:
    from zelyr.engine.observation_framing import ObservationFramingEngine
    from zelyr.engine.verifier import verify_command_output
    from zelyr.tools.workspace_file import ManageWorkspaceFileTool

    # 1. Multiline command failures preserve failure lines instead of first banner line
    pytest_fail = (
        "Exit code 1\n"
        "============================= test session starts ==============================\n"
        ">       with pytest.raises(ValueError):\n"
        "E       Failed: DID NOT RAISE ValueError\n"
        "FAILED tests/test_versioning.py::test_invalid - Failed: DID NOT RAISE ValueError\n"
        "1 failed, 3 passed in 0.02s"
    )
    framed = ObservationFramingEngine.frame_observation("execute_command", pytest_fail)
    parsed = ObservationFramingEngine.parse_observation(framed)
    assert "test session starts" not in parsed["error"]
    assert "FAILED tests/test_versioning.py::test_invalid" in parsed["error"]
    assert "1 failed, 3 passed" in parsed["error"]
    assert "DID NOT RAISE ValueError" in parsed["output"]

    from zelyr.cli.main import _summarize_cli_tool_result
    cli_st, cli_ec, cli_sum = _summarize_cli_tool_result(pytest_fail, verbose=False)
    assert cli_st == "error" and cli_ec == 1
    assert "test session starts" not in cli_sum
    assert "FAILED tests/test_versioning.py::test_invalid" in cli_sum

    # 2. WorkspaceFileTool write includes bounded preview metadata like patch
    wf = ManageWorkspaceFileTool()
    f_py = tmp_path / "mod.py"
    w_res = await wf.execute({"action": "write", "path": str(f_py), "content": "x = 1\ny = 2\n", "cwd": str(tmp_path)})
    assert w_res.success is True
    assert w_res.output["line_count"] == 2
    assert w_res.output["written_content"] == "x = 1\ny = 2\n"

    # 3. verify_command_output includes compact output snippet on mismatch
    v_res = verify_command_output("echo failure_reason_xyz", "expected_pattern_abc", cwd=str(tmp_path))
    assert v_res.success is False
    assert v_res.error is not None
    assert "Expected pattern 'expected_pattern_abc' not found in command output" in v_res.error
    assert "failure_reason_xyz" in v_res.error

    # 4. Mutating file edits surface unmatched command_output acceptance failures immediately
    t_script = tmp_path / "fail_script.py"
    seen_tool_contents: list[str] = []
    orch = NativeOrchestrator(max_steps=4)
    monkeypatch.setattr(orch, "get_active_window_context", AsyncMock(return_value=None))

    async def wf_exec(name: str, **kwargs):
        res = await wf.execute(dict(kwargs))
        return res.to_dict() if hasattr(res, "to_dict") else res

    turn = 0

    async def stream_cmd_verify(payload: dict):
        nonlocal turn
        turn += 1
        for msg in payload.get("messages", []):
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                seen_tool_contents.append(msg["content"])
        if turn == 1:
            yield {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_wf_1",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": json.dumps({
                                    "action": "write",
                                    "path": str(t_script),
                                    "content": "print('STILL_BAD')\n",
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
                            "id": "call_wf_2",
                            "type": "function",
                            "function": {
                                "name": "manage_workspace_file",
                                "arguments": json.dumps({
                                    "action": "write",
                                    "path": str(t_script),
                                    "content": "print('NOW_GOOD')\n",
                                }),
                            },
                        }]
                    }
                }]
            }
        else:
            yield {"choices": [{"delta": {"content": "Updated and verified."}}]}

    monkeypatch.chdir(tmp_path)
    async for _ in orch.generate_stream(
        {
            "messages": [{"role": "user", "content": f"Update {t_script} so python3 {t_script.name} prints NOW_GOOD."}],
            "acceptance_criteria": "command_output:echo STILL_BAD:NOW_GOOD",
        },
        stream_provider=stream_cmd_verify,
        tool_executor=wf_exec,
    ):
        pass
    assert any("[Outcome Verification Failed]:" in c for c in seen_tool_contents)
