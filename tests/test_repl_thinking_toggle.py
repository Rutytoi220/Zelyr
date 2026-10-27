"""Test verification suite for Collapsible Reasoning and Ctrl+O Toggle in Zelyr REPL."""

import asyncio
import io
import sys
from unittest.mock import AsyncMock, patch

from zelyr.cli.repl import InlineRepl, COMMANDS
from zelyr.config import ZelyrConfig, get_config


def test_config_persistence():
    print("[TEST 1] Testing show_thinking config persistence...")
    cfg = get_config()
    orig = cfg.show_thinking

    try:
        repl = InlineRepl(session_id="test_toggle_session")
        # Ensure starting state
        repl.config.show_thinking = True
        repl.config.save()

        # Toggle to False
        new_state = repl.toggle_thinking()
        assert new_state is False, f"Expected False, got {new_state}"
        loaded_cfg = ZelyrConfig.load()
        assert loaded_cfg.show_thinking is False, "Config file did not persist show_thinking=False"

        # Toggle back to True
        new_state = repl.toggle_thinking()
        assert new_state is True, f"Expected True, got {new_state}"
        loaded_cfg = ZelyrConfig.load()
        assert loaded_cfg.show_thinking is True, "Config file did not persist show_thinking=True"
        print("  ✓ Config persistence verified.")
    finally:
        cfg.show_thinking = orig
        cfg.save()


def test_keybinding_and_commands():
    print("[TEST 2] Testing keybindings registry and /thought command...")
    repl = InlineRepl(session_id="test_cmd_session")

    # Check /thought in COMMANDS
    assert "/thought" in COMMANDS, "/thought missing from COMMANDS registry"

    # Check keybindings registered on prompt_session
    kb = repl.prompt_session.key_bindings
    assert kb is not None, "Keybindings not configured on PromptSession"
    bindings = kb.bindings
    c_o_bindings = [
        b for b in bindings
        if any("c-o" in getattr(k, "value", "") or "ControlO" in str(k) or "c-o" in str(k) for k in b.keys)
    ]
    assert len(c_o_bindings) > 0, "c-o keybinding not found on PromptSession"

    # Test /thought when empty
    repl.last_thought = ""
    handled = repl.handle_command("/thought")
    assert handled is True, "/thought command failed to handle"

    # Test /thought with stored trace
    repl.last_thought = "Let's factor x^2 + 5x + 6 = (x+2)(x+3)"
    handled = repl.handle_command("/thought")
    assert handled is True

    # Test /thought toggle
    curr = repl.config.show_thinking
    handled = repl.handle_command("/thought toggle")
    assert handled is True
    assert repl.config.show_thinking != curr
    repl.handle_command("/thought toggle")  # Restore
    print("  ✓ Keybindings and /thought command verified.")


def test_collapsed_reasoning_stream():
    print("[TEST 3] Testing Collapsed Mode (show_thinking=False)...")
    repl = InlineRepl(session_id="test_collapsed_session")
    repl.config.show_thinking = False

    async def mock_stream(payload):
        yield {"choices": [{"delta": {"reasoning_content": "Step 1: analyze"}}]}
        yield {"choices": [{"delta": {"reasoning_content": " Step 2: calculate"}}]}
        yield {"choices": [{"delta": {"content": "Result is 42."}}]}

    repl.orchestrator.generate_stream = mock_stream

    captured_out = io.StringIO()
    with patch("sys.stdout.write", side_effect=captured_out.write):
        asyncio.run(repl.stream_response("What is the answer?"))

    out = captured_out.getvalue()
    assert "Thinking..." in out, f"Transient status 'Thinking...' missing from output: {out}"
    assert "▾ Thought for" in out, f"Summary '▾ Thought for' missing from output: {out}"
    assert "Ctrl+O to view" in out, f"'Ctrl+O to view' missing from output: {out}"
    assert "Result is 42." in out, f"Result content missing from output: {out}"
    assert "Step 1: analyze Step 2: calculate" in repl.last_thought, f"Unexpected last_thought: {repl.last_thought}"
    print("  ✓ Collapsed mode verified.")


def test_expanded_reasoning_stream():
    print("[TEST 4] Testing Expanded Mode (show_thinking=True)...")
    repl = InlineRepl(session_id="test_expanded_session")
    repl.config.show_thinking = True

    async def mock_stream(payload):
        yield {"choices": [{"delta": {"content": "<think>Deconstruct problem step by step."}}]}
        yield {"choices": [{"delta": {"content": " Conclude proof.</think>Here is the proof."}}]}

    repl.orchestrator.generate_stream = mock_stream

    captured_out = io.StringIO()
    with patch("sys.stdout.write", side_effect=captured_out.write):
        asyncio.run(repl.stream_response("Prove theorem"))

    out = captured_out.getvalue()
    assert "▾ Thinking:" in out, f"Header '▾ Thinking:' missing from output: {out}"
    assert "Deconstruct problem step by step." in out, f"Reasoning missing from output: {out}"
    assert "───" in out, f"Divider '───' missing from output: {out}"
    assert "Here is the proof." in out, f"Content missing from output: {out}"
    assert "Deconstruct problem step by step. Conclude proof." in repl.last_thought
    print("  ✓ Expanded mode verified.")


if __name__ == "__main__":
    test_config_persistence()
    test_keybinding_and_commands()
    test_collapsed_reasoning_stream()
    test_expanded_reasoning_stream()
    print("\n✅ ALL REPL COLLAPSIBLE REASONING TESTS PASSED!")
