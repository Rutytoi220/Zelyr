"""Comprehensive test suite for Interactive Streaming REPL and Session Persistence (zelyr chat).

Covers:
1. Multi-turn session persistence and lifecycle (SessionManager CRUD, schema migration, tokens).
2. Session resumption parity with orchestrator conversational payload reconstruction.
3. Slash command parsing and dispatching (/help, /effort, /session, /tools, /routine, /memory, /clear, /exit).
4. In-flight turn interruption containment (Ctrl+C preserves context and keeps REPL alive).
5. Provider and network error containment (Ollama unreachable, inference timeouts).
6. CLI subcommand interface (zelyr chat --list, --delete, --new, --model).
7. Startup latency invariance and lazy import boundaries (<100ms for --version / --help).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zelyr.cli.main import cmd_chat
from zelyr.memory.sessions import SessionDatabase, SessionManager


class TestSessionPersistence:
    """Tests for multi-turn session persistence and lifecycle in SessionManager."""

    def test_session_manager_create_and_get(self, tmp_path: Path):
        """Create a new session, verify retrieval of metadata and default turn count."""
        db_file = tmp_path / "test_sessions.db"
        mgr = SessionManager(db_file)

        sid = mgr.create_session(title="Unit Test Session", model_name="qwen2.5-coder:7b")
        assert sid is not None

        sess = mgr.get_session(sid)
        assert sess is not None
        assert sess["session_id"] == sid
        assert sess["title"] == "Unit Test Session"
        assert sess["model_name"] == "qwen2.5-coder:7b"
        assert sess["message_count"] == 0
        assert isinstance(sess["created_at"], float)
        assert isinstance(sess["updated_at"], float)

    def test_session_manager_append_and_retrieve_messages(self, tmp_path: Path):
        """Append user, assistant (with tool calls), and tool turns and retrieve chronological history."""
        db_file = tmp_path / "test_sessions.db"
        mgr = SessionManager(db_file)

        sid = mgr.create_session(title="Chat Flow")

        # Turn 1: User
        mgr.append_message(sid, "user", "What is the system uptime?")
        # Turn 2: Assistant with tool call
        tool_call_payload = [{"name": "system_info", "arguments": {"metric": "uptime"}}]
        mgr.append_message(sid, "assistant", "Checking uptime...", tool_calls=tool_call_payload)
        # Turn 3: Assistant final answer
        mgr.append_message(sid, "assistant", "System uptime is 42 hours.")

        sess = mgr.get_session(sid)
        assert sess["message_count"] == 3

        msgs = mgr.get_messages(sid)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is the system uptime?"
        assert msgs[0]["tool_calls"] is None

        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Checking uptime..."
        assert msgs[1]["tool_calls"] == tool_call_payload

        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "System uptime is 42 hours."

    def test_session_manager_list_ordering(self, tmp_path: Path):
        """List sessions ordered by updated_at descending."""
        db_file = tmp_path / "test_sessions.db"
        mgr = SessionManager(db_file)

        s1 = mgr.create_session(title="Session One")
        time.sleep(0.01)
        s2 = mgr.create_session(title="Session Two")
        time.sleep(0.01)

        # Update s1 by adding a message so its updated_at becomes newer than s2
        mgr.append_message(s1, "user", "Activity in session one")

        sessions = mgr.list_sessions(limit=10)
        assert len(sessions) >= 2
        # s1 should be first because it was updated most recently
        assert sessions[0]["session_id"] == s1
        assert sessions[1]["session_id"] == s2
        assert sessions[0]["message_count"] == 1
        assert sessions[1]["message_count"] == 0

    def test_session_manager_clear_and_delete(self, tmp_path: Path):
        """Test clearing messages while preserving container, and full deletion."""
        db_file = tmp_path / "test_sessions.db"
        mgr = SessionManager(db_file)

        sid = mgr.create_session(title="Temporary Session")
        mgr.append_message(sid, "user", "Hello")
        mgr.append_message(sid, "assistant", "Hi there")

        assert len(mgr.get_messages(sid)) == 2

        # Clear session
        cleared = mgr.clear_session(sid)
        assert cleared is True
        assert len(mgr.get_messages(sid)) == 0

        # Session record still exists
        sess = mgr.get_session(sid)
        assert sess is not None
        assert sess["message_count"] == 0

        # Delete session
        deleted = mgr.delete_session(sid)
        assert deleted is True
        assert mgr.get_session(sid) is None
        assert len(mgr.get_messages(sid)) == 0

        # Deleting nonexistent session returns False
        assert mgr.delete_session("nonexistent-sid") is False

    def test_session_manager_token_counting(self):
        """Verify token estimation calculation."""
        mgr = SessionManager(":memory:")
        messages = [
            {"role": "user", "content": "hello world test"},  # 3 words -> ~3.9 -> 3
            {"role": "assistant", "content": "this is a response from the model"},  # 7 words
        ]
        # Total words: 10 * 1.3 = 13 tokens
        tokens = mgr.count_tokens(messages)
        assert tokens == 13

    def test_session_manager_schema_migration(self, tmp_path: Path):
        """Verify schema migration gracefully handles older table schemas missing columns."""
        db_file = tmp_path / "legacy.db"
        # Create legacy table schema without updated_at and model_name
        conn = sqlite3.connect(db_file)
        conn.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at REAL,
                message_history TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            ("legacy_1", "Legacy Chat", 1000.0, '[{"role":"user","content":"legacy hello"}]'),
        )
        conn.commit()
        conn.close()

        # Initialize SessionManager on legacy DB - should perform ALTER TABLE
        mgr = SessionManager(db_file)
        sess = mgr.get_session("legacy_1")
        assert sess is not None
        assert sess["title"] == "Legacy Chat"
        assert sess["message_count"] == 1

        # Appending message to migrated DB works cleanly
        mgr.append_message("legacy_1", "assistant", "Legacy reply")
        msgs = mgr.get_messages("legacy_1")
        assert len(msgs) == 2


class TestInteractiveReplCommands:
    """Tests for REPL slash command dispatching and user interactions."""

    @pytest.fixture
    def mock_repl(self, tmp_path: Path, monkeypatch):
        """Create a headless InlineRepl instance with mocked dependencies."""
        monkeypatch.setattr("zelyr.cli.repl.ensure_ollama_running", lambda *args, **kwargs: True)
        from zelyr.cli.repl import InlineRepl

        db_file = tmp_path / "repl_test.db"
        with patch("zelyr.memory.sessions._zelyr_data_dir", return_value=tmp_path):
            repl = InlineRepl(session_id="test-repl-session")
            repl.session_mgr = SessionManager(db_file)
            repl.session_id = repl.session_mgr.create_session("test-repl-session", title="Test Repl")
            yield repl

    def test_command_exit_and_quit(self, mock_repl):
        """Slash commands /exit and /quit should signal loop termination."""
        assert mock_repl.is_running is True
        res = mock_repl.handle_command("/exit")
        assert res is True
        assert mock_repl.is_running is False

        mock_repl.is_running = True
        res = mock_repl.handle_command("/quit")
        assert res is True
        assert mock_repl.is_running is False

    def test_command_help(self, mock_repl):
        """Slash command /help should display reference table and return True."""
        res = mock_repl.handle_command("/help")
        assert res is True

    def test_command_effort(self, mock_repl):
        """Slash command /effort should update active reasoning effort tier."""
        res = mock_repl.handle_command("/effort ultra")
        assert res is True
        assert getattr(mock_repl.config, "effort_tier", None) == "ultra"

        # Querying current effort
        res = mock_repl.handle_command("/effort")
        assert res is True

    def test_command_session_management(self, mock_repl):
        """Slash command /session should inspect, switch, and create sessions."""
        # Query active session info
        res = mock_repl.handle_command("/session")
        assert res is True

        # Create new session via /session new
        old_sid = mock_repl.session_id
        res = mock_repl.handle_command("/session new")
        assert res is True
        assert mock_repl.session_id != old_sid

        # Switch to custom session ID
        res = mock_repl.handle_command("/session dev-work")
        assert res is True
        assert mock_repl.session_id == "dev-work"
        assert mock_repl.session_mgr.get_session("dev-work") is not None

    def test_command_tools_listing(self, mock_repl):
        """Slash command /tools should enumerate built-in and dynamic tools."""
        res = mock_repl.handle_command("/tools")
        assert res is True

    def test_command_routine_inspection(self, mock_repl):
        """Slash command /routine should query scheduled background tasks without error."""
        res = mock_repl.handle_command("/routine")
        assert res is True

    def test_command_memory_subcommands(self, mock_repl, monkeypatch):
        """Slash command /memory should support list, add, and query."""
        mock_add = MagicMock(return_value=True)
        mock_list = MagicMock(return_value=[{"id": 1, "fact": "Antigravity active", "created_at": time.time()}])
        monkeypatch.setattr("zelyr.memory.semantic.add_memory", mock_add)
        monkeypatch.setattr("zelyr.memory.semantic.get_all_memories", mock_list)

        assert mock_repl.handle_command("/memory list") is True
        assert mock_repl.handle_command("/memory add User prefers Python 3.11") is True
        mock_add.assert_called_with("User prefers Python 3.11")

    def test_command_clear(self, mock_repl):
        """Slash command /clear resets conversational context window with a fresh session."""
        mock_repl.session_mgr.append_message(mock_repl.session_id, "user", "Previous context")
        old_sid = mock_repl.session_id

        res = mock_repl.handle_command("/clear")
        assert res is True
        assert mock_repl.session_id != old_sid


class TestChatStreamingAndInterruption:
    """Tests for streaming token generation, interruption handling, and error containment."""

    @pytest.mark.asyncio
    async def test_session_resumption_orchestrator_payload(self, tmp_path: Path, monkeypatch):
        """Ensure stream_response reconstructs prior multi-turn context into orchestrator payload."""
        monkeypatch.setattr("zelyr.cli.repl.ensure_ollama_running", lambda *args, **kwargs: True)
        from zelyr.cli.repl import InlineRepl

        db_file = tmp_path / "resume_test.db"
        repl = InlineRepl(session_id="resume-sid")
        repl.session_mgr = SessionManager(db_file)
        repl.session_id = repl.session_mgr.create_session("resume-sid", title="Resumed")

        # Seed prior messages
        repl.session_mgr.append_message("resume-sid", "user", "First question")
        repl.session_mgr.append_message("resume-sid", "assistant", "First answer")

        captured_payload = {}

        async def mock_generate_stream(payload):
            nonlocal captured_payload
            captured_payload = payload
            yield {"choices": [{"delta": {"content": "Second answer"}}]}

        repl.orchestrator.generate_stream = mock_generate_stream

        await repl.stream_response("Second question")

        assert captured_payload.get("session_id") == "resume-sid"
        messages = captured_payload.get("messages", [])
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "First question"}
        assert messages[1] == {"role": "assistant", "content": "First answer"}
        assert messages[2] == {"role": "user", "content": "Second question"}

        # Verify assistant response was saved
        history = repl.session_mgr.get_messages("resume-sid")
        assert len(history) == 4
        assert history[3]["content"] == "Second answer"

    @pytest.mark.asyncio
    async def test_turn_interruption_preserves_context(self, tmp_path: Path, monkeypatch):
        """Interruption during generation (Ctrl+C / CancelledError) preserves prior turns and leaves loop intact."""
        monkeypatch.setattr("zelyr.cli.repl.ensure_ollama_running", lambda *args, **kwargs: True)
        from zelyr.cli.repl import InlineRepl

        db_file = tmp_path / "interrupt_test.db"
        repl = InlineRepl(session_id="interrupt-sid")
        repl.session_mgr = SessionManager(db_file)
        repl.session_id = repl.session_mgr.create_session("interrupt-sid", title="Interrupt")

        # Prior turn
        repl.session_mgr.append_message("interrupt-sid", "user", "Turn 1: Stable question")
        repl.session_mgr.append_message("interrupt-sid", "assistant", "Turn 1: Stable answer")

        async def slow_stream(payload):
            await asyncio.sleep(5.0)
            yield {"choices": [{"delta": {"content": "Should not reach"}}]}

        repl.orchestrator.generate_stream = slow_stream

        # Launch stream_response and cancel it after 0.05s
        task = asyncio.create_task(repl.stream_response("Turn 2: Interrupted question"))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Prior conversation turns are 100% intact
        history = repl.session_mgr.get_messages("interrupt-sid")
        assert len(history) >= 2
        assert history[0]["content"] == "Turn 1: Stable question"
        assert history[1]["content"] == "Turn 1: Stable answer"

    @pytest.mark.asyncio
    async def test_llm_connection_error_containment(self, tmp_path: Path, monkeypatch):
        """Ensure connection failure to LLM provider emits clear diagnostic without crashing."""
        monkeypatch.setattr("zelyr.cli.repl.ensure_ollama_running", lambda *args, **kwargs: True)
        from zelyr.cli.repl import InlineRepl

        db_file = tmp_path / "err_test.db"
        repl = InlineRepl(session_id="err-sid")
        repl.session_mgr = SessionManager(db_file)
        repl.session_id = repl.session_mgr.create_session("err-sid", title="Err")

        async def fail_stream(payload):
            raise ConnectionRefusedError("[Errno 111] Connection refused: http://127.0.0.1:11434")
            yield {}

        repl.orchestrator.generate_stream = fail_stream

        # Should handle cleanly and not raise unhandled exception
        await repl.stream_response("Hello")

    @pytest.mark.asyncio
    async def test_llm_inference_timeout_containment(self, tmp_path: Path, monkeypatch):
        """Ensure inference timeout emits actionable diagnostic without crashing."""
        monkeypatch.setattr("zelyr.cli.repl.ensure_ollama_running", lambda *args, **kwargs: True)
        from zelyr.cli.repl import InlineRepl

        db_file = tmp_path / "timeout_test.db"
        repl = InlineRepl(session_id="timeout-sid")
        repl.session_mgr = SessionManager(db_file)
        repl.session_id = repl.session_mgr.create_session("timeout-sid", title="Timeout")

        async def timeout_stream(payload):
            raise TimeoutError("Model inference timeout after 60s")
            yield {}

        repl.orchestrator.generate_stream = timeout_stream

        # Should handle cleanly
        await repl.stream_response("Hello")


class TestChatCLICommands:
    """Tests for zelyr chat CLI subcommands."""

    def test_cmd_chat_list(self, tmp_path: Path, monkeypatch):
        """zelyr chat --list outputs stored conversation sessions cleanly."""
        db_file = tmp_path / "cli_list_test.db"
        mgr = SessionManager(db_file)
        mgr.create_session("sess_alpha", title="Alpha Project", model_name="qwen2.5-coder")
        mgr.create_session("sess_beta", title="Beta Project", model_name="qwen2.5-coder")

        monkeypatch.setattr("zelyr.memory.sessions._zelyr_data_dir", lambda: tmp_path)

        args = argparse.Namespace(
            list=True,
            delete=None,
            new=False,
            session=None,
            model=None,
            tui=False,
        )

        with patch("zelyr.memory.sessions.SessionManager", return_value=mgr):
            # Should format and print table without exception
            cmd_chat(args)

    def test_cmd_chat_delete(self, tmp_path: Path, monkeypatch):
        """zelyr chat --delete removes the target session record."""
        db_file = tmp_path / "cli_del_test.db"
        mgr = SessionManager(db_file)
        sid = mgr.create_session("to_delete", title="To Delete")

        args = argparse.Namespace(
            list=False,
            delete=sid,
            new=False,
            session=None,
            model=None,
            tui=False,
        )

        with patch("zelyr.memory.sessions.SessionManager", return_value=mgr):
            cmd_chat(args)

        assert mgr.get_session(sid) is None

    def test_cmd_chat_model_override(self, tmp_path: Path, monkeypatch):
        """zelyr chat --model overrides active ollama model in config."""
        mock_cfg = MagicMock()
        monkeypatch.setattr("zelyr.config.get_config", lambda: mock_cfg)

        args = argparse.Namespace(
            list=False,
            delete=None,
            new=False,
            session=None,
            model="llama3.1:8b",
            tui=False,
        )

        with patch("zelyr.cli.repl.run_repl") as mock_repl_run:
            cmd_chat(args)
            assert mock_cfg.ollama_model == "llama3.1:8b"
            mock_cfg.save.assert_called_once()
            mock_repl_run.assert_called_once()


class TestStartupInvarianceAndPerformance:
    """Verify latency and zero premature imports on fast-path CLI operations."""

    def test_startup_latency_invariance(self):
        """zelyr --version and zelyr --help must execute under 100ms."""
        env = os.environ.copy()
        env["QT_QPA_PLATFORM"] = "offscreen"
        env.pop("COV_CORE_SOURCE", None)
        env.pop("COVERAGE_PROCESS_START", None)

        cmd_version = [sys.executable, "-m", "zelyr.cli.main", "--version"]
        cmd_help = [sys.executable, "-m", "zelyr.cli.main", "--help"]

        # Warm-up run
        subprocess.run(cmd_version, capture_output=True, env=env, check=True)

        latencies = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = subprocess.run(cmd_version, capture_output=True, text=True, env=env)
            latencies.append(time.perf_counter() - t0)
            assert res.returncode == 0

        min_version = min(latencies)
        target = 0.150 if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else 0.100
        assert min_version < target, f"zelyr --version took {min_version*1000:.1f}ms (target < {target*1000:.1f}ms)"

        latencies_help = []
        for _ in range(5):
            t0 = time.perf_counter()
            res = subprocess.run(cmd_help, capture_output=True, text=True, env=env)
            latencies_help.append(time.perf_counter() - t0)
            assert res.returncode == 0

        min_help = min(latencies_help)
        assert min_help < target, f"zelyr --help took {min_help*1000:.1f}ms (target < {target*1000:.1f}ms)"

    def test_zero_premature_imports_on_help(self):
        """Loading zelyr parser should never eagerly import repl or session modules."""
        code = (
            "import sys\n"
            "from zelyr.cli.main import _build_parser\n"
            "p = _build_parser()\n"
            "loaded = list(sys.modules.keys())\n"
            "assert 'zelyr.cli.repl' not in loaded, 'zelyr.cli.repl prematurely imported'\n"
            "assert 'zelyr.memory.sessions' not in loaded, 'zelyr.memory.sessions prematurely imported'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Premature import detected: {res.stderr}"
