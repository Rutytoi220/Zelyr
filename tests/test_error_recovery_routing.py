"""Tests for Task 7: Structured Error Recovery Routing (Anti-Consultant Gate Hardening).

Verifies:
1. Consultant deferral suppression and authoritative remediation directive injection.
2. Active tool scoping during recovery prioritizing allowed_actions + foundational diagnostic tools.
3. Autonomous resolution clearing recovery state and ledger unresolved failures.
4. Terminal blocker ceiling (max 2 attempts) loop prevention.
5. Unresolvable external environment boundaries (e.g. PERMISSION_DENIED) exiting cleanly.
"""

import asyncio
import contextlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.execution_ledger import (
    TurnExecutionLedger,
    detect_consultant_deferral,
    is_truthful_failure_report,
    is_unresolvable_terminal_blocker,
)
from zelyr.tools.core import ToolResult


class TestErrorRecoveryPatternDetection(unittest.TestCase):
    """Test consultant deferral detection and terminal blocker classification."""

    def test_consultant_deferral_patterns(self):
        deferrals = [
            "Please check if the server is running on port 8080.",
            "Please verify the file path exists.",
            "You can try running the build command manually.",
            "You should check the permissions on /var/log.",
            "Let me know if you would like me to proceed.",
            "I apologize, I am unable to locate the configuration file.",
            "I'm sorry, the tool failed to run.",
            "Please provide the database connection string.",
            "This requires manual intervention from the administrator.",
            "Please manually inspect the target directory.",
            "You may want to verify your network connection.",
        ]
        for text in deferrals:
            self.assertTrue(
                detect_consultant_deferral(text),
                f"Failed to detect consultant deferral in: {text}",
            )

    def test_legitimate_prose_not_consultant_deferral(self):
        non_deferrals = [
            "Located build log in /var/log/build.log.",
            "Created file config.json with required parameters.",
            "The process terminated with exit code 127 because jq is not installed in the PATH.",
            "Inspection revealed that directory /tmp/test contains 3 files.",
        ]
        for text in non_deferrals:
            self.assertFalse(
                detect_consultant_deferral(text),
                f"False positive consultant deferral in: {text}",
            )

    def test_unresolvable_terminal_blocker_detection(self):
        # Permission denied is an unresolvable external environment boundary
        self.assertTrue(
            is_unresolvable_terminal_blocker("PERMISSION_DENIED", "Permission denied: /etc/shadow", "execute_command")
        )
        self.assertTrue(
            is_unresolvable_terminal_blocker("EXIT_ERROR", "bash: /root/secure: Operation not permitted", "execute_command")
        )
        self.assertTrue(
            is_unresolvable_terminal_blocker("FILE_ERROR", "Read-only file system", "manage_workspace_file")
        )
        # Resolvable errors should return False
        self.assertFalse(
            is_unresolvable_terminal_blocker("COMMAND_NOT_FOUND", "bash: jq: command not found", "execute_command")
        )
        self.assertFalse(
            is_unresolvable_terminal_blocker("FILE_NOT_FOUND", "No such file or directory: app.py", "manage_workspace_file")
        )
        self.assertFalse(
            is_unresolvable_terminal_blocker("SYNTAX_ERROR", "SyntaxError: invalid syntax", "execute_command")
        )


class TestOrchestratorRecoveryRouting(unittest.IsolatedAsyncioTestCase):
    """Integration tests for NativeOrchestrator recovery routing and anti-consultant enforcement."""

    async def test_consultant_deferral_suppression_and_directive_injection(self):
        """Test orchestrator suppresses consultant deferral and injects authoritative directive."""
        orch = NativeOrchestrator()
        executed_tools = []

        async def mock_execute(name, **kwargs):
            executed_tools.append((name, kwargs))
            return "Exit code 1\ncat: /tmp/missing.conf: No such file or directory"

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                # Model calls tool that fails
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"execute_command\\", \\"arguments\\": {\\"command\\": \\"cat /tmp/missing.conf\\"}}"}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter == 2:
                # Model offers consultant deferral
                lines = [
                    'data: {"choices": [{"delta": {"content": "Please check if /tmp/missing.conf exists or create it manually."}}]}',
                    "data: [DONE]",
                ]
            else:
                # Model complies and diagnoses autonomously
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"execute_command\\", \\"arguments\\": {\\"command\\": \\"ls -la /tmp\\"}}"}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for l in lines:
                    yield l

            resp.aiter_lines = aiter_lines
            yield resp

        payload = {
            "messages": [{"role": "user", "content": "Read /tmp/missing.conf"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            streamed = []
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    c = choices[0].get("delta", {}).get("content", "")
                    if c:
                        streamed.append(c)

            final_text = "".join(streamed)

            # Assert consultant deferral was suppressed from user stream
            self.assertNotIn("Please check if /tmp/missing.conf exists", final_text)

            # Assert directive was injected
            directive_messages = [
                m for m in payload["messages"]
                if m.get("role") == "user" and "[SYSTEM DIRECTIVE: Autonomous Diagnostic & Remediation Required]" in str(m.get("content", ""))
            ]
            self.assertEqual(len(directive_messages), 1)
            directive_content = directive_messages[0]["content"]
            self.assertIn("[ACTUATOR RECOVERY REQUIRED", directive_content)
            self.assertIn("Do NOT ask the user to investigate", directive_content)
            self.assertIn("You possess full host capabilities", directive_content)
            self.assertIn("Allowed Actions:", directive_content)

    async def test_recovery_tool_scoping_prioritizes_allowed_actions(self):
        """Test active tool scoping filters payload tools to allowed actions + foundational tools during recovery."""
        orch = NativeOrchestrator()
        captured_tools = []

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            captured_tools.append(kwargs.get("json", {}).get("tools", []))
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"execute_command\\", \\"arguments\\": {\\"command\\": \\"git checkout branch-x\\"}}"}}]}',
                    "data: [DONE]",
                ]
            else:
                lines = [
                    'data: {"choices": [{"delta": {"content": "Checked branches."}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for l in lines:
                    yield l

            resp.aiter_lines = aiter_lines
            yield resp

        async def mock_execute(name, **kwargs):
            return "Exit code 1\nerror: pathspec 'branch-x' did not match any file(s) known to git"

        tools_list = [
            {"type": "function", "function": {"name": "execute_command", "description": "Run shell command", "parameters": {}}},
            {"type": "function", "function": {"name": "manage_workspace_file", "description": "File manager", "parameters": {}}},
            {"type": "function", "function": {"name": "interact_with_browser", "description": "Web browser", "parameters": {}}},
            {"type": "function", "function": {"name": "random_unrelated_tool", "description": "Unrelated", "parameters": {}}},
        ]
        payload = {
            "messages": [{"role": "user", "content": "Checkout branch"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            async for _ in orch.generate_stream(payload, tool_schemas=tools_list):
                pass

            # In turn 1 (initial dispatch), all tools are present
            t1_names = {t.get("function", {}).get("name") for t in captured_tools[0]}
            self.assertIn("random_unrelated_tool", t1_names)

            # In turn 2 (recovery turn), tools are scoped: unrelated tool must be filtered out
            t2_names = {t.get("function", {}).get("name") for t in captured_tools[1]}
            self.assertIn("execute_command", t2_names)
            self.assertIn("manage_workspace_file", t2_names)
            self.assertNotIn("random_unrelated_tool", t2_names)

    async def test_autonomous_resolution_clears_recovery_and_ledger(self):
        """Test autonomous recovery action resolves ledger failure and completes workflow."""
        orch = NativeOrchestrator()
        executed_tools = []

        async def mock_execute(name, **kwargs):
            executed_tools.append((name, kwargs))
            if name == "execute_command" and "bad_cmd" in kwargs.get("command", ""):
                return "Exit code 127\nbash: bad_cmd: command not found"
            return "Exit code 0\nsuccess"

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                # Turn 1: tool fails
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"execute_command\\", \\"arguments\\": {\\"command\\": \\"bad_cmd\\"}}"}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter == 2:
                # Turn 2: consultant deferral -> suppressed
                lines = [
                    'data: {"choices": [{"delta": {"content": "Please check if bad_cmd is installed."}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter == 3:
                # Turn 3: autonomous recovery tool call
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"execute_command\\", \\"arguments\\": {\\"command\\": \\"good_cmd\\"}}"}}]}',
                    "data: [DONE]",
                ]
            else:
                # Turn 4: final successful synthesis
                lines = [
                    'data: {"choices": [{"delta": {"content": "Resolved by executing good_cmd successfully."}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for l in lines:
                    yield l

            resp.aiter_lines = aiter_lines
            yield resp

        payload = {
            "messages": [{"role": "user", "content": "Execute task"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            streamed = []
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    c = choices[0].get("delta", {}).get("content", "")
                    if c:
                        streamed.append(c)

            final_text = "".join(streamed)

            self.assertIn("Resolved by executing good_cmd successfully.", final_text)
            self.assertEqual(len(executed_tools), 2)
            self.assertEqual(executed_tools[0][1]["command"], "bad_cmd")
            self.assertEqual(executed_tools[1][1]["command"], "good_cmd")

            ledger = getattr(orch, "current_ledger", None)
            self.assertIsNotNone(ledger)
            self.assertFalse(ledger.has_unresolved_failures())
            self.assertEqual(len(ledger.unresolved_failures()), 0)

    async def test_terminal_blocker_ceiling_exits_after_two_attempts(self):
        """Test recovery attempts ceiling (max 2 overrides): after 2 attempts, model failure output is accepted."""
        orch = NativeOrchestrator()

        async def mock_execute(name, **kwargs):
            return "Exit code 127\nbash: non_existent_binary: command not found"

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                # Turn 1: tool fails
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"execute_command\\", \\"arguments\\": {\\"command\\": \\"non_existent_binary\\"}}"}}]}',
                    "data: [DONE]",
                ]
            elif turn_counter in (2, 3):
                # Turn 2 & 3: model responds with consultant deferrals (attempts 1 & 2)
                lines = [
                    'data: {"choices": [{"delta": {"content": "Please check if non_existent_binary is installed."}}]}',
                    "data: [DONE]",
                ]
            else:
                # Turn 4: ceiling reached, truthful terminal explanation
                lines = [
                    'data: {"choices": [{"delta": {"content": "The command non_existent_binary failed with exit code 127. The tool is unavailable on this host."}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for l in lines:
                    yield l

            resp.aiter_lines = aiter_lines
            yield resp

        payload = {
            "messages": [{"role": "user", "content": "Run non_existent_binary"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            streamed = []
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    c = choices[0].get("delta", {}).get("content", "")
                    if c:
                        streamed.append(c)

            final_text = "".join(streamed)

            # Exactly 2 override messages should be injected (attempts 1 and 2)
            override_messages = [
                m for m in payload["messages"]
                if m.get("role") == "user" and "[ACTUATOR RECOVERY REQUIRED]" in str(m.get("content", ""))
            ]
            self.assertEqual(len(override_messages), 2)

            # Final explanation is successfully streamed to user without looping
            self.assertIn("The command non_existent_binary failed", final_text)
            self.assertIn("unavailable on this host", final_text)

            # Ledger record is marked terminal
            ledger = getattr(orch, "current_ledger", None)
            self.assertIsNotNone(ledger)
            failed_records = ledger.failed_actions()
            self.assertTrue(len(failed_records) > 0)
            self.assertTrue(failed_records[0].terminal)

    async def test_unresolvable_permission_denial_exits_cleanly(self):
        """Test unresolvable permission denial exits immediately without recovery looping."""
        orch = NativeOrchestrator()

        async def mock_execute(name, **kwargs):
            return "Exit code 126\nbash: /root/secret: Permission denied"

        turn_counter = 0

        @contextlib.asynccontextmanager
        async def mock_stream_client(method, url, **kwargs):
            nonlocal turn_counter
            turn_counter += 1
            resp = AsyncMock()
            resp.status_code = 200

            if turn_counter == 1:
                lines = [
                    'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"execute_command\\", \\"arguments\\": {\\"command\\": \\"cat /root/secret\\"}}"}}]}',
                    "data: [DONE]",
                ]
            else:
                # Truthful failure report on unresolvable permission boundary
                lines = [
                    'data: {"choices": [{"delta": {"content": "Access to /root/secret failed due to Permission Denied. Root access is not permitted."}}]}',
                    "data: [DONE]",
                ]

            async def aiter_lines():
                for l in lines:
                    yield l

            resp.aiter_lines = aiter_lines
            yield resp

        payload = {
            "messages": [{"role": "user", "content": "Read /root/secret"}],
        }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None), \
             patch("httpx.AsyncClient") as mock_client:

            mock_client.return_value.__aenter__.return_value.stream = mock_stream_client
            mock_client.return_value.__aenter__.return_value.post = AsyncMock()

            streamed = []
            async for chunk in orch.generate_stream(payload):
                choices = chunk.get("choices", [])
                if choices:
                    c = choices[0].get("delta", {}).get("content", "")
                    if c:
                        streamed.append(c)

            final_text = "".join(streamed)

            # Zero recovery overrides injected for unresolvable boundary
            override_messages = [
                m for m in payload["messages"]
                if m.get("role") == "user" and "[ACTUATOR RECOVERY REQUIRED]" in str(m.get("content", ""))
            ]
            self.assertEqual(len(override_messages), 0)

            # Model output streamed cleanly
            self.assertIn("Permission Denied", final_text)

            # Ledger record marked terminal
            ledger = getattr(orch, "current_ledger", None)
            self.assertIsNotNone(ledger)
            failed_records = ledger.failed_actions()
            self.assertTrue(len(failed_records) > 0)
            self.assertTrue(failed_records[0].terminal)


if __name__ == "__main__":
    unittest.main()
