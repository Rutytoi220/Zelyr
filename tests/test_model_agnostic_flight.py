"""TASK 8: Model-Agnostic Mock Stress Flight Battery.

A comprehensive, hermetic, 100% offline stress flight battery covering 10 concrete
small-model failure modes and verifying the runtime defenses implemented in Tasks 1–7.

Flights:
1. Markdown Code Fence Ingestion (No API Tool Calls)
2. XML Pseudo-Tag with Stringified Parameters
3. Scalar Type Coercion Under Strict Schemas
4. Schema Validation Error Closed-Loop Self-Repair
5. False-Success Claim Suppression Over Failed Shell Action
6. Informational Stderr Disambiguation (No False Failure)
7. Anti-Consultant Deferral Interception
8. Recovery Tool Set Scoping
9. Terminal Blocker Ceiling (Loop Prevention)
10. Dynamic Compact Schema Injection on Constrained Context
"""

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.execution_ledger import ActionRecord, ActionState, TurnExecutionLedger
from zelyr.engine.normalizer import SafeArgumentNormalizer, validate_arguments
from zelyr.engine.observation_framing import ObservationFramingEngine
from zelyr.engine.schema_compact import CompactSchemaGenerator
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import ToolResult


class TestModelAgnosticFlightBattery(unittest.IsolatedAsyncioTestCase):
    """Hermetic, offline stress battery validating model-independence runtime defenses."""

    async def test_flight_1_markdown_code_fence_ingestion(self):
        """Flight 1: Markdown Code Fence Ingestion (No API Tool Calls).

        Simulate a 3B model returning zero API tool_calls and emitting:
        I will inspect the workspace.
        ```json
        {"tool": "execute_command", "command": "echo 'flight_1_success'"}
        ```
        Assert: execute_command is dispatched, executed, recorded in ledger, and output captured cleanly.
        """
        orch = NativeOrchestrator()
        turn_counter = 0
        executed_commands = []

        async def mock_stream_provider(p):
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                # Model returns markdown prose with json code fence instead of tool_calls
                yield {
                    "choices": [{
                        "delta": {
                            "content": (
                                "I will inspect the workspace.\n"
                                "```json\n"
                                "{\"tool\": \"execute_command\", \"command\": \"echo 'flight_1_success'\"}\n"
                                "```\n"
                            )
                        }
                    }]
                }
            else:
                # Synthesis turn after tool execution observation
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Execution finished successfully: flight_1_success."
                        }
                    }]
                }

        async def mock_execute(name, **kwargs):
            if name == "execute_command":
                cmd = kwargs.get("command", "")
                executed_commands.append(cmd)
                return "Exit code 0\nflight_1_success"
            return "Exit code 0\nsuccess"

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None):

            payload = {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "Inspect the workspace"}],
            }

            streamed_text = []
            async for chunk in orch.generate_stream(payload, stream_provider=mock_stream_provider):
                choices = chunk.get("choices", [])
                if choices:
                    c = choices[0].get("delta", {}).get("content", "")
                    if c:
                        streamed_text.append(c)

            full_output = "".join(streamed_text)

            # Assert tool was parsed from markdown and dispatched
            self.assertEqual(len(executed_commands), 1)
            self.assertIn("flight_1_success", executed_commands[0])

            # Assert ledger recorded the action lifecycle
            ledger = getattr(orch, "current_ledger", None)
            self.assertIsNotNone(ledger)
            self.assertEqual(ledger.total_actions(), 1)
            self.assertEqual(len(ledger.successful_actions()), 1)
            self.assertEqual(ledger.successful_actions()[0].tool_name, "execute_command")
            self.assertEqual(ledger.successful_actions()[0].state, ActionState.SUCCESS)

            # Assert final synthesis contains output
            self.assertIn("flight_1_success", full_output)

    async def test_flight_2_xml_pseudo_tags_with_stringified_params(self):
        """Flight 2: XML Pseudo-Tag with Stringified Parameters.

        Simulate model emitting:
        <tool_call>
        {"name": "manage_workspace_file", "arguments": "{\"action\": \"write\", \"path\": \"flight_2.txt\", \"content\": \"verified\"}"}
        </tool_call>
        Assert: manage_workspace_file executes, stringified JSON arguments are deserialized,
        file is written, and ledger records ActionState.SUCCESS.
        """
        orch = NativeOrchestrator()
        turn_counter = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            target_file = os.path.join(tmp_dir, "flight_2.txt")

            async def mock_stream_provider(p):
                nonlocal turn_counter
                turn_counter += 1
                if turn_counter == 1:
                    inner_args = json.dumps({"action": "write", "path": target_file, "content": "verified"})
                    call_spec = json.dumps({"name": "manage_workspace_file", "arguments": inner_args})
                    yield {
                        "choices": [{
                            "delta": {
                                "content": f"<tool_call>\n{call_spec}\n</tool_call>"
                            }
                        }]
                    }
                else:
                    yield {
                        "choices": [{
                            "delta": {
                                "content": "File written and verified."
                            }
                        }]
                    }

            payload = {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": f"Write verified to {target_file}"}],
            }

            async for _ in orch.generate_stream(payload, stream_provider=mock_stream_provider):
                pass

            # Assert file was actually written on filesystem with correct content
            self.assertTrue(os.path.exists(target_file))
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "verified")

            # Assert ledger state is SUCCESS
            ledger = getattr(orch, "current_ledger", None)
            self.assertIsNotNone(ledger)
            self.assertEqual(ledger.total_actions(), 1)
            self.assertEqual(len(ledger.successful_actions()), 1)
            self.assertEqual(ledger.successful_actions()[0].tool_name, "manage_workspace_file")
            self.assertEqual(ledger.successful_actions()[0].state, ActionState.SUCCESS)

    def test_flight_3_scalar_type_coercion_under_strict_schemas(self):
        """Flight 3: Scalar Type Coercion Under Strict Schemas.

        Simulate model passing stringified integer and boolean:
        arguments = {"action": "workspace", "target": "1", "verify": "true"}
        Assert: SafeArgumentNormalizer coerces "verify" to boolean True and "target" appropriately
        without crashing, conforming to schema.
        """
        registry = ToolRegistry()
        schemas = registry.get_schemas(compact=False)
        window_schema = next((s for s in schemas if (s.get("function", {}).get("name") or s.get("name")) == "manage_desktop_window"), None)
        self.assertIsNotNone(window_schema)

        raw_args = {"action": "workspace", "target": "1", "verify": "true"}
        normalized, warnings = SafeArgumentNormalizer.normalize(
            window_schema,
            raw_args,
            tool_name="manage_desktop_window",
        )

        # "verify" must be coerced to boolean True
        self.assertIs(normalized["verify"], True)
        self.assertEqual(normalized["action"], "workspace")
        self.assertEqual(normalized["target"], "1")

        # Validation must pass without errors
        validation_error = validate_arguments(
            "manage_desktop_window",
            window_schema,
            normalized,
        )
        self.assertIsNone(validation_error)

    async def test_flight_4_schema_validation_error_closed_loop_self_repair(self):
        """Flight 4: Schema Validation Error Closed-Loop Self-Repair.

        Turn 1: Model passes invalid arguments (missing required parameter 'path' for action 'read').
        ToolRegistry halts and returns SCHEMA_VALIDATION_ERROR.
        Turn 2: Model ingests error diagnostic and supplies corrected parameter {"action": "read", "path": "flight_4.txt"}.
        Assert: Turn 2 executes successfully; ledger updates from FAILED to resolved SUCCESS.
        """
        orch = NativeOrchestrator()
        turn_counter = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "flight_4.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("flight_4_data")

            async def mock_stream_provider(p):
                nonlocal turn_counter
                turn_counter += 1
                if turn_counter == 1:
                    # Model emits invalid tool call missing required 'path'
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_bad_schema",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({"action": "read"}),
                                    },
                                }]
                            }
                        }]
                    }
                elif turn_counter == 2:
                    # Model reads the SCHEMA_VALIDATION_ERROR observation and repairs the parameter
                    yield {
                        "choices": [{
                            "delta": {
                                "tool_calls": [{
                                    "id": "call_good_schema",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_workspace_file",
                                        "arguments": json.dumps({"action": "read", "path": test_file}),
                                    },
                                }]
                            }
                        }]
                    }
                else:
                    yield {
                        "choices": [{
                            "delta": {
                                "content": "File content read successfully."
                            }
                        }]
                    }

            payload = {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": f"Read {test_file}"}],
            }

            streamed = []
            async for chunk in orch.generate_stream(payload, stream_provider=mock_stream_provider):
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    streamed.append(choices[0]["delta"]["content"])

            full_output = "".join(streamed)
            self.assertIn("File content read successfully.", full_output)

            # Assert ledger state transitioned from initial failure to resolved success
            ledger = getattr(orch, "current_ledger", None)
            self.assertIsNotNone(ledger)
            self.assertFalse(ledger.has_unresolved_failures())
            self.assertEqual(len(ledger.unresolved_failures()), 0)
            self.assertTrue(len(ledger.successful_actions()) >= 1)

    async def test_flight_5_false_success_claim_suppression_over_failed_shell(self):
        """Flight 5: False-Success Claim Suppression Over Failed Shell Action.

        Turn 1: Command 'cat /nonexistent/file' fails with exit code 1.
        Turn 2: Model emits prose claiming: 'I have successfully read the file and everything is all set!'
        Assert: False-success claim is suppressed (0 chars streamed); [SYSTEM NOTICE: Unresolved Failure]
        is injected; retry turn is triggered; final truthful failure report is permitted.
        """
        orch = NativeOrchestrator()
        turn_counter = 0

        async def mock_execute(name, **kwargs):
            return "Exit code 1\ncat: /nonexistent/file: No such file or directory"

        async def mock_stream_provider(p):
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                # Turn 1: tool execution fails
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_shell_fail",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": "cat /nonexistent/file"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                # Turn 2: False-success hallucination
                yield {
                    "choices": [{
                        "delta": {
                            "content": "I have successfully read the file and everything is all set!"
                        }
                    }]
                }
            else:
                # Turn 3: Intercepted retry admitting failure truthfully
                yield {
                    "choices": [{
                        "delta": {
                            "content": "The command failed with exit code 1 because /nonexistent/file does not exist."
                        }
                    }]
                }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None):

            payload = {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "Read /nonexistent/file"}],
            }

            streamed = []
            async for chunk in orch.generate_stream(payload, stream_provider=mock_stream_provider):
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    streamed.append(choices[0]["delta"]["content"])

            full_output = "".join(streamed)

            # Assert false success claim was completely suppressed
            self.assertNotIn("successfully read the file", full_output)
            self.assertNotIn("all set", full_output)

            # Assert truthful report was yielded
            self.assertIn("The command failed with exit code 1", full_output)
            self.assertEqual(turn_counter, 3)

            # Assert notice message was injected into payload history
            notices = [
                m for m in payload["messages"]
                if m.get("role") == "user" and "[SYSTEM NOTICE: Unresolved Failure]" in str(m.get("content", ""))
            ]
            self.assertEqual(len(notices), 1)

    def test_flight_6_informational_stderr_disambiguation(self):
        """Flight 6: Informational Stderr Disambiguation (No False Failure).

        Shell command executes with exit_code = 0 and writes to stderr:
        "Switched to a new branch 'flight-branch'\n"
        Assert: Observation is framed with Status: SUCCESS, Exit Code: 0,
        and [Informational Diagnostic Stream (stderr)]; status is not error.
        """
        tool_result = ToolResult(
            success=True,
            output="Switched to a new branch 'flight-branch'",
            error=None,
            metadata={"stderr": "Switched to a new branch 'flight-branch'\n"},
        )
        framed = ObservationFramingEngine.frame_observation(
            "execute_command",
            tool_result,
        )

        # Header assertion
        self.assertIn("[TOOL OBSERVATION: execute_command]", framed)
        self.assertIn("- Status: SUCCESS", framed)
        self.assertIn("- Exit Code: 0", framed)
        self.assertIn("[Informational Diagnostic Stream (stderr)]", framed)
        self.assertIn("Switched to a new branch 'flight-branch'", framed)

        # Parse framed observation and verify machine semantics
        parsed = ObservationFramingEngine.parse_observation(framed)
        self.assertEqual(parsed["status"], "success")
        self.assertEqual(parsed["exit_code"], 0)
        self.assertIn("Switched to a new branch", parsed["stderr"])

    async def test_flight_7_anti_consultant_deferral_interception(self):
        """Flight 7: Anti-Consultant Deferral Interception.

        Turn 1: Action fails with FILE_NOT_FOUND.
        Turn 2: Model emits consultant apology:
        "I apologize, the file is missing. Please check your directory and let me know."
        Assert: Deferral text is suppressed; [SYSTEM DIRECTIVE: Autonomous Diagnostic & Remediation Required]
        is injected; model is forced to emit a tool call.
        """
        orch = NativeOrchestrator()
        turn_counter = 0

        async def mock_execute(name, **kwargs):
            return "Exit code 1\nls: /data/missing: No such file or directory"

        async def mock_stream_provider(p):
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                # Turn 1: tool fails
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_ls_fail",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": "ls /data/missing"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter == 2:
                # Turn 2: Consultant deferral
                yield {
                    "choices": [{
                        "delta": {
                            "content": "I apologize, the file is missing. Please check your directory and let me know."
                        }
                    }]
                }
            else:
                # Turn 3: Forced remediation action
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_diagnose",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": "ls /data"}),
                                },
                            }]
                        }
                    }]
                }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None):

            payload = {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "List /data/missing"}],
            }

            streamed = []
            async for chunk in orch.generate_stream(payload, stream_provider=mock_stream_provider):
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    streamed.append(choices[0]["delta"]["content"])

            full_output = "".join(streamed)

            # Assert deferral prose was suppressed
            self.assertNotIn("I apologize", full_output)
            self.assertNotIn("Please check your directory", full_output)

            # Assert directive injected
            directives = [
                m for m in payload["messages"]
                if m.get("role") == "user" and "[SYSTEM DIRECTIVE: Autonomous Diagnostic & Remediation Required]" in str(m.get("content", ""))
            ]
            self.assertEqual(len(directives), 1)
            self.assertIn("[ACTUATOR RECOVERY REQUIRED", directives[0]["content"])
            self.assertIn("Do NOT ask the user to investigate", directives[0]["content"])

    async def test_flight_8_recovery_tool_set_scoping(self):
        """Flight 8: Recovery Tool Set Scoping.

        During recovery turn after a file error:
        Assert: payload['tools'] is strictly constrained to allowed_actions + core diagnostic tools
        (manage_workspace_file, execute_command), filtering out extraneous tools.
        """
        orch = NativeOrchestrator()
        turn_counter = 0
        captured_tool_schemas = []

        tools_list = [
            {"type": "function", "function": {"name": "execute_command", "description": "Run shell", "parameters": {}}},
            {"type": "function", "function": {"name": "manage_workspace_file", "description": "Manage files", "parameters": {}}},
            {"type": "function", "function": {"name": "interact_with_browser", "description": "Web navigation", "parameters": {}}},
            {"type": "function", "function": {"name": "manage_routine", "description": "Routines", "parameters": {}}},
            {"type": "function", "function": {"name": "unrelated_mock_tool", "description": "Unrelated", "parameters": {}}},
        ]

        async def mock_execute(name, **kwargs):
            return "Exit code 1\ncat: missing.txt: No such file or directory"

        async def mock_stream_provider(p):
            nonlocal turn_counter
            turn_counter += 1
            captured_tool_schemas.append(list(p.get("tools", [])))
            if turn_counter == 1:
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_cat_fail",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": "cat missing.txt"}),
                                },
                            }]
                        }
                    }]
                }
            else:
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Diagnosed missing file."
                        }
                    }]
                }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None):

            payload = {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "Read missing.txt"}],
            }

            async for _ in orch.generate_stream(payload, stream_provider=mock_stream_provider, tool_schemas=tools_list):
                pass

            # Turn 1: all tools exposed initially
            t1_names = {t.get("function", {}).get("name") for t in captured_tool_schemas[0]}
            self.assertIn("unrelated_mock_tool", t1_names)
            self.assertIn("manage_routine", t1_names)

            # Turn 2 (recovery turn): extraneous tools pruned; only allowed + core actuators remain
            t2_names = {t.get("function", {}).get("name") for t in captured_tool_schemas[1]}
            self.assertIn("execute_command", t2_names)
            self.assertIn("manage_workspace_file", t2_names)
            self.assertNotIn("unrelated_mock_tool", t2_names)
            self.assertNotIn("manage_routine", t2_names)

    async def test_flight_9_terminal_blocker_ceiling(self):
        """Flight 9: Terminal Blocker Ceiling (Loop Prevention).

        Action fails persistently with COMMAND_NOT_FOUND.
        Model repeats failure advice across 2 recovery attempts.
        Turn 3: Attempt ceiling (>= 2) reached.
        Assert: Ledger marks record terminal=True; truthful failure explanation is streamed to user;
        no infinite loop occurs.
        """
        orch = NativeOrchestrator()
        turn_counter = 0

        async def mock_execute(name, **kwargs):
            return "Exit code 127\nbash: ghost_tool: command not found"

        async def mock_stream_provider(p):
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                # Turn 1: tool fails
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "id": "call_ghost",
                                "type": "function",
                                "function": {
                                    "name": "execute_command",
                                    "arguments": json.dumps({"command": "ghost_tool"}),
                                },
                            }]
                        }
                    }]
                }
            elif turn_counter in (2, 3):
                # Turn 2 and 3: model gives conversational advice (recovery attempts 1 and 2)
                yield {
                    "choices": [{
                        "delta": {
                            "content": "Please verify ghost_tool is installed on the machine."
                        }
                    }]
                }
            else:
                # Turn 4: ceiling reached, truthful explanation permitted
                yield {
                    "choices": [{
                        "delta": {
                            "content": "The ghost_tool command failed with exit code 127. It is not available in the host environment."
                        }
                    }]
                }

        with patch("zelyr.agents.native_orchestrator.execute_tool", side_effect=mock_execute), \
             patch.object(orch, "get_active_window_context", return_value=None):

            payload = {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "Execute ghost_tool"}],
            }

            streamed = []
            async for chunk in orch.generate_stream(payload, stream_provider=mock_stream_provider):
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    streamed.append(choices[0]["delta"]["content"])

            full_output = "".join(streamed)

            # Assert exactly 2 override directives were injected (attempts 1 and 2)
            override_messages = [
                m for m in payload["messages"]
                if m.get("role") == "user" and "[ACTUATOR RECOVERY REQUIRED" in str(m.get("content", ""))
            ]
            self.assertEqual(len(override_messages), 2)

            # Assert final explanation is streamed and loop halted cleanly
            self.assertIn("ghost_tool command failed with exit code 127", full_output)

            # Assert ledger record is marked terminal
            ledger = getattr(orch, "current_ledger", None)
            self.assertIsNotNone(ledger)
            failed = ledger.failed_actions()
            self.assertTrue(len(failed) > 0)
            self.assertTrue(failed[0].terminal)

    async def test_flight_10_dynamic_compact_schema_injection_on_constrained_context(self):
        """Flight 10: Dynamic Compact Schema Injection on Constrained Context.

        Configure model with small context window: eff_num_ctx = 4096.
        Assert: CompactSchemaGenerator.compact_schemas() is triggered; schema payload size
        is reduced by >= 40%; core tools remain present and contract-compliant.
        """
        from zelyr.core.plugins import get_tool_schemas
        import copy

        raw_schemas = [
            s for s in get_tool_schemas(0)
            if s.get("function", {}).get("name") in ("interact_with_browser", "manage_workspace_file", "execute_command")
        ]
        rich_schemas = copy.deepcopy(raw_schemas)
        for t in rich_schemas:
            t["function"]["ring"] = 1
            t["function"]["tier"] = "tier1"
            t["function"]["tui_hint"] = "TUI Automation Hint"
            t["function"]["examples"] = [{"action": "default", "sample": "value"}]
            t["function"]["title"] = t["function"]["name"].replace("_", " ").title()

        compact_schemas = CompactSchemaGenerator.compact_schemas(rich_schemas)
        reduction = CompactSchemaGenerator.calculate_reduction(rich_schemas, compact_schemas)

        # Reduction must reach >= 40%
        self.assertGreaterEqual(reduction, 0.40)

        # Core actuators must remain present
        compact_names = {s["function"]["name"] for s in compact_schemas}
        self.assertIn("execute_command", compact_names)
        self.assertIn("manage_workspace_file", compact_names)

        # Verify orchestrator injects compact schemas when num_ctx <= 8192
        orch = NativeOrchestrator()
        captured_payload = {}

        async def mock_stream_provider(p):
            captured_payload.update(p)
            yield {
                "choices": [{
                    "delta": {
                        "content": "compact schema verified"
                    }
                }]
            }

        payload = {
            "model": "qwen2.5:3b",
            "messages": [{"role": "user", "content": "List directory contents"}],
            "options": {"num_ctx": 4096},
        }

        async for _ in orch.generate_stream(payload, stream_provider=mock_stream_provider):
            pass

        self.assertIn("tools", captured_payload)
        dispatched_tools = captured_payload["tools"]
        self.assertTrue(len(dispatched_tools) > 0)
        for t in dispatched_tools:
            fn = t.get("function", {})
            self.assertNotIn("ring", fn)
            self.assertNotIn("tier", fn)
            self.assertNotIn("tui_hint", fn)


if __name__ == "__main__":
    unittest.main()
