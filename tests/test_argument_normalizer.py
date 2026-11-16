"""Unit and Integration Tests for SafeArgumentNormalizer and Schema Validation."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.normalizer import SafeArgumentNormalizer, validate_arguments
from zelyr.tool_registry import ToolRegistry
from zelyr.tools.core import ToolResult


class TestSafeArgumentNormalizerUnit:
    """Verify deterministic coercions and non-speculative normalization."""

    @pytest.fixture
    def sample_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "config": {"type": "object"},
                        "items": {"type": "array"},
                        "label": {"type": "string"},
                    },
                    "required": ["label", "count"],
                },
            },
        }

    def test_integer_coercion(self, sample_schema):
        # String integer -> int
        args = {"count": "123", "label": "test"}
        norm, warnings = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["count"] == 123
        assert isinstance(norm["count"], int)
        assert warnings is not None

        # Float with zero fractional part -> int
        args = {"count": 123.0, "label": "test"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["count"] == 123
        assert isinstance(norm["count"], int)

        # Negative integer string -> int
        args = {"count": "-42", "label": "test"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["count"] == -42

        # Invalid string integer -> untouched
        args = {"count": "abc", "label": "test"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["count"] == "abc"

        # Float with fraction -> untouched
        args = {"count": 1.5, "label": "test"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["count"] == 1.5

    def test_number_coercion(self, sample_schema):
        args = {"ratio": "3.14159"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["ratio"] == 3.14159
        assert isinstance(norm["ratio"], float)

        args = {"ratio": "42"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["ratio"] == 42.0

        # Invalid float string -> untouched
        args = {"ratio": "not_a_float"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["ratio"] == "not_a_float"

    def test_boolean_coercion(self, sample_schema):
        # Case-insensitive string true
        for true_val in ("true", "True", "TRUE", "1", "yes", "YES", "t", "T"):
            args = {"enabled": true_val}
            norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
            assert norm["enabled"] is True, f"Failed on {true_val}"

        # Case-insensitive string false
        for false_val in ("false", "False", "FALSE", "0", "no", "NO", "f", "F"):
            args = {"enabled": false_val}
            norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
            assert norm["enabled"] is False, f"Failed on {false_val}"

        # Integer 1 and 0
        args = {"enabled": 1}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["enabled"] is True

        args = {"enabled": 0}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["enabled"] is False

        # Ambiguous values left untouched
        for ambig in ("maybe", "unknown", 2, -1):
            args = {"enabled": ambig}
            norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
            assert norm["enabled"] == ambig

    def test_object_and_array_deserialization(self, sample_schema):
        # JSON-encoded object string
        args = {"config": '{"theme": "dark", "retries": 3}'}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["config"] == {"theme": "dark", "retries": 3}
        assert isinstance(norm["config"], dict)

        # JSON-encoded array string
        args = {"items": '["alpha", "beta", "gamma"]'}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["items"] == ["alpha", "beta", "gamma"]
        assert isinstance(norm["items"], list)

        # Malformed JSON left untouched
        args = {"config": "{bad json"}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["config"] == "{bad json"

    def test_string_trimming(self, sample_schema):
        args = {"label": "   padded_string   "}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["label"] == "padded_string"

    def test_non_speculative_preservation_of_unknown_arguments(self, sample_schema):
        # Unknown arguments are NOT dropped, guessed, or re-assigned
        args = {"label": "valid", "count": 5, "unknown_field": "preserved", "another_unknown": 42}
        norm, _ = SafeArgumentNormalizer.normalize(sample_schema, args)
        assert norm["unknown_field"] == "preserved"
        assert norm["another_unknown"] == 42
        assert "label" in norm
        assert "count" in norm

    def test_approved_canonical_alias_mapping(self):
        schema = {
            "type": "function",
            "function": {
                "name": "manage_workspace_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["action", "path"],
                },
            },
        }
        args = {"action": "read", "file_path": "/tmp/test.txt"}
        norm, warnings = SafeArgumentNormalizer.normalize(schema, args, tool_name="manage_workspace_file")
        assert "path" in norm
        assert norm["path"] == "/tmp/test.txt"
        assert "file_path" not in norm
        assert warnings is not None


class TestSchemaValidationErrors:
    """Verify validate_arguments produces strict SCHEMA_VALIDATION_ERROR envelopes."""

    @pytest.fixture
    def file_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "manage_workspace_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "path": {"type": "string"},
                        "max_lines": {"type": "integer"},
                    },
                    "required": ["action", "path"],
                },
            },
        }

    def test_missing_required_field_returns_schema_error(self, file_schema):
        args = {"action": "read"}  # Missing required 'path'
        envelope = validate_arguments("manage_workspace_file", file_schema, args)

        assert envelope is not None
        assert envelope["status"] == "error"
        assert envelope["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert envelope["invalid_parameter"] == "path"
        assert envelope["expected_type"] == "string"
        assert envelope["received_value"] == "missing"
        assert "Missing required parameter 'path'" in envelope["error"]
        assert "remedy_hint" in envelope

    def test_empty_required_field_returns_schema_error(self, file_schema):
        args = {"action": "read", "path": ""}  # Empty string for required path
        envelope = validate_arguments("manage_workspace_file", file_schema, args)

        assert envelope is not None
        assert envelope["status"] == "error"
        assert envelope["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert envelope["invalid_parameter"] == "path"
        assert envelope["received_value"] == "empty"
        assert "cannot be empty" in envelope["error"]

    def test_type_mismatch_returns_schema_error(self, file_schema):
        args = {"action": "read", "path": "/valid/path.txt", "max_lines": "not_an_int"}
        envelope = validate_arguments("manage_workspace_file", file_schema, args)

        assert envelope is not None
        assert envelope["status"] == "error"
        assert envelope["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert envelope["invalid_parameter"] == "max_lines"
        assert envelope["expected_type"] == "integer"
        assert envelope["received_value"] == "not_an_int"
        assert "must be of type integer" in envelope["error"]

    def test_valid_arguments_return_none(self, file_schema):
        args = {"action": "read", "path": "/valid/path.txt", "max_lines": 100}
        envelope = validate_arguments("manage_workspace_file", file_schema, args)
        assert envelope is None


class TestToolRegistryValidationIntegration:
    """Verify ToolRegistry halts execution and returns ToolResult on schema validation failure."""

    def test_registry_execute_halts_on_invalid_argument_type(self):
        reg = ToolRegistry()
        # manage_routine requires action (string) and limit (integer)
        result = reg.execute("manage_routine", action="list", limit="uncoercible_string")
        assert isinstance(result, ToolResult)
        assert not result.success
        assert result.metadata.get("error_type") == "SCHEMA_VALIDATION_ERROR"
        assert "limit" in str(result.error)

    def test_registry_execute_coerces_valid_string_types(self):
        reg = ToolRegistry()
        # manage_desktop_window has verify (boolean)
        result = reg.execute("manage_desktop_window", action="list", verify="true")
        assert isinstance(result, ToolResult)
        assert result.success

    @pytest.mark.asyncio
    async def test_registry_execute_async_halts_on_missing_required_argument(self):
        reg = ToolRegistry()
        # manage_desktop_window requires 'action'
        result = await reg.execute_async("manage_desktop_window", target="0x123")
        assert isinstance(result, ToolResult)
        assert not result.success
        assert result.metadata.get("error_type") == "SCHEMA_VALIDATION_ERROR"
        assert "action" in str(result.error)


class TestNativeOrchestratorFallbackCoercionIntegration:
    """Verify FallbackToolParser extracted Markdown tool calls have arguments coerced safely."""

    @pytest.mark.asyncio
    async def test_orchestrator_coerces_markdown_stringified_arguments(self):
        orch = NativeOrchestrator()
        dispatched_args = {}

        async def mock_execute(name, **kwargs):
            nonlocal dispatched_args
            dispatched_args = dict(kwargs)
            return json.dumps({"status": "success", "window": "0x123"})

        orch.execute_tool = mock_execute

        # Model stream emits Markdown code fence with stringified boolean "verify": "true"
        markdown_response = (
            "I will switch to the target workspace now:\n\n"
            "```json\n"
            "{\n"
            '  "name": "manage_desktop_window",\n'
            '  "arguments": {\n'
            '    "action": "workspace",\n'
            '    "target": "2",\n'
            '    "verify": "true"\n'
            "  }\n"
            "}\n"
            "```\n"
        )

        async def mock_stream_provider(p):
            yield {
                "choices": [{
                    "delta": {
                        "content": markdown_response,
                    }
                }]
            }

        payload = {
            "model": "qwen2.5-coder:7b",
            "messages": [{"role": "user", "content": "switch to workspace 2"}],
        }

        async for _ in orch.generate_stream(payload, stream_provider=mock_stream_provider):
            pass

        # Assert actuator received native boolean True, not string "true"
        assert "verify" in dispatched_args
        assert dispatched_args["verify"] is True
        assert isinstance(dispatched_args["verify"], bool)
        assert dispatched_args["action"] == "workspace"
