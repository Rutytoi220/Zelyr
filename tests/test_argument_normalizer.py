"""Unit and Integration Tests for SafeArgumentNormalizer and Schema Validation."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

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
        # Mock external Hyprland IPC execution boundary to isolate argument normalization from host compositor availability
        with patch("zelyr.tools.hyprland_ipc._handle_window_action", new_callable=AsyncMock) as mock_handle:
            with patch("shutil.which", return_value="/usr/bin/hyprctl"):
                mock_handle.return_value = {"status": "success", "action": "list", "clients": []}

                # 1. String "true" is normalized to boolean True and passed to execution boundary
                result = reg.execute("manage_desktop_window", action="list", verify="true")
                assert isinstance(result, ToolResult)
                assert result.success
                mock_handle.assert_called_once()
                _, kwargs = mock_handle.call_args
                assert kwargs["verify"] is True
                assert type(kwargs["verify"]) is bool

                # 2. String "false" is normalized to boolean False
                mock_handle.reset_mock()
                result_false = reg.execute("manage_desktop_window", action="list", verify="false")
                assert result_false.success
                _, kwargs_false = mock_handle.call_args
                assert kwargs_false["verify"] is False
                assert type(kwargs_false["verify"]) is bool

                # 3. Invalid boolean string fails closed before execution boundary
                mock_handle.reset_mock()
                result_invalid = reg.execute("manage_desktop_window", action="list", verify="uncoercible_bool")
                assert isinstance(result_invalid, ToolResult)
                assert not result_invalid.success
                assert result_invalid.metadata.get("error_type") == "SCHEMA_VALIDATION_ERROR"
                assert mock_handle.call_count == 0

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


# ---------------------------------------------------------------------------
# Comprehensive Hardening, Security Boundary & Integration Tests (Zelyr 1.4)
# ---------------------------------------------------------------------------


class TestArgumentNormalizationEdgeAndConstraintCases:
    """Covers strict JSON parsing, nested/optional schemas, constraints, and ambiguity rejection."""

    def test_valid_json_string_and_empty_object_where_permitted(self):
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        opt_schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1},
                "verbose": {"type": "boolean"},
                "path": {"type": "string"},
            },
            "additionalProperties": False,
        }

        norm_empty, warnings_empty, err_empty = SafeArgumentNormalizer.normalize_and_validate(
            "list_items", opt_schema, "  {}  ", argument_source="model"
        )
        assert err_empty is None
        assert norm_empty == {}
        assert warnings_empty == []

        norm_str, warnings_str, err_str = SafeArgumentNormalizer.normalize_and_validate(
            "list_items",
            opt_schema,
            '  {"limit": " 25 ", "verbose": "true", "path": "  src/app.py  "}  ',
            argument_source="model",
        )
        assert err_str is None
        assert norm_str == {"limit": 25, "verbose": True, "path": "src/app.py"}
        assert len(warnings_str) >= 2

    def test_nested_objects_arrays_and_additional_properties_schema(self):
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        schema = {
            "type": "object",
            "properties": {
                "config": {
                    # Intentionally omits explicit "type": "object" to verify nested inference
                    "properties": {
                        "retries": {"type": "integer", "minimum": 0, "maximum": 5},
                        "dry_run": {"type": "boolean"},
                    },
                    "required": ["retries"],
                    "additionalProperties": False,
                },
                "ports": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "uniqueItems": True,
                    "items": {"type": "integer", "minimum": 1},
                },
                "labels": {
                    "type": "object",
                    "minProperties": 1,
                    "maxProperties": 2,
                    "additionalProperties": {"type": "integer"},
                },
            },
            "required": ["config", "ports"],
            "additionalProperties": False,
        }

        norm, _, err = SafeArgumentNormalizer.normalize_and_validate(
            "deploy",
            schema,
            {
                "config": '{"retries": "2", "dry_run": "false"}',
                "ports": ["80", "443"],
                "labels": {"worker": "4"},
            },
            argument_source="model",
        )
        assert err is None
        assert norm == {
            "config": {"retries": 2, "dry_run": False},
            "ports": [80, 443],
            "labels": {"worker": 4},
        }

        # Unknown nested field rejected under additionalProperties=False
        _, _, unknown_err = SafeArgumentNormalizer.normalize_and_validate(
            "deploy",
            schema,
            {"config": {"retries": 1, "extra": "nope"}, "ports": [80]},
            argument_source="model",
        )
        assert unknown_err is not None
        assert unknown_err["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert unknown_err["error_category"] == "unknown_parameter"
        assert unknown_err["invalid_parameter"] == "config.extra"
        assert unknown_err["recoverable"] is True

        # Unique array items constraint failure
        _, _, dup_err = SafeArgumentNormalizer.normalize_and_validate(
            "deploy",
            schema,
            {"config": {"retries": 1}, "ports": ["80", 80]},
            argument_source="model",
        )
        assert dup_err is not None
        assert dup_err["error_category"] == "constraint_violation"
        assert dup_err["invalid_parameter"] == "ports"

        # minProperties violation on labels
        _, _, prop_count_err = SafeArgumentNormalizer.normalize_and_validate(
            "deploy",
            schema,
            {"config": {"retries": 1}, "ports": [80], "labels": {}},
            argument_source="model",
        )
        assert prop_count_err is not None
        assert prop_count_err["error_category"] == "constraint_violation"
        assert prop_count_err["invalid_parameter"] == "labels"

    def test_invalid_json_single_quotes_nan_and_no_code_execution(self, monkeypatch):
        import ast as _ast
        import builtins
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        # Prove eval/exec/ast.literal_eval are never invoked during normalization
        monkeypatch.setattr(_ast, "literal_eval", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("ast.literal_eval called")))
        monkeypatch.setattr(builtins, "eval", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("eval called")))
        monkeypatch.setattr(builtins, "exec", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("exec called")))

        schema = {"type": "object", "properties": {"x": {"type": "number"}}, "additionalProperties": False}
        for bad_json in ("{'x': 1}", '{"x": NaN}', '{"x": Infinity}', "", "{broken"):
            _, _, err = SafeArgumentNormalizer.normalize_and_validate(
                "calc", schema, bad_json, argument_source="model"
            )
            assert err is not None
            assert err["error_type"] == "SCHEMA_VALIDATION_ERROR"
            assert err["error_category"] == "parse_error"
            assert err["recoverable"] is True

    def test_wrong_top_level_types_rejected(self):
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        for bad_top in ([1, 2], "[1, 2]", 123, "42", False, None, "null"):
            _, _, err = SafeArgumentNormalizer.normalize_and_validate(
                "noop", schema, bad_top, argument_source="model"
            )
            assert err is not None
            assert err["error_type"] == "SCHEMA_VALIDATION_ERROR"
            assert err["error_category"] == "invalid_top_level_type"
            assert err["expected"] == "object"

    def test_ambiguous_lossy_and_multi_type_conversions_rejected(self):
        import math
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        int_schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
        for bad_int in (1.5, "1.5", "1.0", "010", True, False, math.nan, math.inf, "NaN"):
            _, _, err = SafeArgumentNormalizer.normalize_and_validate(
                "counter", int_schema, {"count": bad_int}, argument_source="model"
            )
            assert err is not None
            assert err["error_type"] == "SCHEMA_VALIDATION_ERROR"
            assert err["invalid_parameter"] == "count"
            if not isinstance(bad_int, bool):
                assert err["error_category"] == "ambiguous_or_lossy_conversion"
            else:
                assert err["error_category"] == "type_mismatch"

        bool_schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}, "required": ["flag"]}
        for bad_bool in ("maybe", 2, -1, 1.5):
            _, _, err = SafeArgumentNormalizer.normalize_and_validate(
                "toggle", bool_schema, {"flag": bad_bool}, argument_source="model"
            )
            assert err is not None
            assert err["error_type"] == "SCHEMA_VALIDATION_ERROR"

        str_schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        for bad_str in (123, True, ["a"], {"a": 1}):
            _, _, err = SafeArgumentNormalizer.normalize_and_validate(
                "greet", str_schema, {"name": bad_str}, argument_source="model"
            )
            assert err is not None
            assert err["error_category"] == "type_mismatch"

        # Multi-type ambiguous union: "42" stays string (must not speculatively coerce)
        union_schema = {
            "type": "object",
            "properties": {"val": {"type": ["string", "integer"]}},
            "required": ["val"],
        }
        norm_u, _, err_u = SafeArgumentNormalizer.normalize_and_validate(
            "union_tool", union_schema, {"val": "42"}, argument_source="model"
        )
        assert err_u is None
        assert norm_u["val"] == "42" and isinstance(norm_u["val"], str)

        # Ambiguous union where neither matches: ("42" with ["boolean", "integer"]) -> no guessing
        ambig_schema = {
            "type": "object",
            "properties": {"val": {"type": ["boolean", "integer"]}},
            "required": ["val"],
        }
        _, _, err_ambig = SafeArgumentNormalizer.normalize_and_validate(
            "ambig_tool", ambig_schema, {"val": "42"}, argument_source="model"
        )
        assert err_ambig is not None
        assert err_ambig["error_type"] == "SCHEMA_VALIDATION_ERROR"

    def test_null_handling_respects_schema_and_source(self):
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        schema = {
            "type": "object",
            "properties": {
                "nullable_union": {"type": ["string", "null"]},
                "nullable_anyof": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "strict_opt": {"type": "string"},
            },
            "additionalProperties": False,
        }

        norm_ok, _, err_ok = SafeArgumentNormalizer.normalize_and_validate(
            "null_test",
            schema,
            {"nullable_union": None, "nullable_anyof": "7"},
            argument_source="model",
        )
        assert err_ok is None
        assert norm_ok == {"nullable_union": None, "nullable_anyof": 7}

        # Model cannot pass explicit null into a non-nullable optional property
        _, _, err_model_null = SafeArgumentNormalizer.normalize_and_validate(
            "null_test",
            schema,
            {"strict_opt": None},
            argument_source="model",
        )
        assert err_model_null is not None
        assert err_model_null["error_category"] == "type_mismatch"
        assert err_model_null["invalid_parameter"] == "strict_opt"

        # Internal Python callers may still forward None for non-required Python Optional fields
        norm_internal, _, err_internal = SafeArgumentNormalizer.normalize_and_validate(
            "null_test",
            schema,
            {"strict_opt": None},
            argument_source="internal",
        )
        assert err_internal is None
        assert norm_internal == {"strict_opt": None}

    def test_constraints_enum_const_bounds_multipleof_pattern_and_combinators(self):
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        schema = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "safe"]},
                "version": {"type": "integer", "const": 2},
                "step": {"type": "number", "minimum": 0.5, "exclusiveMaximum": 10.0, "multipleOf": 0.5},
                "tag": {"type": "string", "minLength": 3, "maxLength": 8, "pattern": r"^[a-z0-9-]+$"},
                "target": {"oneOf": [{"type": "integer"}, {"type": "number", "minimum": 0}]},
            },
            "required": ["mode", "version", "step", "tag"],
            "additionalProperties": False,
        }

        ok_args, _, ok_err = SafeArgumentNormalizer.normalize_and_validate(
            "constrained",
            schema,
            {"mode": "fast", "version": "2", "step": "2.5", "tag": "v1-ok"},
            argument_source="model",
        )
        assert ok_err is None
        assert ok_args == {"mode": "fast", "version": 2, "step": 2.5, "tag": "v1-ok"}

        bad_cases = [
            ({"mode": "turbo", "version": 2, "step": 2.5, "tag": "v1-ok"}, "mode"),
            ({"mode": "fast", "version": 3, "step": 2.5, "tag": "v1-ok"}, "version"),
            ({"mode": "fast", "version": 2, "step": 2.3, "tag": "v1-ok"}, "step"),
            ({"mode": "fast", "version": 2, "step": 10.0, "tag": "v1-ok"}, "step"),
            ({"mode": "fast", "version": 2, "step": 2.5, "tag": "ab"}, "tag"),
            ({"mode": "fast", "version": 2, "step": 2.5, "tag": "BAD!"}, "tag"),
            ({"mode": "fast", "version": 2, "step": 2.5, "tag": "v1-ok", "target": 5}, "target"),  # matches both oneOf!
        ]
        for args_payload, expected_param in bad_cases:
            _, _, err = SafeArgumentNormalizer.normalize_and_validate(
                "constrained", schema, args_payload, argument_source="model"
            )
            assert err is not None
            assert err["invalid_parameter"] == expected_param

    def test_compact_schema_generator_compatibility(self):
        from zelyr.engine.normalizer import SafeArgumentNormalizer
        from zelyr.engine.schema_compact import CompactSchemaGenerator
        from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

        class _SampleBoundedTool(BaseTool):
            def __init__(self):
                super().__init__("bounded_op", "bounded_op", "Sample bounded operation")
                self.add_parameter(ToolParameter("count", "integer", "Item count", required=True))
                self.add_parameter(ToolParameter("mode", "string", "Execution mode", required=False))

            def execute(self, count: int, mode: str = "dry") -> ToolResult:
                return ToolResult(True, output={"count": count, "mode": mode})

        raw_schema = {
            "type": "function",
            "function": {
                "name": "bounded_op",
                "description": "Sample bounded operation",
                "parameters": _SampleBoundedTool().schema,
            },
        }
        raw_schema["function"]["parameters"]["properties"]["mode"]["enum"] = ["dry", "live"]
        compact_tool = CompactSchemaGenerator.compact_tool_schema(raw_schema)
        norm_ok, _, err_ok = SafeArgumentNormalizer.normalize_and_validate(
            "bounded_op", compact_tool, '{"count": "7", "mode": "dry"}', argument_source="model"
        )
        assert err_ok is None
        assert norm_ok == {"count": 7, "mode": "dry"}

        _, _, err_bad = SafeArgumentNormalizer.normalize_and_validate(
            "bounded_op", compact_tool, {"count": 7, "mode": "invalid", "bogus": 1}, argument_source="model"
        )
        assert err_bad is not None
        assert err_bad["error_type"] == "SCHEMA_VALIDATION_ERROR"


class TestArgumentSecurityAndRedaction:
    """Proves secret redaction and fail-closed security boundaries across execution paths."""

    def test_redacts_secrets_tokens_and_sensitive_env_values_with_bounded_previews(self, monkeypatch):
        from zelyr.engine.normalizer import SafeArgumentNormalizer

        secret_env = "zelyr_super_secret_env_token_987654"
        monkeypatch.setenv("ZELYR_TEST_API_SECRET", secret_env)
        # Ensure normal path env vars do not trigger unintended scrubbing
        monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/ssh-safe-socket-path")

        schema = {
            "type": "object",
            "properties": {
                "api_key": {"type": "integer"},
                "payload": {"type": "integer"},
                "safe_path": {"type": "string", "minLength": 100},
            },
            "additionalProperties": False,
        }

        _, _, err_key = SafeArgumentNormalizer.normalize_and_validate(
            "auth_tool",
            schema,
            {"api_key": "plaintext-secret-value-not-leaked"},
            argument_source="model",
        )
        assert err_key is not None
        serialized = json.dumps(err_key)
        assert "plaintext-secret-value-not-leaked" not in serialized
        assert err_key["received_value"] == "[REDACTED]"

        long_secret_blob = (
            f"sk-AbCdEf1234567890GhIjKlMnOpQrStUv "
            f"ghp_1234567890abcdefghijklmnopqrstuvwxyz1234 "
            f"Bearer my_jwt_token_value_123 {secret_env} "
            + ("A" * 300)
        )
        _, _, err_blob = SafeArgumentNormalizer.normalize_and_validate(
            "auth_tool",
            schema,
            {"payload": long_secret_blob},
            argument_source="model",
        )
        assert err_blob is not None
        blob_json = json.dumps(err_blob)
        assert "sk-AbCdEf1234567890GhIjKlMnOpQrStUv" not in blob_json
        assert "ghp_1234567890abcdefghijklmnopqrstuvwxyz1234" not in blob_json
        assert "my_jwt_token_value_123" not in blob_json
        assert secret_env not in blob_json
        assert len(err_blob["received_value"]) <= 180

        _, _, err_sock = SafeArgumentNormalizer.normalize_and_validate(
            "auth_tool",
            schema,
            {"safe_path": "/tmp/ssh-safe-socket-path"},
            argument_source="model",
        )
        assert err_sock is not None
        assert err_sock["received_value"] == "/tmp/ssh-safe-socket-path"

    def test_rejects_reserved_auth_metadata_and_distinguishes_internal_vs_model_callers(self):
        from zelyr.tool_registry import ToolRegistry, authorize_and_validate_tool_call
        from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

        executed_calls = []

        class _SpyTool(BaseTool):
            REQUIRED_RING = 2

            def __init__(self):
                super().__init__("spy_op", "spy_op", "Spy tool")
                self.required_ring = 2
                self.add_parameter(ToolParameter("count", "integer", "Count", required=True))

            def execute(self, count: int) -> ToolResult:
                executed_calls.append(count)
                return ToolResult(True, output={"count": count})

        reg = ToolRegistry()
        reg._load_builtins = lambda: None
        reg.register(_SpyTool())

        for reserved_key, reserved_val in (
            ("caller_ring", 0),
            ("allowed_tools", ["spy_op"]),
            ("task_local_scope", {"fake": True}),
            ("auth_ctx", {"ring": 0}),
            ("auth_context", {"ring": 0}),
        ):
            for payload in (
                {"count": 1, reserved_key: reserved_val},
                json.dumps({"count": 1, reserved_key: reserved_val}),
            ):
                dec = authorize_and_validate_tool_call(
                    "spy_op",
                    payload,
                    caller_ring=2,
                    allowed_tools=["spy_op"],
                    tool_registry=reg,
                    argument_source="model",
                )
                assert dec.allowed is False
                assert dec.error_type == "AUTHORIZATION_DENIED"
                assert dec.validation_error is not None
                assert dec.validation_error["error_category"] == "security_metadata_override"
                assert dec.validation_error["recoverable"] is False

        # Malformed or schema-invalid calls also never reach tool.execute()
        bad_dec = authorize_and_validate_tool_call(
            "spy_op",
            {"count": "1.5"},
            caller_ring=2,
            allowed_tools=["spy_op"],
            tool_registry=reg,
            argument_source="model",
        )
        assert bad_dec.allowed is False
        assert bad_dec.error_type == "SCHEMA_VALIDATION_ERROR"
        assert executed_calls == []

        # Trusted internal call runs cleanly through ToolRegistry.execute with explicit caller_ring
        res_internal = reg.execute("spy_op", caller_ring=2, allowed_tools=["spy_op"], count="9")
        assert res_internal.success is True
        assert res_internal.output == {"count": 9}
        assert executed_calls == [9]


class TestRuntimeEntrypointNormalizationIntegration:
    """Verifies NativeOrchestrator, FallbackToolParser, MCP Gateway, and Spotless-Arms Tier 3 boundaries."""

    @pytest.mark.asyncio
    async def test_native_orchestrator_rejects_malformed_args_never_executes_and_recovers(self, monkeypatch):
        from unittest.mock import AsyncMock
        from zelyr.agents.native_orchestrator import NativeOrchestrator
        from zelyr.tool_registry import ToolRegistry
        from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

        exec_calls = []

        class _MathTool(BaseTool):
            def __init__(self):
                super().__init__("math_add", "math_add", "Add two integers")
                self.add_parameter(ToolParameter("a", "integer", "First", required=True))
                self.add_parameter(ToolParameter("b", "integer", "Second", required=True))

            def execute(self, a: int, b: int) -> ToolResult:
                exec_calls.append((a, b))
                return ToolResult(True, output=a + b)

        reg = ToolRegistry()
        reg._load_builtins = lambda: None
        reg.register(_MathTool())

        turn = 0
        seen_tool_messages = []
        math_tool_schema = {
            "type": "function",
            "function": {
                "name": "math_add",
                "description": "Add two integers",
                "parameters": _MathTool().schema,
            },
        }

        async def fake_stream(payload):
            nonlocal turn
            turn += 1
            for m in payload.get("messages", []):
                if m.get("role") == "tool" and m not in seen_tool_messages:
                    seen_tool_messages.append(m)
            if turn == 1:
                # 1) Reserved auth key injection attempt
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_sec",
                                        "function": {
                                            "name": "math_add",
                                            "arguments": '{"a": 1, "b": 2, "caller_ring": 0}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            elif turn == 2:
                # 2) Single-quoted invalid JSON (never parsed by eval/ast.literal_eval)
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_bad_json",
                                        "function": {
                                            "name": "math_add",
                                            "arguments": "{'a': 1, 'b': 2}",
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            elif turn == 3:
                # 3) Fractional lossy float for integer schema
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_lossy",
                                        "function": {
                                            "name": "math_add",
                                            "arguments": '{"a": 1.5, "b": 2}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            elif turn == 4:
                # 4) Valid recoverable stringified integers -> normalized to (4, 6) and executed once
                yield {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_ok",
                                        "function": {
                                            "name": "math_add",
                                            "arguments": '{"a": "4", "b": "6"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            else:
                yield {"choices": [{"delta": {"content": "Sum is 10."}}]}

        async def _dispatch(name: str, **kwargs):
            res = await reg.execute_async(name, caller_ring=0, **kwargs)
            return json.dumps(res.to_dict())

        orch = NativeOrchestrator(
            stream_provider=fake_stream,
            tool_registry=reg,
            execute_tool=_dispatch,
            caller_ring=0,
            allowed_tools=["math_add"],
            max_steps=6,
        )
        monkeypatch.setattr(orch, "get_active_window_context", AsyncMock(return_value=None))

        chunks = []
        async for chunk in orch.generate_stream(
            {"messages": [{"role": "user", "content": "Add 4 and 6"}], "tools": [math_tool_schema]},
        ):
            chunks.append(chunk)

        assert exec_calls == [(4, 6)]
        assert len(seen_tool_messages) >= 4
        sec_obs = json.loads(seen_tool_messages[0]["content"])
        bad_json_obs = json.loads(seen_tool_messages[1]["content"])
        lossy_obs = json.loads(seen_tool_messages[2]["content"])

        assert sec_obs["error_type"] == "AUTHORIZATION_DENIED"
        assert sec_obs["error_category"] == "security_metadata_override"
        assert bad_json_obs["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert bad_json_obs["error_category"] == "parse_error"
        assert lossy_obs["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert lossy_obs["error_category"] == "ambiguous_or_lossy_conversion"

    def test_fallback_tool_parser_rejects_non_json_and_reserved_auth_keys(self):
        from zelyr.agents.fallback_parser import FallbackToolParser

        single_quoted = "<tool_call>{'name': 'execute_command', 'arguments': {'command': 'whoami'}}</tool_call>"
        assert (
            FallbackToolParser.parse_tool_calls(
                single_quoted,
                known_tools={"execute_command"},
                caller_ring=0,
                allowed_tools=["execute_command"],
            )
            == []
        )

        injected = '<tool_call>{"name": "execute_command", "arguments": {"command": "whoami", "caller_ring": 0}}</tool_call>'
        assert (
            FallbackToolParser.parse_tool_calls(
                injected,
                known_tools={"execute_command"},
                caller_ring=0,
                allowed_tools=["execute_command"],
            )
            == []
        )

    @pytest.mark.asyncio
    async def test_mcp_gateway_normalizes_valid_coercions_and_rejects_reserved_or_invalid_args(self):
        import mcp.types as types
        from tests.test_mcp_server_gateway import close_connected_session, create_connected_session
        from zelyr.mcp.server import ZelyrMCPServerGateway
        from zelyr.tool_registry import ToolRegistry
        from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

        calls = []

        class _StrictCalcTool(BaseTool):
            def __init__(self):
                super().__init__("strict_calc", "strict_calc", "Strict calculator")
                self.add_parameter(ToolParameter("a", "integer", "First", required=True))
                self.add_parameter(ToolParameter("b", "integer", "Second", required=True))

            def execute(self, a: int, b: int) -> ToolResult:
                calls.append((a, b))
                return ToolResult(True, output={"sum": a + b})

        reg = ToolRegistry()
        reg._load_builtins = lambda: None
        reg.register(_StrictCalcTool())

        gateway = ZelyrMCPServerGateway(registry=reg, caller_ring=0)
        handler = gateway.server.request_handlers[types.CallToolRequest]

        # Direct gateway handler: stringified integers normalize cleanly and execute
        ok_server_res = await handler(
            types.CallToolRequest(params=types.CallToolRequestParams(name="strict_calc", arguments={"a": "10", "b": "5"}))
        )
        ok_res = ok_server_res.root
        assert ok_res.isError is False
        assert calls == [(10, 5)]

        # Lossy float -> int rejected before execute()
        bad_server_res = await handler(
            types.CallToolRequest(params=types.CallToolRequestParams(name="strict_calc", arguments={"a": 1.5, "b": 5}))
        )
        bad_res = bad_server_res.root
        assert bad_res.isError is True
        assert "InvalidArguments" in bad_res.content[0].text
        assert calls == [(10, 5)]

        # Reserved auth key override rejected fail-closed without TypeError or execution
        auth_server_res = await handler(
            types.CallToolRequest(
                params=types.CallToolRequestParams(name="strict_calc", arguments={"a": 1, "b": 2, "caller_ring": 0})
            )
        )
        auth_res = auth_server_res.root
        assert auth_res.isError is True
        assert "AuthorizationError" in auth_res.content[0].text
        assert calls == [(10, 5)]

        # Full MCP ClientSession round-trip executes cleanly for valid integers and fails closed on invalid input
        client, tg = await create_connected_session(gateway)
        try:
            await client.initialize()
            e2e_ok = await client.call_tool("strict_calc", {"a": 3, "b": 4})
            assert e2e_ok.isError is False
            assert calls == [(10, 5), (3, 4)]

            e2e_bad = await client.call_tool("strict_calc", {"a": 1, "b": 2, "caller_ring": 0})
            assert e2e_bad.isError is True
            assert calls == [(10, 5), (3, 4)]
        finally:
            await close_connected_session(client, tg)

    @pytest.mark.asyncio
    async def test_spotless_arms_interact_with_ui_boundary_unaffected(self, monkeypatch):
        from zelyr.tools.vision import InteractWithUITool, trusted_ui_execution_context

        monkeypatch.setattr("zelyr.tools.vision.is_spotless_arms_available", lambda: False)
        monkeypatch.setattr("zelyr.tool_registry.get_active_task_local_scope", lambda: None)

        tool = InteractWithUITool()
        res_injected = await tool.execute({"instruction": "click ok", "caller_ring": 0})
        assert res_injected.success is False
        assert res_injected.error_type == "AUTHORIZATION_DENIED"

        with trusted_ui_execution_context(caller_ring=1, allowed_tools=["interact_with_ui"]):
            res_ring1 = await tool.execute({"instruction": "click ok"})
        assert res_ring1.success is False
        assert res_ring1.error_type == "AUTHORIZATION_DENIED"
