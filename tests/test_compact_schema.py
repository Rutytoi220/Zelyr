"""Tests for Token-Optimized Dynamic Schema Generator for Small Contexts.

Validates schema compaction efficiency (>= 40% reduction), strict contract preservation,
JSON schema standard compliance, SafeArgumentNormalizer compatibility, and dynamic
orchestrator context/budget selection.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from zelyr.agents.native_orchestrator import NativeOrchestrator
from zelyr.engine.normalizer import SafeArgumentNormalizer, validate_arguments
from zelyr.engine.schema_compact import CompactSchemaGenerator
from zelyr.tool_registry import BUILTIN_TOOLS, ToolRegistry


@pytest.fixture
def rich_representative_schemas():
    """Fixtures representing rich tool schemas with verbose descriptions and metadata."""
    from zelyr.core.plugins import get_tool_schemas
    import copy

    raw_schemas = [
        s for s in get_tool_schemas(0)
        if s["function"]["name"] in ("interact_with_browser", "manage_workspace_file", "execute_command")
    ]
    tools = copy.deepcopy(raw_schemas)
    for t in tools:
        t["function"]["ring"] = 1
        t["function"]["tier"] = "tier1"
        t["function"]["tui_hint"] = "TUI Automation Hint"
        t["function"]["examples"] = [{"action": "default", "sample": "value"}]
        t["function"]["title"] = t["function"]["name"].replace("_", " ").title()
        t["function"]["alias"] = t["function"]["name"][:4]
    return tools


class TestCompactSchemaGeneratorUnit:
    """Unit tests verifying compaction efficiency, contract preservation, and schema structure."""

    def test_compaction_efficiency_reaches_at_least_40_percent(self, rich_representative_schemas):
        compact = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        reduction = CompactSchemaGenerator.calculate_reduction(rich_representative_schemas, compact)
        assert reduction >= 0.40, f"Expected reduction >= 40%, got {reduction:.2%}"

    def test_contract_preservation_types_enums_required(self, rich_representative_schemas):
        compact = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        for orig, comp in zip(rich_representative_schemas, compact):
            orig_fn = orig["function"]
            comp_fn = comp["function"]

            assert orig_fn["name"] == comp_fn["name"]

            orig_params = orig_fn.get("parameters", {})
            comp_params = comp_fn.get("parameters", {})

            # Validate required list exact 1:1 match
            assert orig_params.get("required") == comp_params.get("required")

            # Validate property types, enums, and defaults
            orig_props = orig_params.get("properties", {})
            comp_props = comp_params.get("properties", {})
            assert set(orig_props.keys()) == set(comp_props.keys())

            for p_name, orig_p in orig_props.items():
                comp_p = comp_props[p_name]
                assert orig_p.get("type") == comp_p.get("type")
                if "enum" in orig_p:
                    assert orig_p["enum"] == comp_p["enum"]
                if "default" in orig_p:
                    assert orig_p["default"] == comp_p["default"]

    def test_json_schema_conformance(self, rich_representative_schemas):
        compact = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        for item in compact:
            assert item.get("type") == "function"
            fn = item.get("function")
            assert isinstance(fn, dict)
            assert isinstance(fn.get("name"), str) and len(fn["name"]) > 0
            assert isinstance(fn.get("description"), str)
            params = fn.get("parameters")
            assert isinstance(params, dict)
            assert params.get("type") == "object"
            assert isinstance(params.get("properties"), dict)
            if "required" in params:
                assert isinstance(params["required"], list)

    def test_non_standard_metadata_stripped(self, rich_representative_schemas):
        compact = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        stripped_keys = CompactSchemaGenerator.METADATA_KEYS_TO_STRIP
        for item in compact:
            fn = item["function"]
            for key in stripped_keys:
                assert key not in fn, f"Function schema leaked stripped key: {key}"
            for p_name, p_def in fn.get("parameters", {}).get("properties", {}).items():
                for key in stripped_keys:
                    assert key not in p_def, f"Property {p_name} leaked stripped key: {key}"

    def test_description_condensation_rules(self):
        desc = (
            "First sentence is clear and concise. Second sentence has verbose internal reasoning. "
            "Third sentence is totally redundant."
        )
        condensed = CompactSchemaGenerator.condense_description(desc, max_len=90)
        assert condensed == "First sentence is clear and concise."
        assert len(condensed) <= 90

        # Description without period exceeding max_len
        long_desc = "A" * 120
        condensed_long = CompactSchemaGenerator.condense_description(long_desc, max_len=90)
        assert len(condensed_long) <= 90

    def test_abbreviation_safety_in_descriptions(self):
        desc = "Specify the output format (e.g. json, yaml). Ignored if raw mode is requested."
        condensed = CompactSchemaGenerator.condense_property_description(desc, max_len=60)
        assert "e.g. json, yaml" in condensed
        assert "Ignored if" not in condensed

    def test_idempotency_of_compact_schemas(self, rich_representative_schemas):
        compact1 = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        compact2 = CompactSchemaGenerator.compact_schemas(compact1)
        assert json.dumps(compact1) == json.dumps(compact2)


class TestSafeArgumentNormalizerCompatibility:
    """Tests ensuring CompactSchemaGenerator is 100% compatible with SafeArgumentNormalizer."""

    def test_validate_arguments_success_against_compact_schema(self, rich_representative_schemas):
        compact_list = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        file_tool = next(s for s in compact_list if s["function"]["name"] == "manage_workspace_file")

        raw_args = {"action": "read", "path": "test.txt"}
        norm_args, _ = SafeArgumentNormalizer.normalize(file_tool, raw_args, tool_name="manage_workspace_file")
        err = validate_arguments("manage_workspace_file", file_tool, norm_args)
        assert err is None

    def test_validate_arguments_missing_required_against_compact_schema(self, rich_representative_schemas):
        compact_list = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        file_tool = next(s for s in compact_list if s["function"]["name"] == "manage_workspace_file")

        raw_args = {}  # missing required 'action'
        norm_args, _ = SafeArgumentNormalizer.normalize(file_tool, raw_args, tool_name="manage_workspace_file")
        err = validate_arguments("manage_workspace_file", file_tool, norm_args)
        assert err is not None
        assert err["error_type"] == "SCHEMA_VALIDATION_ERROR"
        assert "parameter 'action'" in err["error"]

    def test_validate_arguments_coercion_against_compact_schema(self, rich_representative_schemas):
        compact_list = CompactSchemaGenerator.compact_schemas(rich_representative_schemas)
        file_tool = next(s for s in compact_list if s["function"]["name"] == "manage_workspace_file")

        raw_args = {"action": "read", "max_lines": "50"}  # string integer
        norm_args, _ = SafeArgumentNormalizer.normalize(file_tool, raw_args, tool_name="manage_workspace_file")
        assert norm_args["max_lines"] == 50
        err = validate_arguments("manage_workspace_file", file_tool, norm_args)
        assert err is None


class TestToolRegistrySchemaModes:
    """Tests ToolRegistry schema retrieval in full vs compact mode."""

    def test_registry_get_schemas_full_vs_compact(self):
        reg = ToolRegistry()
        for tid in BUILTIN_TOOLS:
            reg.get_tool(tid)

        full_schemas = reg.get_schemas(compact=False)
        compact_schemas = reg.get_schemas(compact=True)

        assert len(full_schemas) > 0
        assert len(full_schemas) == len(compact_schemas)

        full_len = len(json.dumps(full_schemas))
        compact_len = len(json.dumps(compact_schemas))
        assert compact_len < full_len

    def test_registry_get_tool_schemas_compact_forwarding(self):
        reg = ToolRegistry()
        for tid in BUILTIN_TOOLS:
            reg.get_tool(tid)

        compact_schemas = reg.get_tool_schemas(compact=True)
        for s in compact_schemas:
            assert s["type"] == "function"
            fn = s["function"]
            assert "ring" not in fn
            assert "tier" not in fn

    def test_lazy_loading_of_builtin_tools_when_empty(self):
        reg = ToolRegistry()
        # Verify internal list is empty initially
        assert len(reg._core_registry.list_tools()) == 0
        schemas = reg.get_schemas(compact=True)
        assert len(schemas) > 0
        assert any(s["function"]["name"] == "shell" for s in schemas)


class TestNativeOrchestratorDynamicSchemaSelection:
    """Tests dynamic schema selection and Core Sandbox Protection in NativeOrchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_dispatches_compact_schemas_on_small_context(self):
        orchestrator = NativeOrchestrator()
        payload = {
            "model": "qwen2.5-coder:7b",
            "messages": [{"role": "user", "content": "list project files"}],
            "options": {"num_ctx": 4096},
        }

        captured_payload = {}

        async def fake_stream_events(p, base_url, stream_provider=None):
            captured_payload.update(p)
            yield {"choices": [{"delta": {"content": "ok"}}]}

        orchestrator._stream_model_events = fake_stream_events

        async for _ in orchestrator.generate_stream(payload):
            pass

        assert "tools" in captured_payload
        tools = captured_payload["tools"]
        assert len(tools) > 0
        # All dispatched tools must be minified and have non-standard metadata stripped
        for t in tools:
            assert t["type"] == "function"
            fn = t["function"]
            assert "ring" not in fn
            assert "tier" not in fn
            assert "tui_hint" not in fn

    @pytest.mark.asyncio
    async def test_orchestrator_core_sandbox_protection(self):
        """Verifies execute_command and manage_workspace_file are preserved even in tier2_browser."""
        orchestrator = NativeOrchestrator()
        payload = {
            "model": "qwen2.5-coder:7b",
            "messages": [{"role": "user", "content": "open browser and navigate to google.com"}],
            "options": {"num_ctx": 4096},
        }

        captured_payload = {}

        async def fake_stream_events(p, base_url, stream_provider=None):
            captured_payload.update(p)
            yield {"choices": [{"delta": {"content": "navigating"}}]}

        orchestrator._stream_model_events = fake_stream_events

        async for _ in orchestrator.generate_stream(payload):
            pass

        tools = captured_payload["tools"]
        tool_names = {t["function"]["name"] for t in tools}
        assert "execute_command" in tool_names, "Core sandbox protection failed: execute_command missing"
        assert "manage_workspace_file" in tool_names, "Core sandbox protection failed: manage_workspace_file missing"

    @pytest.mark.asyncio
    async def test_orchestrator_honors_disable_tools(self):
        orchestrator = NativeOrchestrator()
        payload = {
            "model": "qwen2.5-coder:7b",
            "messages": [{"role": "user", "content": "just say hello"}],
            "options": {"num_ctx": 4096},
        }

        captured_payload = {}

        async def fake_stream_events(p, base_url, stream_provider=None):
            captured_payload.update(p)
            yield {"choices": [{"delta": {"content": "hello"}}]}

        orchestrator._stream_model_events = fake_stream_events

        async for _ in orchestrator.generate_stream(payload, disable_tools=True):
            pass

        assert captured_payload["tools"] == []
