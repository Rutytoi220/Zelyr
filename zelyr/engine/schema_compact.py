"""Token-Optimized Dynamic Schema Generator for Small Contexts.

Provides standard-compliant, minified OpenAI function definitions that preserve
strict argument types, required lists, and enums while stripping non-standard
metadata and condensing descriptions to conserve prompt token context.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, ClassVar, Dict, List, Optional, Set


class CompactSchemaGenerator:
    """Produces token-optimized, standard-compliant OpenAI tool schemas."""

    METADATA_KEYS_TO_STRIP: ClassVar[Set[str]] = {
        "ring",
        "tier",
        "tui_hint",
        "internal",
        "examples",
        "alias",
        "category",
        "title",
        "$schema",
    }

    VALIDATION_KEYS_TO_PRESERVE: ClassVar[tuple[str, ...]] = (
        "type",
        "enum",
        "const",
        "default",
        "nullable",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "additionalProperties",
    )

    COMPOSITION_KEYS: ClassVar[tuple[str, ...]] = ("oneOf", "anyOf", "allOf")

    @classmethod
    def condense_description(cls, desc: Optional[str], max_len: int = 90) -> str:
        """Condense a tool or function description to its first sentence or max length."""
        if not desc or not isinstance(desc, str):
            return ""
        text = desc.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        first_block = lines[0]

        # Match first sentence ending with punctuation, ignoring e.g., i.e., etc.
        pattern = r"^(.*?(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?])(?:\s+[A-Z]|\s*$)"
        match = re.search(pattern, first_block)
        sentence = match.group(1).strip() if match else first_block

        if len(sentence) > max_len:
            truncated = sentence[:max_len].strip()
            if " " in truncated and len(truncated) > (max_len // 2):
                truncated = truncated.rsplit(" ", 1)[0].rstrip(" ,;:-")
            sentence = truncated
        return sentence

    @classmethod
    def condense_property_description(cls, desc: Optional[str], max_len: int = 60) -> str:
        """Condense a parameter property description to max length while retaining format hints."""
        if not desc or not isinstance(desc, str):
            return ""
        text = desc.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        first_block = lines[0]

        # Preserve format hints and sentence boundary
        pattern = r"^(.*?(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?])(?:\s+[A-Z]|\s*$)"
        match = re.search(pattern, first_block)
        sentence = match.group(1).strip() if match else first_block

        if len(sentence) > max_len:
            truncated = sentence[:max_len].strip()
            if " " in truncated and len(truncated) > (max_len // 2):
                truncated = truncated.rsplit(" ", 1)[0].rstrip(" ,;:-")
            sentence = truncated
        return sentence

    @classmethod
    def compact_property(cls, prop_def: Dict[str, Any]) -> Dict[str, Any]:
        """Minify a single parameter property definition preserving types, enums, defaults, and items."""
        if not isinstance(prop_def, dict):
            return prop_def

        compact_prop: Dict[str, Any] = {}

        # Standard schema validation keys to preserve verbatim
        for key in cls.VALIDATION_KEYS_TO_PRESERVE:
            if key in prop_def:
                if key == "additionalProperties" and isinstance(prop_def[key], dict):
                    compact_prop[key] = cls.compact_property(prop_def[key])
                else:
                    compact_prop[key] = copy.deepcopy(prop_def[key])

        for comp_key in cls.COMPOSITION_KEYS:
            if comp_key in prop_def and isinstance(prop_def[comp_key], list):
                compact_prop[comp_key] = [
                    cls.compact_property(sub) if isinstance(sub, dict) else copy.deepcopy(sub)
                    for sub in prop_def[comp_key]
                ]

        # Condense description
        if "description" in prop_def and isinstance(prop_def["description"], str):
            compact_prop["description"] = cls.condense_property_description(
                prop_def["description"], max_len=60
            )

        # Handle array items
        if "items" in prop_def:
            items_val = prop_def["items"]
            if isinstance(items_val, dict):
                compact_prop["items"] = cls.compact_property(items_val)
            else:
                compact_prop["items"] = copy.deepcopy(items_val)

        # Handle nested object properties
        if "properties" in prop_def and isinstance(prop_def["properties"], dict):
            compact_prop["properties"] = {
                k: cls.compact_property(v)
                for k, v in prop_def["properties"].items()
                if isinstance(v, dict)
            }
            if "required" in prop_def and isinstance(prop_def["required"], (list, tuple)):
                compact_prop["required"] = list(prop_def["required"])

        # Explicitly purge non-standard metadata keys
        for meta_key in cls.METADATA_KEYS_TO_STRIP:
            compact_prop.pop(meta_key, None)

        return compact_prop

    @classmethod
    def compact_tool_schema(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Produce a standard-compliant, minified OpenAI function definition."""
        if not isinstance(schema, dict):
            return schema

        # Handle wrapped {"type": "function", "function": {...}} vs bare function dict
        if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
            raw_fn = schema.get("function", {})
        else:
            raw_fn = schema

        fn_name = raw_fn.get("name", "")
        raw_desc = raw_fn.get("description", "")
        condensed_desc = cls.condense_description(raw_desc, max_len=90)

        raw_params = raw_fn.get("parameters", {})
        if not isinstance(raw_params, dict):
            raw_params = {"type": "object", "properties": {}}

        compact_params: Dict[str, Any] = {
            "type": raw_params.get("type", "object"),
            "properties": {},
        }

        raw_props = raw_params.get("properties", {})
        if isinstance(raw_props, dict):
            for p_name, p_def in raw_props.items():
                if isinstance(p_def, dict):
                    compact_params["properties"][p_name] = cls.compact_property(p_def)

        # Preserve exact required list and additionalProperties restrictions
        if "required" in raw_params and isinstance(raw_params["required"], (list, tuple)):
            compact_params["required"] = list(raw_params["required"])
        if "additionalProperties" in raw_params:
            ap = raw_params["additionalProperties"]
            compact_params["additionalProperties"] = cls.compact_property(ap) if isinstance(ap, dict) else copy.deepcopy(ap)

        compact_fn: Dict[str, Any] = {
            "name": fn_name,
            "description": condensed_desc,
            "parameters": compact_params,
        }
        for exec_key in ("requires_approval", "required_ring"):
            if exec_key in raw_fn:
                compact_fn[exec_key] = copy.deepcopy(raw_fn[exec_key])

        return {"type": "function", "function": compact_fn}

    compact_schema = compact_tool_schema

    @classmethod
    def compact_schemas(cls, schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compact a list of tool schemas."""
        if not schemas:
            return []
        return [cls.compact_tool_schema(s) for s in schemas]

    @classmethod
    def calculate_reduction(cls, original: Any, compact: Any) -> float:
        """Calculate serialized JSON character reduction ratio: 1.0 - (len(compact) / len(original))."""
        orig_len = len(json.dumps(original))
        comp_len = len(json.dumps(compact))
        if orig_len == 0:
            return 0.0
        return max(0.0, 1.0 - (comp_len / orig_len))
