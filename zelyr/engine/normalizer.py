"""Safe Argument Normalization and Schema Validation Engine for Zelyr Runtime.

Performs deterministic, non-speculative argument type coercions and strict
JSON schema validation. Never guesses missing required arguments, invents
parameters, or converts arbitrary prose into tool arguments.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

APPROVED_CANONICAL_ALIASES: Dict[str, Dict[str, str]] = {
    "execute_command": {
        "cmd": "command",
        "script": "command",
        "shell_command": "command",
        "command_line": "command",
        "input": "command",
    },
    "manage_workspace_file": {
        "file_path": "path",
        "filepath": "path",
        "file": "path",
        "filename": "path",
    },
    "verify_outcome": {
        "assertion": "check",
        "check_type": "check",
        "file_path": "target",
        "filepath": "target",
        "file": "target",
        "cmd": "target",
        "command": "target",
        "process": "target",
        "port": "target",
        "pattern": "expected",
        "sha256": "expected",
        "regex": "expected",
    },
    "manage_desktop_window": {
        "workspace_id": "target",
        "ws": "target",
        "window_id": "target",
        "address": "target",
    },
    "interact_with_browser": {
        "element": "element_id",
        "node_id": "element_id",
        "elementId": "element_id",
    },
}


def extract_parameter_schema(tool_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the parameter object schema dict across OpenAI/Ollama/JSON-Schema shapes."""
    if not isinstance(tool_schema, dict):
        return {}

    if "function" in tool_schema and isinstance(tool_schema["function"], dict):
        params = tool_schema["function"].get("parameters", {})
        if isinstance(params, dict):
            return params

    if "parameters" in tool_schema and isinstance(tool_schema["parameters"], dict):
        return tool_schema["parameters"]

    if "properties" in tool_schema or "required" in tool_schema or "additionalProperties" in tool_schema:
        return tool_schema

    return {}


def extract_properties_and_required(tool_schema: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Extract property definitions and required field list from various schema representations."""
    params = extract_parameter_schema(tool_schema)
    if not params:
        return {}, []
    props = params.get("properties", {})
    req = params.get("required", [])
    return (props if isinstance(props, dict) else {}), (list(req) if isinstance(req, list) else [])


def _primary_type(expected_type: Any) -> Optional[str]:
    """Return a single unambiguous non-null JSON type string if declared."""
    if isinstance(expected_type, str):
        return expected_type
    if isinstance(expected_type, list):
        non_null = [t for t in expected_type if isinstance(t, str) and t != "null"]
        if len(non_null) == 1:
            return non_null[0]
    return None


def _matches_json_type(val: Any, expected_type: str) -> bool:
    if expected_type == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if expected_type == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if expected_type == "boolean":
        return isinstance(val, bool)
    if expected_type == "string":
        return isinstance(val, str)
    if expected_type == "object":
        return isinstance(val, dict)
    if expected_type == "array":
        return isinstance(val, list)
    if expected_type == "null":
        return val is None
    return True


def _schema_error(
    tool_name: str,
    param: str,
    expected_type: Any,
    received_value: str,
    message: str,
    remedy_hint: Optional[str] = None,
) -> Dict[str, Any]:
    exp_label = "|".join(expected_type) if isinstance(expected_type, list) else str(expected_type or "any")
    return {
        "tool": tool_name,
        "status": "error",
        "error_type": "SCHEMA_VALIDATION_ERROR",
        "error": message,
        "invalid_parameter": param,
        "expected_type": exp_label,
        "received_value": received_value,
        "remedy_hint": remedy_hint or f"Provide parameter '{param}' as a valid {exp_label} and retry.",
    }


class SafeArgumentNormalizer:
    """Deterministic, non-speculative tool argument normalizer."""

    @classmethod
    def _normalize_value(cls, key: str, val: Any, prop_def: Dict[str, Any], warnings: List[str]) -> Any:
        expected_type = _primary_type(prop_def.get("type"))
        if val is None or not expected_type:
            return val

        if expected_type == "integer":
            if isinstance(val, str):
                s_val = val.strip()
                if re.fullmatch(r"-?\d+", s_val):
                    try:
                        coerced = int(s_val)
                        warnings.append(f"Coerced '{key}' from string '{val}' to integer {coerced}.")
                        return coerced
                    except ValueError:
                        return val
            elif isinstance(val, float) and not isinstance(val, bool) and val.is_integer():
                coerced = int(val)
                warnings.append(f"Coerced '{key}' from float {val} to integer {coerced}.")
                return coerced
            return val

        if expected_type == "number":
            if isinstance(val, str):
                s_val = val.strip()
                if re.fullmatch(r"-?\d+(?:\.\d+)?", s_val):
                    try:
                        coerced = float(s_val)
                        warnings.append(f"Coerced '{key}' from string '{val}' to float {coerced}.")
                        return coerced
                    except ValueError:
                        return val
            return val

        if expected_type == "boolean":
            if isinstance(val, str):
                s_val = val.strip().lower()
                if s_val in ("true", "1", "yes", "t"):
                    warnings.append(f"Coerced '{key}' from string '{val}' to boolean True.")
                    return True
                if s_val in ("false", "0", "no", "f"):
                    warnings.append(f"Coerced '{key}' from string '{val}' to boolean False.")
                    return False
            elif isinstance(val, int) and not isinstance(val, bool):
                if val == 1:
                    warnings.append(f"Coerced '{key}' from integer 1 to boolean True.")
                    return True
                if val == 0:
                    warnings.append(f"Coerced '{key}' from integer 0 to boolean False.")
                    return False
            return val

        if expected_type in ("object", "array"):
            if isinstance(val, str):
                trimmed = val.strip()
                if (expected_type == "object" and trimmed.startswith("{") and trimmed.endswith("}")) or (
                    expected_type == "array" and trimmed.startswith("[") and trimmed.endswith("]")
                ):
                    try:
                        parsed = json.loads(trimmed)
                        if (expected_type == "object" and isinstance(parsed, dict)) or (
                            expected_type == "array" and isinstance(parsed, list)
                        ):
                            warnings.append(f"Deserialized '{key}' from JSON string into native {expected_type}.")
                            val = parsed
                    except Exception:
                        return val
            if expected_type == "object" and isinstance(val, dict) and isinstance(prop_def.get("properties"), dict):
                nested = dict(val)
                for sub_k, sub_def in prop_def["properties"].items():
                    if sub_k in nested and isinstance(sub_def, dict):
                        nested[sub_k] = cls._normalize_value(f"{key}.{sub_k}", nested[sub_k], sub_def, warnings)
                return nested
            if expected_type == "array" and isinstance(val, list) and isinstance(prop_def.get("items"), dict):
                item_def = prop_def["items"]
                return [cls._normalize_value(f"{key}[{idx}]", item, item_def, warnings) for idx, item in enumerate(val)]
            return val

        if expected_type == "string" and isinstance(val, str):
            return val.strip()

        return val

    @classmethod
    def normalize(
        cls,
        tool_schema: Optional[Dict[str, Any]],
        arguments: Any,
        tool_name: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[List[str]]]:
        """Normalize argument values strictly against property type declarations.

        Returns:
            Tuple of (normalized_arguments_dict, warnings_list_or_None).
        """
        if isinstance(arguments, str):
            trimmed = arguments.strip()
            if not (trimmed.startswith("{") and trimmed.endswith("}")):
                return {}, ["Arguments could not be parsed as a dictionary."]
            try:
                parsed_args = json.loads(trimmed)
                if isinstance(parsed_args, dict):
                    arguments = parsed_args
                else:
                    return {}, ["Arguments could not be parsed as a dictionary."]
            except Exception:
                return {}, ["Arguments could not be parsed as a dictionary."]

        if not isinstance(arguments, dict):
            return {}, ["Arguments could not be parsed as a dictionary."]

        norm_args: Dict[str, Any] = dict(arguments)
        warnings: List[str] = []

        # 1. Map approved canonical aliases if tool_name is recognized
        if tool_name and tool_name in APPROVED_CANONICAL_ALIASES:
            for old_key, canonical_key in APPROVED_CANONICAL_ALIASES[tool_name].items():
                if old_key in norm_args and canonical_key not in norm_args:
                    norm_args[canonical_key] = norm_args.pop(old_key)
                    warnings.append(f"Mapped approved alias '{old_key}' to '{canonical_key}'.")

        properties, _ = extract_properties_and_required(tool_schema)
        if not properties:
            return norm_args, warnings if warnings else None

        # 2. Perform deterministic, non-speculative type coercions
        for key, prop_def in properties.items():
            if key in norm_args and isinstance(prop_def, dict):
                norm_args[key] = cls._normalize_value(key, norm_args[key], prop_def, warnings)

        return norm_args, warnings if warnings else None


def _validate_object_schema(
    tool_name: str,
    schema_obj: Dict[str, Any],
    arguments: Dict[str, Any],
    prefix: str = "",
) -> Optional[Dict[str, Any]]:
    properties = schema_obj.get("properties") if isinstance(schema_obj.get("properties"), dict) else {}
    required = schema_obj.get("required") if isinstance(schema_obj.get("required"), list) else []
    add_props = schema_obj.get("additionalProperties", True)

    def _qual(name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    # 1. Required fields check
    for req_param in required:
        qname = _qual(req_param)
        prop_def = properties.get(req_param, {}) if isinstance(properties.get(req_param), dict) else {}
        expected_type = prop_def.get("type", "any")
        if req_param not in arguments:
            return _schema_error(
                tool_name,
                qname,
                expected_type,
                "missing",
                f"Missing required parameter '{qname}'. Expected {expected_type}.",
            )
        val = arguments[req_param]
        if val is None or (isinstance(val, (str, list, dict)) and len(val) == 0):
            return _schema_error(
                tool_name,
                qname,
                expected_type,
                "empty" if val is not None else "null",
                f"Required parameter '{qname}' cannot be empty.",
            )

    # 2. Reject unknown properties when additionalProperties is False
    if add_props is False:
        for param_name in arguments:
            if param_name not in properties:
                qname = _qual(param_name)
                return _schema_error(
                    tool_name,
                    qname,
                    "none (unknown property)",
                    repr(arguments[param_name]),
                    f"Unknown parameter '{qname}' is not allowed by schema (additionalProperties=False).",
                    remedy_hint=f"Remove unknown parameter '{qname}' and only supply declared parameters.",
                )

    # 3. Validate declared property constraints
    for param_name, val in arguments.items():
        if param_name not in properties or not isinstance(properties[param_name], dict):
            continue
        qname = _qual(param_name)
        err = _validate_value(tool_name, qname, val, properties[param_name])
        if err is not None:
            return err

    return None


def _validate_value(
    tool_name: str,
    param_path: str,
    val: Any,
    prop_def: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    expected_type = prop_def.get("type")
    if val is None:
        if prop_def.get("nullable") is False:
            return _schema_error(
                tool_name,
                param_path,
                expected_type or "non-null",
                "null",
                f"Parameter '{param_path}' is non-nullable and cannot be null.",
            )
        return None

    # Type checking (string or union list)
    if isinstance(expected_type, str):
        if not _matches_json_type(val, expected_type):
            return _schema_error(
                tool_name,
                param_path,
                expected_type,
                str(val),
                f"Parameter '{param_path}' must be of type {expected_type}, received {type(val).__name__} ({val!r}).",
            )
    elif isinstance(expected_type, list) and expected_type:
        if not any(_matches_json_type(val, t) for t in expected_type if isinstance(t, str)):
            return _schema_error(
                tool_name,
                param_path,
                expected_type,
                str(val),
                f"Parameter '{param_path}' must match one of types {expected_type}, received {type(val).__name__} ({val!r}).",
            )

    if "const" in prop_def and val != prop_def["const"]:
        return _schema_error(
            tool_name,
            param_path,
            prop_def.get("type", "const"),
            str(val),
            f"Parameter '{param_path}' must equal constant {prop_def['const']!r}, received {val!r}.",
        )

    if "enum" in prop_def and isinstance(prop_def["enum"], list) and val not in prop_def["enum"]:
        return _schema_error(
            tool_name,
            param_path,
            prop_def.get("type", "enum"),
            str(val),
            f"Parameter '{param_path}' must be one of {prop_def['enum']!r}, received {val!r}.",
            remedy_hint=f"Choose an allowed value for '{param_path}' from {prop_def['enum']!r}.",
        )

    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if "minimum" in prop_def and val < prop_def["minimum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", str(val),
                f"Parameter '{param_path}' must be >= {prop_def['minimum']}, received {val}.",
            )
        if "exclusiveMinimum" in prop_def and val <= prop_def["exclusiveMinimum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", str(val),
                f"Parameter '{param_path}' must be > {prop_def['exclusiveMinimum']}, received {val}.",
            )
        if "maximum" in prop_def and val > prop_def["maximum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", str(val),
                f"Parameter '{param_path}' must be <= {prop_def['maximum']}, received {val}.",
            )
        if "exclusiveMaximum" in prop_def and val >= prop_def["exclusiveMaximum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", str(val),
                f"Parameter '{param_path}' must be < {prop_def['exclusiveMaximum']}, received {val}.",
            )

    if isinstance(val, str):
        if "minLength" in prop_def and len(val) < prop_def["minLength"]:
            return _schema_error(
                tool_name, param_path, "string", val,
                f"Parameter '{param_path}' length must be >= {prop_def['minLength']}.",
            )
        if "maxLength" in prop_def and len(val) > prop_def["maxLength"]:
            return _schema_error(
                tool_name, param_path, "string", val,
                f"Parameter '{param_path}' length must be <= {prop_def['maxLength']}.",
            )
        if "pattern" in prop_def and isinstance(prop_def["pattern"], str):
            try:
                if not re.search(prop_def["pattern"], val):
                    return _schema_error(
                        tool_name, param_path, "string", val,
                        f"Parameter '{param_path}' does not match required pattern {prop_def['pattern']!r}.",
                    )
            except re.error:
                pass

    if isinstance(val, list):
        if "minItems" in prop_def and len(val) < prop_def["minItems"]:
            return _schema_error(
                tool_name, param_path, "array", str(val),
                f"Parameter '{param_path}' must contain at least {prop_def['minItems']} items.",
            )
        if "maxItems" in prop_def and len(val) > prop_def["maxItems"]:
            return _schema_error(
                tool_name, param_path, "array", str(val),
                f"Parameter '{param_path}' must contain at most {prop_def['maxItems']} items.",
            )
        if prop_def.get("uniqueItems"):
            serialized_items = [json.dumps(x, sort_keys=True, default=str) for x in val]
            if len(set(serialized_items)) != len(serialized_items):
                return _schema_error(
                    tool_name, param_path, "array", str(val),
                    f"Parameter '{param_path}' items must be unique.",
                )
        if isinstance(prop_def.get("items"), dict):
            for idx, item in enumerate(val):
                err = _validate_value(tool_name, f"{param_path}[{idx}]", item, prop_def["items"])
                if err is not None:
                    return err

    if isinstance(val, dict) and (
        isinstance(prop_def.get("properties"), dict)
        or isinstance(prop_def.get("required"), list)
        or prop_def.get("additionalProperties") is False
    ):
        err = _validate_object_schema(tool_name, prop_def, val, prefix=param_path)
        if err is not None:
            return err

    return None


def validate_arguments(
    tool_name: str,
    tool_schema: Optional[Dict[str, Any]],
    arguments: Any,
) -> Optional[Dict[str, Any]]:
    """Validate arguments against tool schema constraints and required fields.

    Returns:
        None if validation succeeds, or a structured SCHEMA_VALIDATION_ERROR dictionary.
    """
    if not isinstance(arguments, dict):
        return _schema_error(
            tool_name,
            "arguments",
            "object",
            repr(arguments),
            f"Arguments for '{tool_name}' must be a JSON object/dictionary, received {type(arguments).__name__}.",
        )

    param_schema = extract_parameter_schema(tool_schema)
    if not param_schema:
        return None

    return _validate_object_schema(tool_name, param_schema, arguments)
