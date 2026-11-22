"""Safe Argument Normalization and Schema Validation Engine for Zelyr Runtime.

Performs deterministic, non-speculative argument type coercions and strict
JSON schema validation. Never guesses missing required arguments, invents
parameters, executes arbitrary expressions, or converts arbitrary prose into
tool arguments.
"""

from __future__ import annotations

import copy
import json
import math
import os
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

# Security metadata fields owned exclusively by runtime authorization contexts.
# Untrusted model/MCP arguments must never supply or override these keys.
RESERVED_AUTHORIZATION_KEYS: frozenset[str] = frozenset({
    "caller_ring",
    "allowed_tools",
    "task_local_scope",
    "auth_ctx",
    "auth_context",
})

MAX_ERROR_VALUE_LENGTH = 160

_PRESERVE_WHITESPACE_STRING_KEYS: frozenset[str] = frozenset({
    "content",
    "text",
    "code",
    "source",
    "source_code",
    "script",
    "body",
    "patch",
    "diff",
    "search_text",
    "replace_text",
    "old_text",
    "new_text",
    "message",
    "command",
})

_SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset({
    "token",
    "secret",
    "password",
    "passwd",
    "pwd",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "auth_token",
    "bearer_token",
    "private_key",
    "secret_key",
    "signing_key",
    "encryption_key",
    "credential",
    "credentials",
    "authorization",
    "cookie",
    "cookies",
    "session_id",
})

_SENSITIVE_FIELD_SUFFIXES: tuple[str, ...] = (
    "_token",
    "_secret",
    "_password",
    "_passwd",
    "_api_key",
    "_apikey",
    "_private_key",
    "_secret_key",
    "_credential",
    "_credentials",
)

_ENV_SECRET_KEY_MARKERS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "CREDENTIAL",
    "PRIVATE",
)

_BENIGN_ENV_VALUES: frozenset[str] = frozenset({
    "true",
    "false",
    "1",
    "0",
    "yes",
    "no",
    "none",
    "null",
    "localhost",
    "127.0.0.1",
    "utf-8",
    "utf8",
    "posix",
    "linux",
})

_TEXT_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_\-]{12,}"), "[REDACTED:api_key]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"), "[REDACTED:github_token]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws_access_key]"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}(?:\.[A-Za-z0-9_\-]+)?"), "[REDACTED:jwt]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED:private_key]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|password|passwd|secret|token)\s*([=:])\s*([^\s,;'\"\]\}]+)"), r"\1\2[REDACTED]"),
)

# Strict RFC-8259 numeric patterns (reject leading zeros such as "010", hex, underscores, Inf/NaN).
_STRICT_INT_RE = re.compile(r"-?(?:0|[1-9]\d*)")
_STRICT_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_NUMERIC_LIKE_STR_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$|^0[xXoObB][0-9a-fA-F_]+$|^[0-9_]+$")
_NON_FINITE_STR_VALUES: frozenset[str] = frozenset({"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})


def _reject_non_finite_constant(const: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {const!r} is not allowed.")


def strict_json_loads(raw: str) -> Any:
    """Parse strict RFC-8259 JSON without eval/ast.literal_eval or NaN/Infinity."""
    return json.loads(raw, parse_constant=_reject_non_finite_constant)


def _is_sensitive_param_path(param_path: str) -> bool:
    """Return True if any path segment names a sensitive credential field."""
    if not param_path:
        return False
    segments = [seg.split("[", 1)[0].strip().lower() for seg in param_path.split(".") if seg]
    for seg in segments:
        if seg in _SENSITIVE_FIELD_NAMES or seg.endswith(_SENSITIVE_FIELD_SUFFIXES):
            return True
    return False


def _redact_sensitive_text(text: str) -> str:
    """Scrub secret tokens, credentials, and sensitive env values from diagnostic text."""
    result = str(text)
    for pattern, replacement in _TEXT_SECRET_PATTERNS:
        result = pattern.sub(replacement, result)

    # Scrub non-trivial secret values currently present in environment variables.
    env_secrets: List[str] = []
    _NON_SECRET_ENV_SUFFIXES = (
        "_SOCK",
        "_SOCKET",
        "_DIR",
        "_DIRECTORY",
        "_PATH",
        "_FILE",
        "_PID",
        "_UID",
        "_GID",
        "_PORT",
        "_HOST",
        "_URL",
        "_URI",
    )
    for k, v in os.environ.items():
        if not v or len(v) < 6:
            continue
        k_up = k.upper()
        if k_up == "XAUTHORITY" or k_up.endswith(_NON_SECRET_ENV_SUFFIXES):
            continue
        if any(marker in k_up for marker in _ENV_SECRET_KEY_MARKERS):
            if v.strip().lower() not in _BENIGN_ENV_VALUES:
                env_secrets.append(v)
    if env_secrets:
        env_secrets.sort(key=len, reverse=True)
        for secret_val in env_secrets:
            if secret_val in result:
                result = result.replace(secret_val, "[REDACTED:env_secret]")

    return result


def _value_category(val: Any) -> str:
    """Return a concise type/shape category label without exposing raw contents."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, int):
        return "integer"
    if isinstance(val, float):
        return "number" if math.isfinite(val) else "non_finite_float"
    if isinstance(val, str):
        return "empty_string" if len(val) == 0 else "string"
    if isinstance(val, dict):
        return "empty_object" if len(val) == 0 else "object"
    if isinstance(val, list):
        return "empty_array" if len(val) == 0 else "array"
    return type(val).__name__


def _format_safe_received_value(
    param_path: str,
    val: Any,
    *,
    max_len: int = MAX_ERROR_VALUE_LENGTH,
) -> Tuple[str, str]:
    """Return a secret-redacted, length-bounded preview plus value category."""
    cat = _value_category(val)
    if val is None:
        return "null", cat
    if _is_sensitive_param_path(param_path):
        return "[REDACTED]", cat
    if isinstance(val, str):
        raw = val
    else:
        try:
            raw = repr(val)
        except Exception:
            raw = f"<{type(val).__name__}>"
    scrubbed = _redact_sensitive_text(raw)
    if len(scrubbed) > max_len:
        scrubbed = scrubbed[:max_len] + "...[truncated]"
    return scrubbed, cat


def _strict_json_loads_object(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse a JSON string strictly as a dict; reject empty, invalid, non-finite, or non-object JSON."""
    trimmed = raw.strip()
    if not trimmed:
        return None, {
            "error_category": "parse_error",
            "actual_type": "str",
            "value_category": "empty_string",
            "message": "Arguments string is empty and cannot be parsed as a JSON object.",
            "remedy_hint": "Supply tool arguments as a valid JSON object (e.g. '{}').",
        }
    try:
        parsed = strict_json_loads(trimmed)
    except Exception as exc:
        safe_exc = _redact_sensitive_text(str(exc))
        if len(safe_exc) > MAX_ERROR_VALUE_LENGTH:
            safe_exc = safe_exc[:MAX_ERROR_VALUE_LENGTH] + "...[truncated]"
        return None, {
            "error_category": "parse_error",
            "actual_type": "str",
            "value_category": "malformed_json",
            "message": f"Invalid JSON tool arguments: {safe_exc}",
            "remedy_hint": "Fix JSON syntax (use double-quoted keys/strings, no trailing commas or non-finite constants) and retry.",
        }
    if not isinstance(parsed, dict):
        return None, {
            "error_category": "invalid_top_level_type",
            "actual_type": type(parsed).__name__,
            "value_category": _value_category(parsed),
            "message": f"Tool arguments must decode to a JSON object/dictionary, received JSON {type(parsed).__name__}.",
            "remedy_hint": "Wrap named parameters in a top-level JSON object {...}.",
        }
    return parsed, None


def find_reserved_authorization_key(
    arguments: Any,
    tool_schema: Optional[Dict[str, Any]] = None,
    *,
    prefix: str = "",
) -> Optional[str]:
    """Return dotted path if untrusted arguments contain a reserved runtime authorization key."""
    if not isinstance(arguments, dict):
        return None
    properties, _ = extract_properties_and_required(tool_schema) if tool_schema else ({}, [])
    for key, val in arguments.items():
        if not isinstance(key, str):
            continue
        qkey = f"{prefix}.{key}" if prefix else key
        # Top-level reserved keys are always rejected from untrusted model payloads.
        # Nested keys are rejected unless explicitly declared as an ordinary object property.
        if key in RESERVED_AUTHORIZATION_KEYS and (not prefix or key not in properties):
            return qkey
        if isinstance(val, dict):
            sub_schema = properties.get(key) if isinstance(properties.get(key), dict) else None
            # Recurse into un-schematized / nested object dicts unless sub_schema explicitly declares a freeform dict value?
            # Even if declared, check nested object property schemas cleanly.
            sub_hit = find_reserved_authorization_key(val, sub_schema, prefix=qkey)
            if sub_hit is not None:
                return sub_hit
        elif isinstance(val, list):
            item_schema = None
            if isinstance(properties.get(key), dict) and isinstance(properties[key].get("items"), dict):
                item_schema = properties[key]["items"]
            for idx, item in enumerate(val):
                if isinstance(item, dict):
                    sub_hit = find_reserved_authorization_key(item, item_schema, prefix=f"{qkey}[{idx}]")
                    if sub_hit is not None:
                        return sub_hit
    return None


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

    if (
        "properties" in tool_schema
        or "required" in tool_schema
        or "additionalProperties" in tool_schema
        or "minProperties" in tool_schema
        or "maxProperties" in tool_schema
        or "allOf" in tool_schema
        or "anyOf" in tool_schema
        or "oneOf" in tool_schema
    ):
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


def _effective_prop_schema(prop_def: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve an unambiguous single non-null subschema (handling nullable anyOf/oneOf unions)."""
    if not isinstance(prop_def, dict):
        return None
    if "type" in prop_def:
        if _primary_type(prop_def.get("type")) is not None:
            return prop_def
        return None
    if (
        isinstance(prop_def.get("properties"), dict)
        or isinstance(prop_def.get("additionalProperties"), (dict, bool))
        or isinstance(prop_def.get("required"), list)
    ):
        merged = dict(prop_def)
        merged["type"] = "object"
        return merged
    for combinator in ("anyOf", "oneOf"):
        branches = prop_def.get(combinator)
        if isinstance(branches, list) and branches:
            non_null = [
                b for b in branches
                if isinstance(b, dict) and b.get("type") != "null"
            ]
            if len(non_null) == 1:
                eff = dict(prop_def)
                eff.update(non_null[0])
                if _primary_type(eff.get("type")) is not None:
                    return eff
                if isinstance(eff.get("properties"), dict) or isinstance(eff.get("additionalProperties"), (dict, bool)):
                    eff["type"] = "object"
                    return eff
            return None
    if isinstance(prop_def.get("allOf"), list) and len(prop_def["allOf"]) == 1:
        only = prop_def["allOf"][0]
        if isinstance(only, dict):
            eff = dict(prop_def)
            eff.update(only)
            return _effective_prop_schema(eff)
    return None


def _is_null_allowed(prop_def: Dict[str, Any], *, argument_source: str = "model") -> bool:
    """Return True when null is permitted (strict schema permission for model inputs)."""
    if not isinstance(prop_def, dict):
        return True
    if prop_def.get("nullable") is False:
        return False
    if prop_def.get("nullable") is True:
        return True
    expected_type = prop_def.get("type")
    if expected_type == "null":
        return True
    if isinstance(expected_type, list) and "null" in expected_type:
        return True
    for combinator in ("anyOf", "oneOf"):
        branches = prop_def.get(combinator)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict) and _is_null_allowed(branch, argument_source=argument_source):
                    return True
    all_of = prop_def.get("allOf")
    if isinstance(all_of, list) and all_of:
        if all(isinstance(branch, dict) and _is_null_allowed(branch, argument_source=argument_source) for branch in all_of):
            return True
    if argument_source == "internal":
        return True
    return False


def _matches_json_type(val: Any, expected_type: str) -> bool:
    if expected_type == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if expected_type == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))
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


def _expected_label(expected_type: Any, prop_def: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(expected_type, list) and expected_type:
        return "|".join(str(t) for t in expected_type)
    if expected_type:
        return str(expected_type)
    if isinstance(prop_def, dict):
        for comb in ("oneOf", "anyOf", "allOf"):
            if isinstance(prop_def.get(comb), list):
                opts = [
                    str(b.get("type"))
                    for b in prop_def[comb]
                    if isinstance(b, dict) and b.get("type")
                ]
                if opts:
                    return f"{comb}({'|'.join(opts)})"
    return "any"


def _classify_mismatch_category(val: Any, expected_type: Any) -> str:
    """Distinguish ambiguous/lossy conversions from plain type mismatches."""
    primary = _primary_type(expected_type)
    if isinstance(val, float) and not math.isfinite(val):
        return "ambiguous_or_lossy_conversion"
    if primary == "integer":
        if isinstance(val, float):
            return "ambiguous_or_lossy_conversion"
        if isinstance(val, str):
            s_val = val.strip()
            if s_val.lower() in _NON_FINITE_STR_VALUES or _NUMERIC_LIKE_STR_RE.fullmatch(s_val):
                return "ambiguous_or_lossy_conversion"
    elif primary == "number":
        if isinstance(val, str):
            s_val = val.strip()
            if s_val.lower() in _NON_FINITE_STR_VALUES or _NUMERIC_LIKE_STR_RE.fullmatch(s_val):
                return "ambiguous_or_lossy_conversion"
    elif primary == "boolean":
        if isinstance(val, str) and val.strip().lower() in ("maybe", "on", "off", "y", "n", "2", "-1"):
            return "ambiguous_or_lossy_conversion"
        if isinstance(val, float):
            return "ambiguous_or_lossy_conversion"
    return "type_mismatch"


def _schema_error(
    tool_name: str,
    param: str,
    expected_type: Any,
    received_value: str,
    message: str,
    remedy_hint: Optional[str] = None,
    *,
    error_category: str = "constraint_violation",
    actual_type: str = "unknown",
    value_category: str = "unknown",
    recoverable: bool = True,
    error_type: str = "SCHEMA_VALIDATION_ERROR",
) -> Dict[str, Any]:
    exp_label = "|".join(expected_type) if isinstance(expected_type, list) else str(expected_type or "any")
    safe_message = _redact_sensitive_text(message)
    safe_received = _redact_sensitive_text(str(received_value))
    if _is_sensitive_param_path(param) and safe_received not in ("missing", "empty", "null"):
        safe_received = "[REDACTED]"
    if len(safe_received) > MAX_ERROR_VALUE_LENGTH:
        safe_received = safe_received[:MAX_ERROR_VALUE_LENGTH] + "...[truncated]"
    safe_hint = _redact_sensitive_text(
        remedy_hint or f"Provide parameter '{param}' as a valid {exp_label} and retry."
    )
    return {
        "tool": tool_name,
        "tool_id": tool_name,
        "status": "error",
        "error_type": error_type,
        "error_category": error_category,
        "error": safe_message,
        "invalid_parameter": param,
        "argument_path": param,
        "expected_type": exp_label,
        "expected": exp_label,
        "received_value": safe_received,
        "actual_type": actual_type,
        "value_category": value_category,
        "recoverable": recoverable,
        "retryable": recoverable,
        "remedy_hint": safe_hint,
    }


def _security_override_error(tool_name: str, key_path: str) -> Dict[str, Any]:
    """Build a fail-closed, non-recoverable authorization error for injected auth metadata."""
    msg = (
        f"AuthorizationError: Reserved security metadata key '{key_path}' cannot be "
        "supplied by model-generated tool arguments."
    )
    return {
        "tool": tool_name,
        "tool_id": tool_name,
        "status": "error",
        "error_type": "AUTHORIZATION_DENIED",
        "error_category": "security_metadata_override",
        "error": msg,
        "invalid_parameter": key_path,
        "argument_path": key_path,
        "expected_type": "forbidden_reserved_metadata",
        "expected": "forbidden_reserved_metadata",
        "received_value": "[REDACTED]",
        "actual_type": "reserved_key",
        "value_category": "reserved_security_metadata",
        "recoverable": False,
        "retryable": False,
        "remedy_hint": f"Remove reserved security field '{key_path}' from tool arguments.",
    }


class SafeArgumentNormalizer:
    """Deterministic, non-speculative tool argument normalizer."""

    @classmethod
    def _normalize_value(cls, key: str, val: Any, prop_def: Dict[str, Any], warnings: List[str]) -> Any:
        eff_prop = _effective_prop_schema(prop_def)
        if val is None or eff_prop is None:
            return val

        expected_type = _primary_type(eff_prop.get("type"))
        if not expected_type:
            return val

        if expected_type == "integer":
            if isinstance(val, str):
                s_val = val.strip()
                if _STRICT_INT_RE.fullmatch(s_val):
                    try:
                        coerced = int(s_val)
                        warnings.append(f"Coerced '{key}' from string to integer {coerced}.")
                        return coerced
                    except ValueError:
                        return val
            elif isinstance(val, float) and not isinstance(val, bool) and math.isfinite(val) and val.is_integer():
                coerced = int(val)
                warnings.append(f"Coerced '{key}' from float {val} to integer {coerced}.")
                return coerced
            return val

        if expected_type == "number":
            if isinstance(val, str):
                s_val = val.strip()
                if _STRICT_NUMBER_RE.fullmatch(s_val):
                    try:
                        coerced = float(s_val)
                        if math.isfinite(coerced):
                            warnings.append(f"Coerced '{key}' from string to float {coerced}.")
                            return coerced
                    except ValueError:
                        return val
            return val

        if expected_type == "boolean":
            if isinstance(val, str):
                s_val = val.strip().lower()
                if s_val in ("true", "1", "yes", "t"):
                    warnings.append(f"Coerced '{key}' from string '{s_val}' to boolean True.")
                    return True
                if s_val in ("false", "0", "no", "f"):
                    warnings.append(f"Coerced '{key}' from string '{s_val}' to boolean False.")
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
                        parsed = strict_json_loads(trimmed)
                        if (expected_type == "object" and isinstance(parsed, dict)) or (
                            expected_type == "array" and isinstance(parsed, list)
                        ):
                            warnings.append(f"Deserialized '{key}' from JSON string into native {expected_type}.")
                            val = parsed
                    except Exception:
                        return val
            if expected_type == "object" and isinstance(val, dict):
                nested = copy.deepcopy(val)
                sub_props = eff_prop.get("properties") if isinstance(eff_prop.get("properties"), dict) else {}
                add_props_schema = eff_prop.get("additionalProperties") if isinstance(eff_prop.get("additionalProperties"), dict) else None
                for sub_k in list(nested.keys()):
                    sub_def = sub_props.get(sub_k) if isinstance(sub_props.get(sub_k), dict) else add_props_schema
                    if isinstance(sub_def, dict):
                        nested[sub_k] = cls._normalize_value(f"{key}.{sub_k}", nested[sub_k], sub_def, warnings)
                return nested
            if expected_type == "array" and isinstance(val, list) and isinstance(eff_prop.get("items"), dict):
                item_def = eff_prop["items"]
                return [cls._normalize_value(f"{key}[{idx}]", item, item_def, warnings) for idx, item in enumerate(val)]
            return val

        if expected_type == "string" and isinstance(val, str):
            leaf_key = key.rsplit(".", 1)[-1].split("[", 1)[0]
            if leaf_key in _PRESERVE_WHITESPACE_STRING_KEYS:
                return val
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
            parsed_args, parse_err = _strict_json_loads_object(arguments)
            if parse_err is not None or parsed_args is None:
                return {}, ["Arguments could not be parsed as a dictionary."]
            arguments = parsed_args

        if not isinstance(arguments, dict):
            return {}, ["Arguments could not be parsed as a dictionary."]

        norm_args: Dict[str, Any] = copy.deepcopy(arguments)
        warnings: List[str] = []

        # 1. Map approved canonical aliases if tool_name is recognized
        if tool_name and tool_name in APPROVED_CANONICAL_ALIASES:
            for old_key, canonical_key in APPROVED_CANONICAL_ALIASES[tool_name].items():
                if old_key in norm_args and canonical_key not in norm_args:
                    norm_args[canonical_key] = norm_args.pop(old_key)
                    warnings.append(f"Mapped approved alias '{old_key}' to '{canonical_key}'.")

        params_schema = extract_parameter_schema(tool_schema)
        properties, _ = extract_properties_and_required(tool_schema)
        add_props_schema = (
            params_schema.get("additionalProperties")
            if isinstance(params_schema.get("additionalProperties"), dict)
            else None
        )
        if not properties and add_props_schema is None:
            return norm_args, warnings if warnings else None

        # 2. Perform deterministic, non-speculative type coercions
        for key in list(norm_args.keys()):
            prop_def = properties.get(key) if isinstance(properties.get(key), dict) else add_props_schema
            if isinstance(prop_def, dict):
                norm_args[key] = cls._normalize_value(key, norm_args[key], prop_def, warnings)

        return norm_args, warnings if warnings else None

    @classmethod
    def normalize_and_validate(
        cls,
        tool_name: str,
        tool_schema: Optional[Dict[str, Any]],
        arguments: Any,
        *,
        argument_source: str = "model",
    ) -> Tuple[Dict[str, Any], List[str], Optional[Dict[str, Any]]]:
        """Normalize and validate arguments atomically with fail-closed security boundary checks.

        Returns:
            Tuple of (normalized_arguments_dict, warnings_list, error_envelope_or_None).
        """
        eff_name = tool_name or "unknown"
        if argument_source not in ("model", "internal"):
            raise ValueError(f"Unsupported argument_source {argument_source!r}; expected 'model' or 'internal'.")

        working_args: Any = arguments
        if isinstance(working_args, str):
            parsed_args, parse_err = _strict_json_loads_object(working_args)
            if parse_err is not None or parsed_args is None:
                preview, _ = _format_safe_received_value("arguments", working_args)
                return {}, [], _schema_error(
                    eff_name,
                    "arguments",
                    "object",
                    preview,
                    parse_err["message"] if parse_err else "Arguments could not be parsed as a JSON object.",
                    remedy_hint=parse_err["remedy_hint"] if parse_err else None,
                    error_category=parse_err["error_category"] if parse_err else "parse_error",
                    actual_type=parse_err["actual_type"] if parse_err else "str",
                    value_category=parse_err["value_category"] if parse_err else "malformed_json",
                    recoverable=True,
                )
            working_args = parsed_args
        elif not isinstance(working_args, dict):
            preview, vcat = _format_safe_received_value("arguments", working_args)
            return {}, [], _schema_error(
                eff_name,
                "arguments",
                "object",
                preview,
                f"Arguments for '{eff_name}' must be a JSON object/dictionary, received {type(working_args).__name__}.",
                remedy_hint="Supply arguments as a JSON object mapping parameter names to values.",
                error_category="invalid_top_level_type",
                actual_type=type(working_args).__name__,
                value_category=vcat,
                recoverable=True,
            )

        # Reject model-originated reserved authorization keys before normalization or validation.
        if argument_source == "model":
            reserved_key = find_reserved_authorization_key(working_args, tool_schema)
            if reserved_key is not None:
                return {}, [], _security_override_error(eff_name, reserved_key)

        norm_args, warnings_opt = cls.normalize(tool_schema, working_args, tool_name=eff_name)
        warnings = warnings_opt or []

        if tool_schema is not None:
            val_err = validate_arguments(eff_name, tool_schema, norm_args, argument_source=argument_source)
            if val_err is not None:
                return norm_args, warnings, val_err

        return norm_args, warnings, None


def _validate_object_schema(
    tool_name: str,
    schema_obj: Dict[str, Any],
    arguments: Dict[str, Any],
    prefix: str = "",
    *,
    argument_source: str = "model",
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
                f"Missing required parameter '{qname}'. Expected {_expected_label(expected_type, prop_def)}.",
                error_category="missing_required_field",
                actual_type="missing",
                value_category="missing",
            )
        val = arguments[req_param]
        if val is None or (isinstance(val, (str, list, dict)) and len(val) == 0):
            _, vcat = _format_safe_received_value(qname, val)
            return _schema_error(
                tool_name,
                qname,
                expected_type,
                "empty" if val is not None else "null",
                f"Required parameter '{qname}' cannot be empty.",
                error_category="missing_required_field" if val is None else "constraint_violation",
                actual_type="null" if val is None else type(val).__name__,
                value_category=vcat,
            )

    # 2. Reject unknown properties when additionalProperties is False
    if add_props is False:
        for param_name in arguments:
            if param_name not in properties:
                qname = _qual(param_name)
                preview, vcat = _format_safe_received_value(qname, arguments[param_name])
                return _schema_error(
                    tool_name,
                    qname,
                    "none (unknown property)",
                    preview,
                    f"Unknown parameter '{qname}' is not allowed by schema (additionalProperties=False).",
                    remedy_hint=f"Remove unknown parameter '{qname}' and only supply declared parameters.",
                    error_category="unknown_parameter",
                    actual_type=type(arguments[param_name]).__name__,
                    value_category=vcat,
                )

    # 3. Object property count bounds
    if "minProperties" in schema_obj and isinstance(schema_obj["minProperties"], int):
        if len(arguments) < schema_obj["minProperties"]:
            obj_name = prefix or "arguments"
            preview, vcat = _format_safe_received_value(obj_name, arguments)
            return _schema_error(
                tool_name,
                obj_name,
                "object",
                preview,
                f"Parameter '{obj_name}' must contain at least {schema_obj['minProperties']} properties.",
                error_category="constraint_violation",
                actual_type="dict",
                value_category=vcat,
            )
    if "maxProperties" in schema_obj and isinstance(schema_obj["maxProperties"], int):
        if len(arguments) > schema_obj["maxProperties"]:
            obj_name = prefix or "arguments"
            preview, vcat = _format_safe_received_value(obj_name, arguments)
            return _schema_error(
                tool_name,
                obj_name,
                "object",
                preview,
                f"Parameter '{obj_name}' must contain at most {schema_obj['maxProperties']} properties.",
                error_category="constraint_violation",
                actual_type="dict",
                value_category=vcat,
            )

    # 4. Validate declared properties and additionalProperties dict schema if present
    for param_name, val in arguments.items():
        if param_name in properties and isinstance(properties[param_name], dict):
            qname = _qual(param_name)
            err = _validate_value(tool_name, qname, val, properties[param_name], argument_source=argument_source)
            if err is not None:
                return err
        elif isinstance(add_props, dict):
            qname = _qual(param_name)
            err = _validate_value(tool_name, qname, val, add_props, argument_source=argument_source)
            if err is not None:
                return err

    return None


def _validate_value(
    tool_name: str,
    param_path: str,
    val: Any,
    prop_def: Dict[str, Any],
    *,
    argument_source: str = "model",
) -> Optional[Dict[str, Any]]:
    expected_type = prop_def.get("type")
    if val is None:
        if not _is_null_allowed(prop_def, argument_source=argument_source):
            return _schema_error(
                tool_name,
                param_path,
                expected_type or "non-null",
                "null",
                f"Parameter '{param_path}' is non-nullable and cannot be null.",
                error_category="type_mismatch",
                actual_type="null",
                value_category="null",
            )
        return None

    preview, vcat = _format_safe_received_value(param_path, val)
    actual_type_name = "non_finite_float" if (isinstance(val, float) and not math.isfinite(val)) else type(val).__name__

    # Type checking (string or union list)
    if isinstance(expected_type, str):
        if not _matches_json_type(val, expected_type):
            mismatch_cat = _classify_mismatch_category(val, expected_type)
            return _schema_error(
                tool_name,
                param_path,
                expected_type,
                preview,
                f"Parameter '{param_path}' must be of type {expected_type}, received {actual_type_name}.",
                error_category=mismatch_cat,
                actual_type=actual_type_name,
                value_category=vcat,
            )
    elif isinstance(expected_type, list) and expected_type:
        if not any(_matches_json_type(val, t) for t in expected_type if isinstance(t, str)):
            mismatch_cat = _classify_mismatch_category(val, expected_type)
            return _schema_error(
                tool_name,
                param_path,
                expected_type,
                preview,
                f"Parameter '{param_path}' must match one of types {expected_type}, received {actual_type_name}.",
                error_category=mismatch_cat,
                actual_type=actual_type_name,
                value_category=vcat,
            )

    # Combinators: allOf, anyOf, oneOf
    if isinstance(prop_def.get("allOf"), list) and prop_def["allOf"]:
        for sub_schema in prop_def["allOf"]:
            if isinstance(sub_schema, dict):
                err = _validate_value(tool_name, param_path, val, sub_schema, argument_source=argument_source)
                if err is not None:
                    return err

    if isinstance(prop_def.get("anyOf"), list) and prop_def["anyOf"]:
        branch_errs = [
            _validate_value(tool_name, param_path, val, sub, argument_source=argument_source)
            for sub in prop_def["anyOf"]
            if isinstance(sub, dict)
        ]
        if branch_errs and all(e is not None for e in branch_errs):
            return _schema_error(
                tool_name,
                param_path,
                _expected_label(expected_type, prop_def),
                preview,
                f"Parameter '{param_path}' does not match any allowed schema variant.",
                error_category=_classify_mismatch_category(val, expected_type),
                actual_type=actual_type_name,
                value_category=vcat,
            )

    if isinstance(prop_def.get("oneOf"), list) and prop_def["oneOf"]:
        valid_count = 0
        for sub in prop_def["oneOf"]:
            if isinstance(sub, dict):
                if _validate_value(tool_name, param_path, val, sub, argument_source=argument_source) is None:
                    valid_count += 1
        if valid_count == 0:
            return _schema_error(
                tool_name,
                param_path,
                _expected_label(expected_type, prop_def),
                preview,
                f"Parameter '{param_path}' does not match any oneOf schema variant.",
                error_category=_classify_mismatch_category(val, expected_type),
                actual_type=actual_type_name,
                value_category=vcat,
            )
        if valid_count > 1:
            return _schema_error(
                tool_name,
                param_path,
                _expected_label(expected_type, prop_def),
                preview,
                f"Parameter '{param_path}' matched multiple mutually-exclusive oneOf schemas ({valid_count}).",
                error_category="ambiguous_or_lossy_conversion",
                actual_type=actual_type_name,
                value_category=vcat,
            )

    if "const" in prop_def and val != prop_def["const"]:
        safe_const, _ = _format_safe_received_value(param_path, prop_def["const"])
        return _schema_error(
            tool_name,
            param_path,
            prop_def.get("type", "const"),
            preview,
            f"Parameter '{param_path}' must equal constant {safe_const}, received {preview}.",
            error_category="constraint_violation",
            actual_type=actual_type_name,
            value_category=vcat,
        )

    if "enum" in prop_def and isinstance(prop_def["enum"], list) and val not in prop_def["enum"]:
        allowed_repr = repr(prop_def["enum"])
        return _schema_error(
            tool_name,
            param_path,
            prop_def.get("type", "enum"),
            preview,
            f"Parameter '{param_path}' must be one of {allowed_repr}, received {preview}.",
            remedy_hint=f"Choose an allowed value for '{param_path}' from {allowed_repr}.",
            error_category="constraint_violation",
            actual_type=actual_type_name,
            value_category=vcat,
        )

    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if not math.isfinite(float(val)):
            return _schema_error(
                tool_name,
                param_path,
                expected_type or "number",
                preview,
                f"Parameter '{param_path}' must be a finite number, received {actual_type_name}.",
                error_category="ambiguous_or_lossy_conversion",
                actual_type=actual_type_name,
                value_category=vcat,
            )
        if "minimum" in prop_def and val < prop_def["minimum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", preview,
                f"Parameter '{param_path}' must be >= {prop_def['minimum']}, received {val}.",
                error_category="constraint_violation", actual_type=actual_type_name, value_category=vcat,
            )
        if "exclusiveMinimum" in prop_def and val <= prop_def["exclusiveMinimum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", preview,
                f"Parameter '{param_path}' must be > {prop_def['exclusiveMinimum']}, received {val}.",
                error_category="constraint_violation", actual_type=actual_type_name, value_category=vcat,
            )
        if "maximum" in prop_def and val > prop_def["maximum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", preview,
                f"Parameter '{param_path}' must be <= {prop_def['maximum']}, received {val}.",
                error_category="constraint_violation", actual_type=actual_type_name, value_category=vcat,
            )
        if "exclusiveMaximum" in prop_def and val >= prop_def["exclusiveMaximum"]:
            return _schema_error(
                tool_name, param_path, expected_type or "number", preview,
                f"Parameter '{param_path}' must be < {prop_def['exclusiveMaximum']}, received {val}.",
                error_category="constraint_violation", actual_type=actual_type_name, value_category=vcat,
            )
        if "multipleOf" in prop_def and isinstance(prop_def["multipleOf"], (int, float)) and not isinstance(prop_def["multipleOf"], bool):
            step = prop_def["multipleOf"]
            if step > 0 and math.isfinite(float(step)):
                if isinstance(val, int) and isinstance(step, int):
                    is_mult = (val % step) == 0
                else:
                    ratio = float(val) / float(step)
                    is_mult = abs(ratio - round(ratio)) <= 1e-9 * max(1.0, abs(ratio))
                if not is_mult:
                    return _schema_error(
                        tool_name, param_path, expected_type or "number", preview,
                        f"Parameter '{param_path}' must be a multiple of {step}, received {val}.",
                        error_category="constraint_violation", actual_type=actual_type_name, value_category=vcat,
                    )

    if isinstance(val, str):
        if "minLength" in prop_def and len(val) < prop_def["minLength"]:
            return _schema_error(
                tool_name, param_path, "string", preview,
                f"Parameter '{param_path}' length must be >= {prop_def['minLength']}.",
                error_category="constraint_violation", actual_type="str", value_category=vcat,
            )
        if "maxLength" in prop_def and len(val) > prop_def["maxLength"]:
            return _schema_error(
                tool_name, param_path, "string", preview,
                f"Parameter '{param_path}' length must be <= {prop_def['maxLength']}.",
                error_category="constraint_violation", actual_type="str", value_category=vcat,
            )
        if "pattern" in prop_def and isinstance(prop_def["pattern"], str):
            try:
                if not re.search(prop_def["pattern"], val):
                    return _schema_error(
                        tool_name, param_path, "string", preview,
                        f"Parameter '{param_path}' does not match required pattern {prop_def['pattern']!r}.",
                        error_category="constraint_violation", actual_type="str", value_category=vcat,
                    )
            except re.error:
                pass

    if isinstance(val, list):
        if "minItems" in prop_def and len(val) < prop_def["minItems"]:
            return _schema_error(
                tool_name, param_path, "array", preview,
                f"Parameter '{param_path}' must contain at least {prop_def['minItems']} items.",
                error_category="constraint_violation", actual_type="list", value_category=vcat,
            )
        if "maxItems" in prop_def and len(val) > prop_def["maxItems"]:
            return _schema_error(
                tool_name, param_path, "array", preview,
                f"Parameter '{param_path}' must contain at most {prop_def['maxItems']} items.",
                error_category="constraint_violation", actual_type="list", value_category=vcat,
            )
        if prop_def.get("uniqueItems"):
            serialized_items = [json.dumps(x, sort_keys=True, default=str) for x in val]
            if len(set(serialized_items)) != len(serialized_items):
                return _schema_error(
                    tool_name, param_path, "array", preview,
                    f"Parameter '{param_path}' items must be unique.",
                    error_category="constraint_violation", actual_type="list", value_category=vcat,
                )
        if isinstance(prop_def.get("items"), dict):
            for idx, item in enumerate(val):
                err = _validate_value(
                    tool_name, f"{param_path}[{idx}]", item, prop_def["items"], argument_source=argument_source
                )
                if err is not None:
                    return err

    if isinstance(val, dict) and (
        isinstance(prop_def.get("properties"), dict)
        or isinstance(prop_def.get("required"), list)
        or "additionalProperties" in prop_def
        or "minProperties" in prop_def
        or "maxProperties" in prop_def
    ):
        err = _validate_object_schema(
            tool_name, prop_def, val, prefix=param_path, argument_source=argument_source
        )
        if err is not None:
            return err

    return None


def validate_arguments(
    tool_name: str,
    tool_schema: Optional[Dict[str, Any]],
    arguments: Any,
    *,
    argument_source: str = "model",
) -> Optional[Dict[str, Any]]:
    """Validate arguments against tool schema constraints and required fields.

    Returns:
        None if validation succeeds, or a structured SCHEMA_VALIDATION_ERROR dictionary.
    """
    if not isinstance(arguments, dict):
        preview, vcat = _format_safe_received_value("arguments", arguments)
        return _schema_error(
            tool_name,
            "arguments",
            "object",
            preview,
            f"Arguments for '{tool_name}' must be a JSON object/dictionary, received {type(arguments).__name__}.",
            error_category="invalid_top_level_type",
            actual_type=type(arguments).__name__,
            value_category=vcat,
        )

    param_schema = extract_parameter_schema(tool_schema)
    if not param_schema:
        return None

    return _validate_object_schema(
        tool_name, param_schema, arguments, argument_source=argument_source
    )
