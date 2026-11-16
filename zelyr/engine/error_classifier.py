"""System-wide Error Classification and Actionable Remedy Engine.

Classifies tool execution failures into domain-specific error types using
structured error envelopes first and deterministic fallbacks second, without
exposing credentials or claiming unsafe retries are safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Dict, List, Optional

_NON_RECOVERABLE_ERROR_TYPES = frozenset(
    {
        "TOOL_NOT_FOUND",
        "AUTHORIZATION_DENIED",
        "CANCELLED",
        "TOOL_UNAVAILABLE",
        "DESKTOP_UNAVAILABLE",
        "BROWSER_UNAVAILABLE",
    }
)


def _redact_secrets(text: str) -> str:
    """Scrub common credentials/tokens before surfacing error diagnostics."""
    if not text:
        return ""
    scrubbed = re.sub(r"(?i)(https?://)([^:/@\s]+):([^/@\s]+)@", r"\1\2:[REDACTED]@", text)
    scrubbed = re.sub(r"(?i)\b(bearer\s+)[a-z0-9._\-+=]+", r"\1[REDACTED]", scrubbed)
    scrubbed = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", scrubbed)
    scrubbed = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password)\s*([=:])\s*['\"]?([^\s,'\"&;]+)['\"]?",
        r"\1\2[REDACTED]",
        scrubbed,
    )
    return scrubbed


def _extract_structured_envelope(
    error_envelope: Optional[Dict[str, Any]],
    error_type: Optional[str],
    err_str: str,
    out_str: str,
) -> Dict[str, Any]:
    """Extract structured error metadata from explicit dicts or JSON payloads."""
    merged: Dict[str, Any] = {}
    for candidate in (out_str, err_str):
        stripped = candidate.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    merged.update(parsed)
            except Exception:
                pass
    if isinstance(error_envelope, dict):
        merged.update(error_envelope)
    if error_type and not merged.get("error_type"):
        merged["error_type"] = error_type
    return merged


@dataclass(frozen=True)
class ErrorClassification:
    """Classified error representation with remedy hints and allowed actions."""

    error_type: str
    summary: str
    remedy_hint: str
    allowed_actions: List[str]

    @property
    def is_recoverable(self) -> bool:
        """Return False when automatic retry of the failed action is unsafe or futile."""
        return self.error_type not in _NON_RECOVERABLE_ERROR_TYPES

    def to_dict(self) -> Dict[str, Any]:
        """Convert error classification to dictionary representation."""
        return asdict(self)


def classify_tool_error(
    tool_name: str,
    exit_code: Optional[int] = None,
    output: str = "",
    error: str = "",
    *,
    error_type: Optional[str] = None,
    error_envelope: Optional[Dict[str, Any]] = None,
) -> ErrorClassification:
    """Classify tool execution error into canonical domain error types."""
    raw_err = (error or "").strip()
    raw_out = (output or "").strip()
    envelope = _extract_structured_envelope(error_envelope, error_type, raw_err, raw_out)

    err_str = _redact_secrets(raw_err)
    out_str = _redact_secrets(raw_out)
    tname = (tool_name or "").lower().strip()

    structured_type = str(envelope.get("error_type") or "").upper().strip()
    structured_reason = str(envelope.get("reason") or "").lower().strip()
    structured_msg = _redact_secrets(
        str(envelope.get("error") or envelope.get("message") or envelope.get("detail") or "").strip()
    )
    combined = f"{tname}\n{err_str}\n{out_str}\n{structured_msg}".lower()

    # 1. Invalid tool name (must run before COMMAND_NOT_FOUND so "Tool not found" is distinct)
    if structured_type in ("TOOL_NOT_FOUND", "INVALID_TOOL_NAME", "UNKNOWN_TOOL") or (
        "tool not found:" in combined or "unknown tool" in combined
    ):
        return ErrorClassification(
            error_type="TOOL_NOT_FOUND",
            summary=f"Requested tool '{tool_name}' is not registered" if tool_name else "Requested tool is not registered",
            remedy_hint="The requested tool does not exist in the active registry. Use only registered tool names from active schemas.",
            allowed_actions=[],
        )

    # 2. Schema validation / malformed argument errors
    if (
        structured_type == "SCHEMA_VALIDATION_ERROR"
        or "schema_validation_error" in combined
        or "failed schema validation" in combined
        or "invalid_parameter" in combined
        or "missing required parameter" in combined
        or "arguments could not be parsed" in combined
    ):
        return ErrorClassification(
            error_type="SCHEMA_VALIDATION_ERROR",
            summary="Tool call arguments failed schema validation",
            remedy_hint="Tool call arguments failed schema validation. Inspect 'invalid_parameter' and 'expected_type', adjust arguments, and retry.",
            allowed_actions=[tool_name] if tool_name else ["execute_command"],
        )

    # 3. Runtime authorization / policy denial (distinct from OS file permission denied)
    if (
        structured_type in ("AUTHORIZATION_DENIED", "AUTHORIZATION_ERROR", "PROHIBITED_BY_USER")
        or structured_reason == "prohibited_by_user"
        or "authorizationerror:" in combined
        or "prohibited_by_user" in combined
        or "not in allowed tools list" in combined
        or "requires higher permissions" in combined
        or "user denied execution via the governor" in combined
    ):
        return ErrorClassification(
            error_type="AUTHORIZATION_DENIED",
            summary="Action blocked by runtime authorization or user policy",
            remedy_hint="Authorization or user policy rejected this tool call. Do not retry this blocked action; explain the restriction.",
            allowed_actions=[],
        )

    # 4. Cancellation (distinct from timeout; non-recoverable)
    if structured_type in ("CANCELLED", "TASK_CANCELLED") or any(
        k in combined for k in ("task_cancelled", "cancellederror", "execution cancelled", "operation was cancelled")
    ):
        return ErrorClassification(
            error_type="CANCELLED",
            summary="Tool execution was cancelled",
            remedy_hint="Execution was cancelled before completion. Do not retry automatically; surface the partial/cancelled state.",
            allowed_actions=[],
        )

    # 5. Desktop window manager unavailable
    if (
        structured_type == "DESKTOP_UNAVAILABLE"
        or "desktop_unavailable" in combined
        or "hyprland session not detected" in combined
        or ("hyprland" in combined and any(k in combined for k in ("not detected", "not running", "not found", "unavailable", "unreachable")))
    ):
        return ErrorClassification(
            error_type="DESKTOP_UNAVAILABLE",
            summary="Desktop window manager (Hyprland) is unavailable",
            remedy_hint="Desktop window manager (Hyprland) is not running or socket is unreachable. Use standard shell utilities or verify XDG_RUNTIME_DIR.",
            allowed_actions=["execute_command"],
        )

    # 6. Window not found
    if (
        structured_type == "WINDOW_NOT_FOUND"
        or "window_not_found" in combined
        or "target window was not found" in combined
        or "target window not found" in combined
        or "window not found" in combined
        or ("target window" in combined and "not found" in combined)
        or ("window matching" in combined and "not found" in combined)
    ):
        return ErrorClassification(
            error_type="WINDOW_NOT_FOUND",
            summary="Target window was not found",
            remedy_hint="Target window was not found in client list. Execute manage_desktop_window with action='list' to inspect open windows.",
            allowed_actions=["manage_desktop_window", "execute_command"],
        )

    # 7. Browser unavailable
    if (
        structured_type == "BROWSER_UNAVAILABLE"
        or "browser_unavailable" in combined
        or "browser remote debugging port" in combined
        or "remote debugging port (9222) unreachable" in combined
        or "cdp connection failed" in combined
        or "no browser extension connected" in combined
        or ("remote debugging" in combined and ("unreachable" in combined or "enabled" in combined or "port" in combined))
    ):
        return ErrorClassification(
            error_type="BROWSER_UNAVAILABLE",
            summary="Browser debugging port is unreachable",
            remedy_hint="Browser remote debugging port (9222) unreachable. Launch Chromium/Chrome with --remote-debugging-port=9222 or verify browser is running.",
            allowed_actions=["interact_with_browser", "execute_command"],
        )

    # 8. Element not found
    if (
        structured_type == "ELEMENT_NOT_FOUND"
        or "element_not_found" in combined
        or "target element id was not found" in combined
        or ("element id" in combined and "not found" in combined)
        or ("element not found for id" in combined)
    ):
        return ErrorClassification(
            error_type="ELEMENT_NOT_FOUND",
            summary="Target element not found on page",
            remedy_hint="Target element ID was not found on the active page. Call get_page_snapshot to refresh element IDs.",
            allowed_actions=["interact_with_browser"],
        )

    # 9. General tool unavailable / missing optional dependency / disabled plugin
    if structured_type == "TOOL_UNAVAILABLE" or any(
        k in combined for k in ("requires optional dependency", "optionaldependencyerror", "tool unavailable", "plugin disabled")
    ):
        return ErrorClassification(
            error_type="TOOL_UNAVAILABLE",
            summary="Requested tool or dependency is unavailable in this runtime",
            remedy_hint="Tool capability is unavailable in the current environment. Do not retry this tool; report the missing capability.",
            allowed_actions=[],
        )

    # 10. State verification failure
    if (
        structured_type == "VERIFICATION_FAILED"
        or tname in ("verify_outcome", "verify_state", "check_state", "assert_outcome")
        or "verification failed" in combined
        or "checksum mismatch" in combined
        or "not verified" in combined
    ):
        return ErrorClassification(
            error_type="VERIFICATION_FAILED",
            summary="Observed state did not match expected criteria",
            remedy_hint="Observed state did not match expected criteria. Inspect the observed state, correct the action, and re-verify.",
            allowed_actions=["execute_command", "manage_workspace_file", "verify_outcome"],
        )

    # 11. Timeout
    if (
        structured_type == "TIMEOUT"
        or exit_code == 124
        or "timed out" in combined
        or "timeout" in combined
    ):
        return ErrorClassification(
            error_type="TIMEOUT",
            summary="Command execution timed out",
            remedy_hint="Command timed out. Avoid interactive prompts or pagers, add non-blocking flags, or scope command.",
            allowed_actions=["execute_command"],
        )

    # 12. OS Permission Denied
    if (
        structured_type == "PERMISSION_DENIED"
        or exit_code == 126
        or "permission denied" in combined
        or "operation not permitted" in combined
    ):
        return ErrorClassification(
            error_type="PERMISSION_DENIED",
            summary="Permission denied on target resource",
            remedy_hint="Permission denied on target resource. Inspect permissions using 'ls -la'. Do NOT invoke interactive sudo.",
            allowed_actions=["execute_command", "manage_workspace_file"],
        )

    # 13. Port in use
    if (
        structured_type == "PORT_IN_USE"
        or exit_code == 98
        or "address already in use" in combined
        or "port already in use" in combined
        or "address in use" in combined
        or ("port" in combined and "already in use" in combined)
    ):
        return ErrorClassification(
            error_type="PORT_IN_USE",
            summary="Network port is occupied",
            remedy_hint="Network port is occupied. Identify the process with 'ss -tulpn' or choose an alternate port.",
            allowed_actions=["execute_command", "inspect_network", "verify_outcome"],
        )

    # 14. Path / File / Directory not found
    path_keywords = (
        "no such file",
        "file not found",
        "directory not found",
        "directory does not exist",
        "file does not exist",
        "path is not a regular file",
        "path is not a directory",
        "does not exist",
    )
    if structured_type == "PATH_NOT_FOUND" or any(k in combined for k in path_keywords):
        return ErrorClassification(
            error_type="PATH_NOT_FOUND",
            summary="Target file or directory does not exist",
            remedy_hint="Target file or directory does not exist. Verify active working directory with 'pwd' and list contents with 'ls'.",
            allowed_actions=["execute_command", "manage_workspace_file", "verify_outcome"],
        )

    # 15. Command not found
    if (
        structured_type == "COMMAND_NOT_FOUND"
        or exit_code == 127
        or "command not found" in combined
        or ("not found" in combined and not any(k in combined for k in ("element", "tab", "process", "key")))
    ):
        return ErrorClassification(
            error_type="COMMAND_NOT_FOUND",
            summary="Command not found in PATH",
            remedy_hint="Command not found in PATH. Verify command spelling, inspect installed binaries with 'which' or 'find', or use an alternative tool.",
            allowed_actions=["execute_command", "manage_workspace_file"],
        )

    # 16. Tool crafting failure
    if (
        structured_type == "TOOL_CRAFT_FAILED"
        or tname in ("craft_tool", "create_tool", "write_tool", "hotload_tool")
        or "tool_craft_failed" in combined
        or "tool crafting failed" in combined
        or "sandbox test verification failed" in combined
        or "ast security violation" in combined
        or "self_test" in combined
        or ("craft" in combined and "failed" in combined)
    ):
        return ErrorClassification(
            error_type="TOOL_CRAFT_FAILED",
            summary="Dynamic tool crafting failed validation or sandbox test",
            remedy_hint="Dynamic tool crafting failed validation or sandbox test. Check required exports (TOOL_SCHEMA, self_test) and ensure self_test() returns True.",
            allowed_actions=["craft_tool", "execute_command"],
        )

    # 17. Routine operation failure
    if (
        structured_type == "ROUTINE_ERROR"
        or tname in ("manage_routine", "schedule_task", "create_routine", "schedule_routine", "cron", "cron_job")
        or "routine_error" in combined
        or "routine operation failed" in combined
        or "invalid cron expression" in combined
        or "interval schedule must be greater than" in combined
        or "unparseable schedule expression" in combined
        or ("routine" in combined and any(k in combined for k in ("failed", "error", "not found", "invalid")))
    ):
        return ErrorClassification(
            error_type="ROUTINE_ERROR",
            summary="Routine operation or schedule evaluation failed",
            remedy_hint="Routine operation failed. Verify schedule syntax (interval in seconds or valid cron expression) and routine name.",
            allowed_actions=["manage_routine", "execute_command"],
        )

    # 18. Explicit structured execution failure
    if structured_type == "EXECUTION_FAILURE":
        return ErrorClassification(
            error_type="EXECUTION_FAILURE",
            summary="Tool execution failed",
            remedy_hint="Tool execution failed. Inspect the structured error and exit code before taking corrective action.",
            allowed_actions=[tool_name] if tool_name else ["execute_command", "manage_workspace_file"],
        )

    # 19. Fallback General / Unstructured Error
    return ErrorClassification(
        error_type="GENERAL_ERROR",
        summary="Action execution failed",
        remedy_hint="Action failed. Inspect error details before retrying blindly or choose an alternative approach.",
        allowed_actions=["execute_command", "manage_workspace_file", "verify_outcome"],
    )


class ErrorClassifier:
    """Facade for tool error classification."""

    @staticmethod
    def classify(
        tool_name: str,
        exit_code: Optional[int] = None,
        output: str = "",
        error: str = "",
        *,
        error_type: Optional[str] = None,
        error_envelope: Optional[Dict[str, Any]] = None,
    ) -> ErrorClassification:
        """Classify tool error into canonical category."""
        return classify_tool_error(
            tool_name,
            exit_code,
            output,
            error,
            error_type=error_type,
            error_envelope=error_envelope,
        )
