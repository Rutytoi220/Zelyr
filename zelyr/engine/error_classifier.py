"""System-wide Error Classification and Actionable Remedy Engine.

Classifies tool execution failures into domain-specific error types,
extracts concise summaries, and provides deterministic remediation hints
with permitted recovery tool actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ErrorClassification:
    """Classified error representation with remedy hints and allowed actions."""

    error_type: str
    summary: str
    remedy_hint: str
    allowed_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert error classification to dictionary representation."""
        return asdict(self)


def classify_tool_error(
    tool_name: str,
    exit_code: Optional[int] = None,
    output: str = "",
    error: str = "",
) -> ErrorClassification:
    """Classify tool execution error into canonical domain error types.

    Categories:
    - VERIFICATION_FAILED: verify_outcome failure or criteria mismatch.
    - TIMEOUT: exit code 124 or execution timed out.
    - PERMISSION_DENIED: exit code 126 or access permission rejected.
    - PORT_IN_USE: exit code 98 or TCP/UDP socket address already bound.
    - PATH_NOT_FOUND: exit code 1/2 or file/directory does not exist.
    - COMMAND_NOT_FOUND: exit code 127 or command/executable missing.
    - GENERAL_ERROR: fallback failure category.
    """
    err_str = (error or "").strip()
    out_str = (output or "").strip()
    tname = (tool_name or "").lower().strip()
    combined = f"{tname}\n{err_str}\n{out_str}".lower()

    # 1. Desktop window manager unavailable
    if (
        "desktop_unavailable" in combined
        or "hyprland session not detected" in combined
        or ("hyprland" in combined and any(k in combined for k in ("not detected", "not running", "not found", "unavailable", "unreachable")))
    ):
        return ErrorClassification(
            error_type="DESKTOP_UNAVAILABLE",
            summary="Desktop window manager (Hyprland) is unavailable",
            remedy_hint="Desktop window manager (Hyprland) is not running or socket is unreachable. Use standard shell utilities or verify XDG_RUNTIME_DIR.",
            allowed_actions=["execute_command"],
        )

    # 2. Window not found
    if (
        "window_not_found" in combined
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

    # 3. Browser unavailable
    if (
        "browser_unavailable" in combined
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

    # 4. Element not found
    if (
        "element_not_found" in combined
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

    # 5. State verification failure
    if (
        tname in ("verify_outcome", "verify_state", "check_state", "assert_outcome")
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

    # 4. Timeout
    if (
        exit_code == 124
        or "timed out" in combined
        or "timeout" in combined
    ):
        return ErrorClassification(
            error_type="TIMEOUT",
            summary="Command execution timed out",
            remedy_hint="Command timed out. Avoid interactive prompts or pagers, add non-blocking flags, or scope command.",
            allowed_actions=["execute_command"],
        )

    # 3. Permission Denied
    if (
        exit_code == 126
        or "permission denied" in combined
        or "operation not permitted" in combined
    ):
        return ErrorClassification(
            error_type="PERMISSION_DENIED",
            summary="Permission denied on target resource",
            remedy_hint="Permission denied on target resource. Inspect permissions using 'ls -la'. Do NOT invoke interactive sudo.",
            allowed_actions=["execute_command", "manage_workspace_file"],
        )

    # 4. Port in use
    if (
        exit_code == 98
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

    # 5. Path / File / Directory not found
    path_keywords = (
        "no such file",
        "file not found",
        "directory not found",
        "directory does not exist",
        "file does not exist",
        "path is not a regular file",
        "path is not a directory",
    )
    if any(k in combined for k in path_keywords):
        return ErrorClassification(
            error_type="PATH_NOT_FOUND",
            summary="Target file or directory does not exist",
            remedy_hint="Target file or directory does not exist. Verify active working directory with 'pwd' and list contents with 'ls'.",
            allowed_actions=["execute_command", "manage_workspace_file", "verify_outcome"],
        )

    # 6. Command not found
    if (
        exit_code == 127
        or "command not found" in combined
        or ("not found" in combined and not any(k in combined for k in ("element", "tab", "process", "key")))
    ):
        return ErrorClassification(
            error_type="COMMAND_NOT_FOUND",
            summary="Command not found in PATH",
            remedy_hint="Command not found in PATH. Verify command spelling, inspect installed binaries with 'which' or 'find', or use an alternative tool.",
            allowed_actions=["execute_command", "manage_workspace_file"],
        )

    # 7. Tool crafting failure
    if (
        tname in ("craft_tool", "create_tool", "write_tool", "hotload_tool")
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

    # 8. Routine operation failure
    if (
        tname in ("manage_routine", "schedule_task", "create_routine", "schedule_routine", "cron", "cron_job")
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

    # 9. Fallback General Error
    return ErrorClassification(
        error_type="GENERAL_ERROR",
        summary="Action execution failed",
        remedy_hint="Action failed. Inspect error details and adjust arguments or try an alternative approach.",
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
    ) -> ErrorClassification:
        """Classify tool error into canonical category."""
        return classify_tool_error(tool_name, exit_code, output, error)
