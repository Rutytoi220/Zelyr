"""Deterministic State Verification Tool for Zelyr.

Provides the canonical verify_outcome tool for verifying expected system and
filesystem state outcomes deterministically.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from zelyr.engine.verifier import (
    OutcomeVerifier,
    VerificationResult,
    verify_outcome,
)
from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

REQUIRED_RING = 1
TIER = 1
TUI_HINT = "Verifying system state..."

TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "verify_outcome",
        "description": (
            "Deterministically verify an expected system or filesystem state outcome "
            "(files, contents, checksums, JSON, running processes, listening ports, command outputs)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "check": {
                    "type": "string",
                    "enum": [
                        "file_exists",
                        "file_contains",
                        "file_checksum",
                        "file_json",
                        "process_running",
                        "port_listening",
                        "command_output",
                    ],
                    "description": "The type of state verification to perform.",
                },
                "target": {
                    "type": "string",
                    "description": "Target of the check: file path, process name/pattern, port number (or host:port), or command string.",
                },
                "expected": {
                    "type": "string",
                    "description": "Expected value or pattern (required for file_contains, file_checksum, command_output).",
                },
            },
            "required": ["check", "target"],
        },
    },
}


class VerifyOutcomeTool(BaseTool):
    """Deterministically verifies system state outcomes."""

    REQUIRED_RING = 1
    TIER = 1
    TUI_HINT = "Verifying system state..."

    def __init__(self):
        super().__init__(
            tool_id="verify_outcome",
            name="verify_outcome",
            description=(
                "Deterministically verify an expected system or filesystem state outcome "
                "(files, contents, checksums, JSON, running processes, listening ports, command outputs)."
            ),
            tier=1,
            is_core=True,
        )
        self.parameters = [
            ToolParameter(
                name="check",
                type="string",
                description="Verification check type: file_exists, file_contains, file_checksum, file_json, process_running, port_listening, command_output.",
                required=True,
            ),
            ToolParameter(
                name="target",
                type="string",
                description="Target file path, process pattern, port, or command string.",
                required=True,
            ),
            ToolParameter(
                name="expected",
                type="string",
                description="Expected content, checksum, or pattern.",
                required=False,
                default=None,
            ),
        ]

    async def execute(
        self,
        check: str = "",
        target: str = "",
        expected: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute verification check and return structured ToolResult."""
        if not check and "check" in kwargs:
            check = kwargs.pop("check")
        if not target and "target" in kwargs:
            target = kwargs.pop("target")
        if expected is None and "expected" in kwargs:
            expected = kwargs.pop("expected")

        res: VerificationResult = verify_outcome(check=check, target=target, expected=expected, **kwargs)
        if res.success:
            return ToolResult(
                success=True,
                output=f"[VERIFIED] {res.check} on {res.target}: PASS ({json.dumps(res.observed)})",
                metadata={"verification": res.to_dict()},
            )
        return ToolResult(
            success=False,
            error=res.error or f"Verification failed for {res.check} on {res.target}",
            metadata={"verification": res.to_dict()},
        )


async def execute(
    check: str = "",
    target: str = "",
    expected: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Dynamic tool runner returning JSON string payload."""
    if not check and "check" in kwargs:
        check = kwargs.pop("check")
    if not target and "target" in kwargs:
        target = kwargs.pop("target")
    if expected is None and "expected" in kwargs:
        expected = kwargs.pop("expected")

    res = verify_outcome(check=check, target=target, expected=expected, **kwargs)
    payload = {
        "status": "success" if res.success else "error",
        "verified": res.success,
        "check": res.check,
        "target": res.target,
        "observed": res.observed,
    }
    if res.error:
        payload["error"] = res.error
    return json.dumps(payload)
