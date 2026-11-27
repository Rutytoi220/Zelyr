"""Deterministic State Verification Tool for Zelyr.

Provides the canonical verify_outcome tool for verifying expected system and
filesystem state outcomes deterministically.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from zelyr.engine.verifier import (
    ALL_KNOWN_VERIFICATION_CHECKS,
    OutcomeVerifier,
    VerificationResult,
    verify_outcome_async,
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
            "Deterministically verify an expected system, filesystem, browser, or desktop state outcome "
            "(files, contents, checksums, JSON, processes, listening ports, command outputs, "
            "read-only Chromium DOM/URL checks, read-only Hyprland compositor & AT-SPI2 widget checks)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "check": {
                    "type": "string",
                    "enum": sorted(ALL_KNOWN_VERIFICATION_CHECKS),
                    "description": "The type of state verification to perform.",
                },
                "target": {
                    "type": "string",
                    "description": "Target of the check: file path, process pattern, port, command, CSS selector/URL, or window identifier (or 'window::selector').",
                },
                "expected": {
                    "type": "string",
                    "description": "Expected value, text, state, or pattern.",
                },
                "window": {
                    "type": "string",
                    "description": "Optional explicit target desktop window title/class/address for accessibility checks.",
                },
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector (browser) or AT-SPI2 widget selector (#id, role=..., name=...) for desktop accessibility checks.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory to execute verification checks against.",
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
                description="Expected content, checksum, state, or pattern.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="window",
                type="string",
                description="Optional target desktop window title/class/address for accessibility checks.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="selector",
                type="string",
                description="Optional widget/DOM selector (#id, role=..., name=...).",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="Optional working directory to execute verification checks against.",
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

        res: VerificationResult = await verify_outcome_async(check=check, target=target, expected=expected, **kwargs)
        if res.success:
            return ToolResult(
                success=True,
                output=f"[VERIFIED] {res.check} on {res.target}: PASS ({json.dumps(res.observed)})",
                metadata={"verification": res.to_detailed_dict()},
            )
        return ToolResult(
            success=False,
            error=res.error or f"Verification {res.result} for {res.check} on {res.target}",
            metadata={"verification": res.to_detailed_dict()},
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

    res = await verify_outcome_async(check=check, target=target, expected=expected, **kwargs)
    payload = {
        "status": "success" if res.success else "error",
        "verified": res.success,
        "result": res.result,
        "evidence_source": res.evidence_source,
        "evidence_collected": res.evidence_collected,
        "expected": res.expected,
        "check": res.check,
        "target": res.target,
        "observed": res.observed,
        "criterion": res.criterion,
    }
    if res.error:
        payload["error"] = res.error
    return json.dumps(payload)
