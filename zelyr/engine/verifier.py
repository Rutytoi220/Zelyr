"""Independent Outcome Verifier Engine for deterministic state assertions.

Standard-library based state verification for files, contents, checksums,
JSON formatting, processes, listening ports, and command outputs.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union
import urllib.parse


MAX_COMMAND_OUTPUT_BYTES = 512 * 1024  # 512 KB

ALLOWED_VERIFICATION_BINARIES: Set[str] = {
    "git",
    "pytest",
    "python",
    "python3",
    "echo",
    "sleep",
}

ALLOWED_GIT_SUBCOMMANDS: Set[str] = {
    "status",
    "diff",
    "log",
    "branch",
    "rev-parse",
    "show",
    "describe",
    "tag",
    "check-ignore",
    "version",
}

ALLOWED_PYTHON_MODULES: Set[str] = {
    "pytest",
    "unittest",
    "py_compile",
    "compileall",
}

DISALLOWED_SHELL_INTERPRETERS: Set[str] = {
    "sh",
    "bash",
    "zsh",
    "csh",
    "tcsh",
    "dash",
    "ksh",
    "fish",
    "cmd",
    "powershell",
    "pwsh",
}

DANGEROUS_GIT_FLAGS: Tuple[str, ...] = (
    "--exec-path",
    "--upload-pack",
    "--receive-pack",
    "--output",
    "-c",
    "--config",
    "--paginate",
    "-p",
    "-C",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--ext-diff",
    "--textconv",
)

_MUTATING_GIT_BRANCH_FLAGS: Tuple[str, ...] = (
    "-d", "-D", "-m", "-M", "-c", "-C", "-f",
    "--delete", "--move", "--copy", "--force",
    "--set-upstream-to", "--unset-upstream", "--edit-description", "--create-reflog",
)

_READONLY_GIT_BRANCH_FLAGS: Tuple[str, ...] = (
    "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose",
    "-l", "--list", "--show-current", "--contains", "--no-contains",
    "--merged", "--no-merged",
)

_MUTATING_GIT_TAG_FLAGS: Tuple[str, ...] = (
    "-a", "-s", "-u", "-d", "-f", "-m", "-F",
    "--annotate", "--sign", "--local-user", "--delete", "--force", "--message", "--file",
)

_READONLY_GIT_TAG_FLAGS: Tuple[str, ...] = ("-l", "--list", "-v", "--verify")

STRICT_ENV_ALLOWLIST: Set[str] = {
    # System execution basics
    "PATH",
    "SYSTEMROOT",
    "COMSPEC",
    "PATHEXT",
    # Temporary files
    "TEMP",
    "TMP",
    "TMPDIR",
    # User identity
    "HOME",
    "USER",
    "LOGNAME",
    "USERPROFILE",
    # Localization
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    # Python runtime
    "VIRTUAL_ENV",
    "PYTHONUNBUFFERED",
    "PYTHONIOENCODING",
}


BROWSER_VERIFICATION_CHECKS: Set[str] = {
    "browser_url",
    "browser_navigation",
    "browser_origin",
    "browser_title",
    "browser_element_present",
    "browser_element_absent",
    "browser_visible_text",
    "browser_element_state",
    "browser_field_value",
}

DESKTOP_COMPOSITOR_CHECKS: Set[str] = {
    "desktop_window_exists",
    "desktop_window_title",
    "desktop_active_window",
}

DESKTOP_ACCESSIBILITY_CHECKS: Set[str] = {
    "desktop_visible_text",
    "desktop_widget_state",
    "desktop_text",
    "desktop_status",
    "desktop_ui_state",
}

# Backward-compatible alias preserved for external callers
DESKTOP_SEMANTIC_UNSUPPORTED_CHECKS: Set[str] = DESKTOP_ACCESSIBILITY_CHECKS

DESKTOP_VERIFICATION_CHECKS: Set[str] = (
    DESKTOP_COMPOSITOR_CHECKS | DESKTOP_ACCESSIBILITY_CHECKS
)

FILESYSTEM_AND_HOST_CHECKS: Set[str] = {
    "file_exists", "exists",
    "file_contains", "contains",
    "file_checksum", "checksum", "sha256",
    "file_json", "json",
    "process_running", "process", "proc",
    "port_listening", "port", "listen",
    "command_output", "command", "cmd",
}

ALL_KNOWN_VERIFICATION_CHECKS: Set[str] = (
    FILESYSTEM_AND_HOST_CHECKS
    | BROWSER_VERIFICATION_CHECKS
    | DESKTOP_VERIFICATION_CHECKS
)

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_UNVERIFIED = "UNVERIFIED"
VERDICT_UNSUPPORTED = "UNSUPPORTED"
VALID_VERDICTS = frozenset({
    VERDICT_PASS,
    VERDICT_FAIL,
    VERDICT_UNVERIFIED,
    VERDICT_UNSUPPORTED,
})

_EVIDENCE_SOURCE_BY_CHECK: Dict[str, str] = {
    "file_exists": "filesystem",
    "exists": "filesystem",
    "file_contains": "filesystem",
    "contains": "filesystem",
    "file_checksum": "filesystem",
    "checksum": "filesystem",
    "sha256": "filesystem",
    "file_json": "filesystem",
    "json": "filesystem",
    "process_running": "proc_table",
    "process": "proc_table",
    "proc": "proc_table",
    "port_listening": "socket_probe",
    "port": "socket_probe",
    "listen": "socket_probe",
    "command_output": "command_execution",
    "command": "command_execution",
    "cmd": "command_execution",
    "browser_url": "browser_native_messaging",
    "browser_navigation": "browser_native_messaging",
    "browser_origin": "browser_native_messaging",
    "browser_title": "browser_native_messaging",
    "browser_element_present": "browser_native_messaging",
    "browser_element_absent": "browser_native_messaging",
    "browser_visible_text": "browser_native_messaging",
    "browser_element_state": "browser_native_messaging",
    "browser_field_value": "browser_native_messaging",
    "desktop_window_exists": "hyprland_ipc",
    "desktop_window_title": "hyprland_ipc",
    "desktop_active_window": "hyprland_ipc",
    "desktop_visible_text": "desktop_accessibility",
    "desktop_widget_state": "desktop_accessibility",
    "desktop_text": "desktop_accessibility",
    "desktop_status": "desktop_accessibility",
    "desktop_ui_state": "desktop_accessibility",
}


def _infer_verdict(success: bool, observed: Any, explicit_result: Optional[str]) -> str:
    if explicit_result:
        norm = str(explicit_result).strip().upper()
        if norm in VALID_VERDICTS:
            return norm
    obs = observed if isinstance(observed, dict) else {}
    if obs.get("unsupported"):
        return VERDICT_UNSUPPORTED
    if obs.get("unverified") or (obs.get("evidence_collected") is False and not obs.get("invalid_spec")):
        return VERDICT_UNVERIFIED
    return VERDICT_PASS if success else VERDICT_FAIL


def _infer_evidence_collected(
    *,
    result: str,
    observed: Any,
    explicit: Optional[bool],
) -> bool:
    if explicit is not None:
        return bool(explicit)
    if result in (VERDICT_UNSUPPORTED, VERDICT_UNVERIFIED):
        return False
    obs = observed if isinstance(observed, dict) else {}
    if "evidence_collected" in obs:
        return bool(obs["evidence_collected"])
    if (
        obs.get("invalid_spec")
        or obs.get("safety_blocked")
        or obs.get("workspace_escape")
        or obs.get("interpreter_rejected")
        or obs.get("timeout")
        or obs.get("cancelled")
        or obs.get("execution_error")
    ):
        return False
    return True


@dataclass(frozen=True, init=False)
class VerificationResult:
    """Immutable result of an outcome verification check."""

    success: bool
    check: str
    target: str
    observed: Any
    error: Optional[str] = None

    def __init__(
        self,
        success: bool,
        check: str,
        target: str,
        observed: Any,
        error: Optional[str] = None,
        *,
        result: Optional[str] = None,
        evidence_source: Optional[str] = None,
        expected: Any = None,
        evidence_collected: Optional[bool] = None,
        criterion: Optional[Dict[str, Any]] = None,
    ) -> None:
        verdict = _infer_verdict(bool(success), observed, result)
        # Fail closed: UNSUPPORTED / UNVERIFIED / FAIL cannot report success=True.
        eff_success = bool(success) and verdict == VERDICT_PASS
        object.__setattr__(self, "success", eff_success)
        object.__setattr__(self, "check", check)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "error", error)

        obs = observed if isinstance(observed, dict) else {}
        eff_expected = expected
        if eff_expected is None and isinstance(obs, dict):
            if "expected" in obs:
                eff_expected = obs.get("expected")
            elif "pattern" in obs:
                eff_expected = obs.get("pattern")
        eff_source = str(
            evidence_source
            or _EVIDENCE_SOURCE_BY_CHECK.get(str(check or "").strip().lower(), "verifier")
        )
        eff_collected = _infer_evidence_collected(
            result=verdict,
            observed=observed,
            explicit=evidence_collected,
        )
        eff_criterion = (
            dict(criterion)
            if isinstance(criterion, dict)
            else {"check": check, "target": target, "expected": eff_expected}
        )
        object.__setattr__(
            self,
            "_evidence_meta",
            {
                "result": verdict,
                "evidence_source": eff_source,
                "expected": eff_expected,
                "evidence_collected": eff_collected,
                "criterion": eff_criterion,
            },
        )

    @property
    def result(self) -> str:
        """Normalized four-state outcome: PASS, FAIL, UNVERIFIED, or UNSUPPORTED."""
        return str(self._evidence_meta["result"])

    @property
    def evidence_source(self) -> str:
        """Trusted verification mechanism used to gather evidence."""
        return str(self._evidence_meta["evidence_source"])

    @property
    def expected(self) -> Any:
        """Expected condition or value asserted by the criterion."""
        return self._evidence_meta["expected"]

    @property
    def evidence_collected(self) -> bool:
        """Whether trusted evidence was successfully gathered from the target surface."""
        return bool(self._evidence_meta["evidence_collected"])

    @property
    def criterion(self) -> Dict[str, Any]:
        """Structured criterion specification checked."""
        return dict(self._evidence_meta["criterion"])

    @property
    def status(self) -> str:
        """Return structured outcome status category preserving 5-field dict compatibility."""
        if self.success:
            return "verified"
        if self.result == VERDICT_UNSUPPORTED:
            return "unsupported"
        obs = self.observed if isinstance(self.observed, dict) else {}
        if obs.get("safety_blocked") or obs.get("workspace_escape") or obs.get("interpreter_rejected"):
            return "rejected"
        if obs.get("timeout"):
            return "timeout"
        if obs.get("cancelled"):
            return "cancelled"
        if obs.get("invalid_spec"):
            return "invalid_spec"
        if self.result == VERDICT_UNVERIFIED or obs.get("unverified"):
            return "unverified"
        if obs.get("execution_error") or obs.get("truncated"):
            return "execution_error"
        return "failed"

    def to_dict(self, *, detailed: bool = False) -> Dict[str, Any]:
        """Convert result to a dictionary (5-field compatible by default; rich evidence when detailed=True)."""
        base = asdict(self)
        if not detailed:
            return base
        return {
            **base,
            "result": self.result,
            "status": self.status,
            "criterion": self.criterion,
            "evidence_source": self.evidence_source,
            "expected": self.expected,
            "evidence_collected": self.evidence_collected,
        }

    def to_detailed_dict(self) -> Dict[str, Any]:
        """Return dictionary including full evidence attribution and four-state verdict."""
        return self.to_dict(detailed=True)


def _resolve_target_path(
    target_str: str,
    cwd: Optional[Union[str, Path]] = None,
    allow_external: bool = False,
) -> Path:
    """Resolve file path target and enforce workspace confinement when cwd is supplied."""
    if "\x00" in target_str:
        raise ValueError("Invalid target path: null byte is not permitted.")

    p = Path(os.path.expanduser(target_str))
    if cwd is not None:
        workspace_root = Path(cwd).resolve()
        resolved = p.resolve() if p.is_absolute() else (workspace_root / p).resolve()
        if not allow_external:
            try:
                resolved.relative_to(workspace_root)
            except ValueError as exc:
                raise PermissionError(
                    f"Workspace boundary violation: target '{target_str}' resolves outside workspace '{workspace_root}'."
                ) from exc
        return resolved

    if p.is_absolute():
        return p.resolve()
    return (Path.cwd().resolve() / p).resolve()


def verify_file_exists(
    path: Union[str, Path],
    cwd: Optional[Union[str, Path]] = None,
    expected: Optional[str] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Verify that a target file or path exists."""
    target_str = "" if path is None else str(path).strip()
    if not target_str:
        return VerificationResult(
            success=False,
            check="file_exists",
            target="",
            observed={"exists": False, "invalid_spec": True},
            error="Incomplete verification specification: 'file_exists' requires a non-empty target path.",
        )
    if expected is not None and str(expected).strip() != "":
        return VerificationResult(
            success=False,
            check="file_exists",
            target=target_str,
            observed={"expected_content_provided": expected, "invalid_spec": True},
            error=(
                "Incompatible verification specification: 'file_exists' only verifies file existence "
                "and does not accept expected content. Use 'file_contains:target:expected' instead."
            ),
        )
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd, allow_external=bool(kwargs.get("allow_external", False)))
        if expanded.exists():
            size = expanded.stat().st_size if expanded.is_file() else 0
            is_dir = expanded.is_dir()
            return VerificationResult(
                success=True,
                check="file_exists",
                target=target_str,
                observed={"exists": True, "is_file": not is_dir, "is_dir": is_dir, "size": size},
            )
        return VerificationResult(
            success=False,
            check="file_exists",
            target=target_str,
            observed={"exists": False},
            error=f"File not found: {target_str}",
        )
    except PermissionError as exc:
        return VerificationResult(
            success=False,
            check="file_exists",
            target=target_str,
            observed={"exists": False, "workspace_escape": "Workspace boundary" in str(exc)},
            error=str(exc),
        )
    except ValueError as exc:
        return VerificationResult(
            success=False,
            check="file_exists",
            target=target_str,
            observed={"exists": False, "invalid_spec": True},
            error=str(exc),
        )
    except Exception as exc:
        return VerificationResult(
            success=False,
            check="file_exists",
            target=target_str,
            observed={"exists": False},
            error=f"Error checking file existence '{target_str}': {exc}",
        )


def verify_file_contains(
    path: Union[str, Path],
    pattern: str,
    regex: bool = False,
    cwd: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Verify that a target file contains the specified pattern or regex."""
    target_str = "" if path is None else str(path).strip()
    if not target_str:
        return VerificationResult(
            success=False,
            check="file_contains",
            target="",
            observed={"exists": False, "invalid_spec": True},
            error="Incomplete verification specification: 'file_contains' requires a non-empty target path.",
        )
    if pattern is None or str(pattern) == "":
        return VerificationResult(
            success=False,
            check="file_contains",
            target=target_str,
            observed={"matched": False, "invalid_spec": True},
            error="Incomplete verification specification: 'file_contains' requires expected pattern or content.",
        )
    pattern_str = str(pattern)
    compiled_re: Optional[re.Pattern[str]] = None
    if regex:
        try:
            compiled_re = re.compile(pattern_str)
        except re.error as exc:
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_str,
                observed={"matched": False, "invalid_spec": True},
                error=f"Invalid regular expression pattern for 'file_contains': {exc}",
            )
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd, allow_external=bool(kwargs.get("allow_external", False)))
        if not expanded.exists():
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_str,
                observed={"exists": False, "matched": False},
                error=f"File not found: {target_str}",
            )
        if expanded.is_dir():
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_str,
                observed={"exists": True, "is_dir": True, "matched": False},
                error=f"Target path is a directory, expected regular file: {target_str}",
            )
        if not expanded.is_file():
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_str,
                observed={"exists": True, "is_file": False, "matched": False},
                error=f"Target path is not a regular file: {target_str}",
            )

        content = expanded.read_text(encoding="utf-8", errors="replace")
        if compiled_re is not None:
            match = compiled_re.search(content)
            if match:
                return VerificationResult(
                    success=True,
                    check="file_contains",
                    target=target_str,
                    observed={"matched": True, "snippet": match.group(0)[:120], "regex": True},
                )
        else:
            if pattern_str in content:
                return VerificationResult(
                    success=True,
                    check="file_contains",
                    target=target_str,
                    observed={"matched": True, "snippet": pattern_str[:120], "regex": False},
                )

        return VerificationResult(
            success=False,
            check="file_contains",
            target=target_str,
            observed={"matched": False},
            error=f"Pattern '{pattern_str}' not found in {target_str}",
        )
    except PermissionError as exc:
        return VerificationResult(
            success=False,
            check="file_contains",
            target=target_str,
            observed={"matched": False, "workspace_escape": "Workspace boundary" in str(exc)},
            error=str(exc),
        )
    except ValueError as exc:
        return VerificationResult(
            success=False,
            check="file_contains",
            target=target_str,
            observed={"matched": False, "invalid_spec": True},
            error=str(exc),
        )
    except Exception as exc:
        return VerificationResult(
            success=False,
            check="file_contains",
            target=target_str,
            observed={},
            error=f"Error reading file '{target_str}': {exc}",
        )


def verify_file_checksum(
    path: Union[str, Path],
    expected_sha256: str,
    cwd: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Verify that a target file matches the expected SHA-256 digest."""
    target_str = "" if path is None else str(path).strip()
    if not target_str:
        return VerificationResult(
            success=False,
            check="file_checksum",
            target="",
            observed={"exists": False, "invalid_spec": True},
            error="Incomplete verification specification: 'file_checksum' requires a non-empty target path.",
        )
    if expected_sha256 is None or not str(expected_sha256).strip():
        return VerificationResult(
            success=False,
            check="file_checksum",
            target=target_str,
            observed={"invalid_spec": True},
            error="Incomplete verification specification: 'file_checksum' requires an expected SHA-256 digest.",
        )
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd, allow_external=bool(kwargs.get("allow_external", False)))
        if not expanded.exists():
            return VerificationResult(
                success=False,
                check="file_checksum",
                target=target_str,
                observed={"exists": False},
                error=f"File not found: {target_str}",
            )
        if not expanded.is_file():
            return VerificationResult(
                success=False,
                check="file_checksum",
                target=target_str,
                observed={"exists": True, "is_dir": expanded.is_dir()},
                error=f"Target path is not a regular file: {target_str}",
            )

        hasher = hashlib.sha256()
        with open(expanded, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

        digest = hasher.hexdigest().lower()
        expected = (expected_sha256 or "").strip().lower()

        if digest == expected:
            return VerificationResult(
                success=True,
                check="file_checksum",
                target=target_str,
                observed={"sha256": digest},
            )
        return VerificationResult(
            success=False,
            check="file_checksum",
            target=target_str,
            observed={"sha256": digest},
            error=f"Checksum mismatch for {target_str}: expected {expected}, got {digest}",
        )
    except PermissionError as exc:
        return VerificationResult(
            success=False,
            check="file_checksum",
            target=target_str,
            observed={"workspace_escape": "Workspace boundary" in str(exc)},
            error=str(exc),
        )
    except ValueError as exc:
        return VerificationResult(
            success=False,
            check="file_checksum",
            target=target_str,
            observed={"invalid_spec": True},
            error=str(exc),
        )
    except Exception as exc:
        return VerificationResult(
            success=False,
            check="file_checksum",
            target=target_str,
            observed={},
            error=f"Failed to calculate checksum for '{target_str}': {exc}",
        )


def verify_file_json(
    path: Union[str, Path],
    schema: Optional[Dict[str, Any]] = None,
    cwd: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Verify that a target file contains valid JSON and optionally matches required keys."""
    target_str = "" if path is None else str(path).strip()
    if not target_str:
        return VerificationResult(
            success=False,
            check="file_json",
            target="",
            observed={"exists": False, "invalid_spec": True},
            error="Incomplete verification specification: 'file_json' requires a non-empty target path.",
        )
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd, allow_external=bool(kwargs.get("allow_external", False)))
        if not expanded.exists():
            return VerificationResult(
                success=False,
                check="file_json",
                target=target_str,
                observed={"exists": False},
                error=f"File not found: {target_str}",
            )
        if not expanded.is_file():
            return VerificationResult(
                success=False,
                check="file_json",
                target=target_str,
                observed={"exists": True, "is_dir": expanded.is_dir()},
                error=f"Target path is not a regular file: {target_str}",
            )

        content = expanded.read_text(encoding="utf-8")
        data = json.loads(content)

        observed: Dict[str, Any] = {
            "valid": True,
            "type": type(data).__name__,
        }
        if isinstance(data, dict):
            observed["keys"] = list(data.keys())
        elif isinstance(data, list):
            observed["length"] = len(data)

        if schema and isinstance(schema, dict):
            required_keys = schema.get("required_keys", [])
            if isinstance(required_keys, list) and isinstance(data, dict):
                missing = [k for k in required_keys if k not in data]
                if missing:
                    return VerificationResult(
                        success=False,
                        check="file_json",
                        target=target_str,
                        observed=observed,
                        error=f"JSON in {target_str} missing required keys: {missing}",
                    )

        return VerificationResult(
            success=True,
            check="file_json",
            target=target_str,
            observed=observed,
        )
    except json.JSONDecodeError as exc:
        return VerificationResult(
            success=False,
            check="file_json",
            target=target_str,
            observed={"valid": False},
            error=f"Invalid JSON in {target_str}: {exc}",
        )
    except PermissionError as exc:
        return VerificationResult(
            success=False,
            check="file_json",
            target=target_str,
            observed={"workspace_escape": "Workspace boundary" in str(exc)},
            error=str(exc),
        )
    except ValueError as exc:
        return VerificationResult(
            success=False,
            check="file_json",
            target=target_str,
            observed={"invalid_spec": True},
            error=str(exc),
        )
    except Exception as exc:
        return VerificationResult(
            success=False,
            check="file_json",
            target=target_str,
            observed={},
            error=f"Error reading JSON from '{target_str}': {exc}",
        )


def verify_process_running(pattern: str) -> VerificationResult:
    """Verify that a process matching pattern is currently running."""
    pattern_str = str(pattern).strip()
    if not pattern_str:
        return VerificationResult(
            success=False,
            check="process_running",
            target="",
            observed={"matched_pids": [], "count": 0},
            error="Incomplete verification specification: 'process_running' requires a non-empty process pattern.",
        )
    pids = []
    # 1. Try Linux /proc inspection
    if os.path.exists("/proc"):
        try:
            for entry in os.listdir("/proc"):
                if entry.isdigit():
                    cmdline_path = os.path.join("/proc", entry, "cmdline")
                    try:
                        with open(cmdline_path, "rb") as f:
                            cmd = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
                            if pattern_str in cmd:
                                pids.append(int(entry))
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        continue
        except Exception:
            pass

    # 2. Fallback to ps if no pids found from /proc
    if not pids:
        try:
            res = subprocess.run(
                ["ps", "-eo", "pid,args"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if res.returncode == 0:
                curr_pid = os.getpid()
                for line in res.stdout.splitlines()[1:]:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        pid = int(parts[0])
                        args = parts[1]
                        if pid != curr_pid and pattern_str in args:
                            if pid not in pids:
                                pids.append(pid)
        except Exception:
            pass

    if pids:
        return VerificationResult(
            success=True,
            check="process_running",
            target=pattern_str,
            observed={"matched_pids": sorted(pids), "count": len(pids)},
        )
    return VerificationResult(
        success=False,
        check="process_running",
        target=pattern_str,
        observed={"matched_pids": [], "count": 0},
        error=f"No process matching '{pattern_str}' found",
    )


def verify_port_listening(
    port: Union[int, str],
    host: str = "127.0.0.1",
    timeout: float = 0.5,
) -> VerificationResult:
    """Verify that a TCP port is open and accepting connections."""
    target_str = str(port).strip()
    if not target_str:
        return VerificationResult(
            success=False,
            check="port_listening",
            target="",
            observed={"listening": False},
            error="Incomplete verification specification: 'port_listening' requires a non-empty port number.",
        )
    parsed_host = host
    if ":" in target_str:
        h_part, p_part = target_str.rsplit(":", 1)
        if h_part:
            parsed_host = h_part
        try:
            parsed_port = int(p_part)
        except ValueError:
            return VerificationResult(
                success=False,
                check="port_listening",
                target=target_str,
                observed={"listening": False},
                error=f"Invalid port: {p_part}",
            )
    else:
        try:
            parsed_port = int(target_str)
        except ValueError:
            return VerificationResult(
                success=False,
                check="port_listening",
                target=target_str,
                observed={"listening": False, "invalid_spec": True},
                error=f"Invalid port: {target_str}",
            )

    if parsed_port < 1 or parsed_port > 65535:
        return VerificationResult(
            success=False,
            check="port_listening",
            target=f"{parsed_host}:{parsed_port}",
            observed={"listening": False, "invalid_spec": True},
            error=f"Invalid port {parsed_port}: port must be between 1 and 65535.",
        )

    try:
        with socket.create_connection((parsed_host, parsed_port), timeout=timeout):
            return VerificationResult(
                success=True,
                check="port_listening",
                target=f"{parsed_host}:{parsed_port}",
                observed={"listening": True, "host": parsed_host, "port": parsed_port},
            )
    except (ConnectionRefusedError, socket.timeout, OSError) as exc:
        return VerificationResult(
            success=False,
            check="port_listening",
            target=f"{parsed_host}:{parsed_port}",
            observed={"listening": False, "host": parsed_host, "port": parsed_port},
            error=f"Port {parsed_port} on {parsed_host} is not listening: {exc}",
        )


def _resolve_executable(raw_exe: str, exe_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Strictly resolve allowed binary from trusted directories, ignoring attacker-controlled PATH."""
    venv_bin = Path(sys.executable).parent
    trusted_dirs = [venv_bin]
    if sys.platform != "win32":
        trusted_dirs.extend([Path("/usr/bin"), Path("/bin"), Path("/usr/local/bin")])

    if exe_name in ("python", "python3"):
        return True, sys.executable, None

    if exe_name == "pytest":
        venv_pytest = venv_bin / "pytest"
        if venv_pytest.is_file() and os.access(venv_pytest, os.X_OK):
            return True, str(venv_pytest), None
        for d in trusted_dirs:
            cand = d / "pytest"
            if cand.is_file() and os.access(cand, os.X_OK):
                return True, str(cand), None
        return False, None, "Executable 'pytest' not found in active virtualenv or trusted system directories."

    if exe_name == "git":
        if sys.platform == "win32":
            found = shutil.which("git")
            if found and not found.startswith("."):
                return True, found, None
        else:
            for d in [Path("/usr/bin"), Path("/bin"), Path("/usr/local/bin")]:
                cand = d / "git"
                if cand.is_file() and os.access(cand, os.X_OK):
                    return True, str(cand), None
        return False, None, "Executable 'git' not found in trusted system directories (/usr/bin, /bin, /usr/local/bin)."

    if exe_name == "sleep":
        if sys.platform != "win32":
            for d in [Path("/usr/bin"), Path("/bin")]:
                cand = d / "sleep"
                if cand.is_file() and os.access(cand, os.X_OK):
                    return True, str(cand), None
        return False, None, "Executable 'sleep' not found in trusted system directories."

    if exe_name == "echo":
        if sys.platform != "win32":
            for d in [Path("/bin"), Path("/usr/bin")]:
                cand = d / "echo"
                if cand.is_file() and os.access(cand, os.X_OK):
                    return True, str(cand), None
        return True, "echo", None

    return False, None, f"Command '{raw_exe}' is not an approved verification binary."


def _validate_pytest_target_paths(args: List[str], resolved_workspace: Path) -> Optional[str]:
    """Validate positional and path-bearing pytest arguments stay within resolved_workspace."""
    path_flags = ("--rootdir", "--basetemp", "--confcutdir")
    idx = 0
    while idx < len(args):
        tok = args[idx]
        path_candidate: Optional[str] = None
        if not tok.startswith("-"):
            # Strip pytest nodeid suffix (e.g. tests/test_a.py::test_fn) when checking file path
            path_candidate = tok.split("::", 1)[0]
        else:
            for pf in path_flags:
                if tok.startswith(f"{pf}="):
                    path_candidate = tok.split("=", 1)[1]
                    break
                if tok == pf and idx + 1 < len(args):
                    path_candidate = args[idx + 1]
                    idx += 1
                    break
        if path_candidate:
            cand_path = (
                (resolved_workspace / path_candidate).resolve()
                if not Path(path_candidate).is_absolute()
                else Path(path_candidate).resolve()
            )
            try:
                cand_path.relative_to(resolved_workspace)
            except ValueError:
                return (
                    f"Safety violation: target path '{path_candidate}' must reside within the verified workspace directory ({resolved_workspace})."
                )
        idx += 1
    return None


def _validate_and_parse_command(
    command_str: str,
    cwd: Optional[Union[str, Path]] = None,
    allowed_binaries: Optional[Set[str]] = None,
) -> Tuple[bool, List[str], Optional[str], Dict[str, Any]]:
    """Validate and parse a command string for direct, shell-free execution.

    Returns:
        (is_valid, tokens, error_message, observed_metadata)
    """
    clean_cmd = command_str.strip()
    if not clean_cmd:
        return False, [], "Incomplete verification specification: 'command_output' requires a non-empty command.", {}

    # Reject shell chaining, piping, redirection, backgrounding, substitution
    if re.search(r"[;|<>&`\n\r]|\$\(", clean_cmd):
        return (
            False,
            [],
            "Safety violation: shell operators (chaining, piping, redirection, backgrounding, substitution) are prohibited in verification commands.",
            {"safety_blocked": True},
        )

    # Reject environment variable expansion
    if re.search(r"\$\{?[a-zA-Z_]", clean_cmd):
        return (
            False,
            [],
            "Safety violation: environment variable expansion is prohibited in verification commands.",
            {"safety_blocked": True},
        )

    is_posix = sys.platform != "win32"
    try:
        tokens = shlex.split(clean_cmd, posix=is_posix)
    except ValueError as exc:
        return False, [], f"Invalid command syntax: failed to parse arguments: {exc}", {"safety_blocked": True}

    if not tokens:
        return False, [], "Incomplete verification specification: command cannot be empty.", {}

    raw_exe = tokens[0]
    exe_name = Path(raw_exe).name.lower()
    if sys.platform == "win32" and exe_name.endswith(".exe"):
        exe_name = exe_name[:-4]

    if exe_name in DISALLOWED_SHELL_INTERPRETERS:
        return (
            False,
            [],
            f"Safety violation: shell interpreter '{exe_name}' is not permitted for verification.",
            {"safety_blocked": True, "interpreter_rejected": True},
        )

    # Check against active python
    is_active_python = False
    try:
        if raw_exe == sys.executable or (Path(raw_exe).is_file() and Path(raw_exe).resolve() == Path(sys.executable).resolve()):
            is_active_python = True
            exe_name = "python"
    except Exception:
        pass

    permitted = allowed_binaries or ALLOWED_VERIFICATION_BINARIES
    if exe_name not in permitted:
        return (
            False,
            [],
            f"Command '{raw_exe}' is not permit-listed for verification. Allowed: git (read-only), pytest, python (-m pytest/unittest), echo, sleep.",
            {"safety_blocked": True, "supported": False},
        )

    resolved_workspace = Path(cwd).resolve() if cwd else Path.cwd().resolve()

    # Argument-level validation
    if exe_name == "git":
        for tok in tokens[1:]:
            for d_flag in DANGEROUS_GIT_FLAGS:
                if tok == d_flag or tok.startswith(f"{d_flag}="):
                    return (
                        False,
                        [],
                        f"Safety violation: git flag '{tok}' is prohibited for verification.",
                        {"safety_blocked": True},
                    )
        subcmd = None
        for tok in tokens[1:]:
            if not tok.startswith("-"):
                subcmd = tok.lower()
                break
        if subcmd is None:
            return False, [], "Incomplete verification specification: git requires a subcommand (e.g. 'git status').", {}
        if subcmd not in ALLOWED_GIT_SUBCOMMANDS:
            return (
                False,
                [],
                f"Safety violation: git subcommand '{subcmd}' is not permitted. Only read-only inspection subcommands are allowed ({', '.join(sorted(ALLOWED_GIT_SUBCOMMANDS))}).",
                {"safety_blocked": True},
            )
        sub_args = tokens[tokens.index(next(t for t in tokens[1:] if not t.startswith("-"))) + 1 :]
        if subcmd == "branch":
            has_readonly_flag = any(
                t in _READONLY_GIT_BRANCH_FLAGS or t.startswith("--list=") for t in sub_args
            )
            for tok in sub_args:
                if tok in _MUTATING_GIT_BRANCH_FLAGS or any(tok.startswith(f"{f}=") for f in _MUTATING_GIT_BRANCH_FLAGS):
                    return (
                        False,
                        [],
                        f"Safety violation: mutating git branch flag '{tok}' is prohibited during verification.",
                        {"safety_blocked": True},
                    )
                if not tok.startswith("-") and not has_readonly_flag:
                    return (
                        False,
                        [],
                        "Safety violation: 'git branch <name>' cannot be used without read-only listing flags during verification.",
                        {"safety_blocked": True},
                    )
        elif subcmd == "tag":
            has_readonly_flag = any(t in _READONLY_GIT_TAG_FLAGS for t in sub_args)
            for tok in sub_args:
                if tok in _MUTATING_GIT_TAG_FLAGS or any(tok.startswith(f"{f}=") for f in _MUTATING_GIT_TAG_FLAGS):
                    return (
                        False,
                        [],
                        f"Safety violation: mutating git tag flag '{tok}' is prohibited during verification.",
                        {"safety_blocked": True},
                    )
                if not tok.startswith("-") and not has_readonly_flag:
                    return (
                        False,
                        [],
                        "Safety violation: 'git tag <name>' cannot be used without read-only listing/verify flags during verification.",
                        {"safety_blocked": True},
                    )

    elif exe_name in ("python", "python3"):
        args = tokens[1:]
        for tok in args:
            if tok == "-c" or tok.startswith("-c"):
                return (
                    False,
                    [],
                    "Safety violation: 'python -c' is not permitted for verification.",
                    {"safety_blocked": True},
                )
            if tok in ("-i", "-a"):
                return (
                    False,
                    [],
                    f"Safety violation: interactive python flag '{tok}' is not permitted.",
                    {"safety_blocked": True},
                )
        if "-m" not in args:
            return (
                False,
                [],
                "Safety violation: python invocation in verification must use '-m <module>' (e.g. 'python -m pytest'). Script paths and arbitrary code execution are prohibited.",
                {"safety_blocked": True},
            )
        m_idx = args.index("-m")
        if m_idx + 1 >= len(args):
            return False, [], "Incomplete verification specification: missing module name after '-m'.", {}
        mod_name = args[m_idx + 1].lower()
        if mod_name not in ALLOWED_PYTHON_MODULES:
            return (
                False,
                [],
                f"Safety violation: python module '{mod_name}' is not permitted for verification. Allowed modules: {', '.join(sorted(ALLOWED_PYTHON_MODULES))}.",
                {"safety_blocked": True},
            )

        path_err = _validate_pytest_target_paths(args[m_idx + 2:], resolved_workspace)
        if path_err is not None:
            return False, [], path_err, {"safety_blocked": True}

    elif exe_name == "pytest":
        path_err = _validate_pytest_target_paths(tokens[1:], resolved_workspace)
        if path_err is not None:
            return False, [], path_err, {"safety_blocked": True}

    elif exe_name == "sleep":
        if len(tokens) != 2:
            return False, [], "Verification command 'sleep' requires exactly one duration argument (e.g. 'sleep 1.0').", {}
        try:
            dur = float(tokens[1])
            if dur < 0:
                return False, [], "Duration for 'sleep' cannot be negative.", {}
        except ValueError:
            return False, [], f"Invalid duration for 'sleep': '{tokens[1]}'.", {}

    # Strict executable resolution from trusted directories
    ok, resolved_bin, err = _resolve_executable(raw_exe, exe_name)
    if not ok or not resolved_bin:
        return False, [], err or f"Failed to resolve executable '{raw_exe}'.", {"safety_blocked": True}

    tokens[0] = resolved_bin

    # Git Security Hardening: Inject disarming flags into git command tokens
    if exe_name == "git":
        devnull = "NUL" if sys.platform == "win32" else "/dev/null"
        git_top_level_flags = [
            "--no-pager",
            "-c", "core.pager=cat",
            "-c", "diff.external=",
            "-c", "core.fsmonitor=false",
            "-c", f"core.hooksPath={devnull}",
        ]
        rest_tokens = tokens[1:]
        if rest_tokens and rest_tokens[0] in ("diff", "show", "log"):
            rest_tokens = [rest_tokens[0], "--no-ext-diff", "--no-textconv"] + rest_tokens[1:]
        tokens = [tokens[0]] + git_top_level_flags + rest_tokens

    return True, tokens, None, {}


def _get_sanitized_environment(resolved_cwd: Path) -> Dict[str, str]:
    """Return a strict minimal environment dictionary for verification commands."""
    sanitized: Dict[str, str] = {}
    for k in STRICT_ENV_ALLOWLIST:
        if k in os.environ and k != "PYTHONPATH":
            sanitized[k] = os.environ[k]

    # Strictly controlled PATH: virtualenv bin + trusted system dirs only
    venv_bin = str(Path(sys.executable).parent)
    trusted_dirs = ["/usr/bin", "/bin", "/usr/local/bin"] if sys.platform != "win32" else []
    trusted_path = [venv_bin] + [d for d in trusted_dirs if Path(d).is_dir()]
    sanitized["PATH"] = os.pathsep.join(trusted_path)

    # Git security disarming environment overrides
    sanitized["GIT_PAGER"] = "cat"
    sanitized["PAGER"] = "cat"
    sanitized["GIT_EXTERNAL_DIFF"] = ""
    sanitized["GIT_CONFIG_NOSYSTEM"] = "1"
    sanitized["GIT_TERMINAL_PROMPT"] = "0"
    sanitized["GIT_OPTIONAL_LOCKS"] = "0"
    sanitized["GIT_ALLOW_PROTOCOL"] = ""

    # Controlled Python runtime environment:
    # Do not inherit arbitrary parent PYTHONPATH. Bind strictly to resolved workspace.
    sanitized["PYTHONPATH"] = str(resolved_cwd)
    sanitized["PYTHONNOUSERSITE"] = "1"
    sanitized["PYTHONDONTWRITEBYTECODE"] = "1"
    sanitized["PYTHONUNBUFFERED"] = "1"

    return sanitized


def _kill_process_tree(proc: subprocess.Popen, is_posix: bool) -> None:
    """Terminate the process and all owned child processes in its group.

    Note on Windows: Process tree cleanup via taskkill /F /T is best-effort.
    True process containment and guaranteed descendant termination on Windows
    requires OS-level Win32 Job Objects (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE).
    """
    if is_posix:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.kill()
        except Exception:
            pass
    else:
        # Resolve taskkill binary directly from SystemRoot\System32
        sys_root = os.environ.get("SystemRoot", r"C:\Windows")
        taskkill_path = os.path.join(sys_root, "System32", "taskkill.exe")
        taskkill_bin = taskkill_path if os.path.isfile(taskkill_path) else (shutil.which("taskkill") or "taskkill")
        try:
            subprocess.run(
                [taskkill_bin, "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=2.0,
            )
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass


def _read_stream_bounded(
    stream,
    buffer: List[bytes],
    total_bytes_counter: List[int],
    lock: threading.Lock,
    stop_event: threading.Event,
    max_bytes: int,
) -> None:
    """Read stream in chunks, enforcing hard limit on total bytes across all streams."""
    chunk_size = 4096
    try:
        while not stop_event.is_set():
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            with lock:
                curr = total_bytes_counter[0]
                total_bytes_counter[0] += len(chunk)
                if total_bytes_counter[0] > max_bytes:
                    allowed = max(0, max_bytes - curr)
                    if allowed > 0:
                        buffer.append(chunk[:allowed])
                    stop_event.set()
                    break
                else:
                    buffer.append(chunk)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _extract_command_failure_snippet(combined: str, max_chars: int = 900) -> str:
    """Extract a high-signal failure excerpt preserving both top and summary test failures."""
    raw = str(combined or "").strip()
    if not raw:
        return ""
    lines = [
        ln.strip()
        for ln in raw.splitlines()
        if ln.strip() and not set(ln.strip()) <= set("=-_~* ")
    ]
    if not lines:
        return raw[:max_chars]

    markers = (
        "FAILED",
        "ERROR",
        "AssertionError",
        "ValueError",
        "TypeError",
        "NameError",
        "SyntaxError",
        "IndentationError",
        "ImportError",
        "ModuleNotFoundError",
        "KeyError",
        "AttributeError",
        "ZeroDivisionError",
        "Traceback",
        "E   ",
        "Failed:",
        " failed",
    )
    hits: List[str] = []
    for ln in lines:
        # Exclude pytest source echo lines ("> assert ...") so failure headers/assertions take priority
        if not ln.startswith("> ") and any(m in ln for m in markers) and ln not in hits:
            hits.append(ln)

    chosen = list(hits) if hits else list(lines)
    if hits and lines[-1] not in chosen and (
        "failed" in lines[-1] or "passed" in lines[-1] or "error" in lines[-1].lower()
    ):
        chosen.append(lines[-1])

    joined = " | ".join(chosen)
    if len(joined) <= max_chars:
        return joined

    sep = " | ... | "
    head_parts: List[str] = []
    tail_parts: List[str] = []
    head_len = 0
    tail_len = 0
    left = 0
    right = len(chosen) - 1

    while left <= right:
        progressed = False
        h_item = chosen[left]
        h_add = len(h_item) + (3 if head_parts else 0)
        bridge_after_h = len(sep) if (right > left and tail_parts) else (3 if tail_parts else (len(sep) if right > left else 0))
        if head_len + tail_len + h_add + bridge_after_h <= max_chars:
            head_parts.append(h_item)
            head_len += h_add
            left += 1
            progressed = True

        if left <= right:
            t_item = chosen[right]
            t_add = len(t_item) + (3 if tail_parts else 0)
            bridge_after_t = len(sep) if right > left else (3 if head_parts else 0)
            if head_len + tail_len + t_add + bridge_after_t <= max_chars:
                tail_parts.insert(0, t_item)
                tail_len += t_add
                right -= 1
                progressed = True

        if not progressed:
            break

    if not head_parts and not tail_parts:
        return chosen[0][: max(1, max_chars - 3)].rstrip() + "..."
    if left <= right and head_parts and tail_parts:
        return (" | ".join(head_parts) + sep + " | ".join(tail_parts))[:max_chars]
    return " | ".join(head_parts + tail_parts)[:max_chars]


def verify_command_output(
    command: str,
    expected_pattern: str,
    timeout: float = 5.0,
    cwd: Optional[Union[str, Path]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> VerificationResult:
    """Verify that executing an allowlisted command produces output matching expected_pattern."""
    command_str = "" if command is None else str(command).strip()
    if not command_str:
        return VerificationResult(
            success=False,
            check="command_output",
            target="",
            observed={"invalid_spec": True},
            error="Incomplete verification specification: 'command_output' requires a non-empty command.",
        )
    if expected_pattern is None or str(expected_pattern) == "":
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed={"invalid_spec": True},
            error="Incomplete verification specification: 'command_output' requires an expected pattern or output substring.",
        )
    try:
        timeout_num = float(timeout)
        if timeout_num <= 0 or timeout_num != timeout_num:
            raise ValueError("must be positive")
    except (TypeError, ValueError):
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed={"invalid_spec": True},
            error=f"Invalid timeout value for 'command_output': {timeout!r}.",
        )

    if cancel_event is not None and cancel_event.is_set():
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed={"cancelled": True, "exit_code": None},
            error=f"Command verification cancelled before execution: {command_str}",
        )

    resolved_cwd = str(Path(cwd).resolve()) if cwd else str(Path.cwd().resolve())
    if not Path(resolved_cwd).is_dir():
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed={"invalid_spec": True},
            error=f"Working directory does not exist or is not a directory: {resolved_cwd}",
        )

    # Validate and parse command into direct execution tokens
    is_valid, tokens, err_msg, obs_meta = _validate_and_parse_command(command_str, cwd=resolved_cwd)
    if not is_valid:
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed=obs_meta,
            error=err_msg or "Command validation failed.",
        )

    eff_timeout = min(max(timeout_num, 0.1), 30.0)

    # Native simulation for echo if standalone binary not present
    if tokens[0] == "echo" and not shutil.which("echo"):
        sim_out = " ".join(tokens[1:]) + "\n"
        matched = (expected_pattern in sim_out)
        if not matched:
            try:
                matched = bool(re.search(expected_pattern, sim_out))
            except Exception:
                pass
        observed = {
            "exit_code": 0,
            "stdout": sim_out[:1000],
            "stderr": "",
            "matched": matched,
            "truncated": False,
        }
        if matched:
            return VerificationResult(
                success=True,
                check="command_output",
                target=command_str,
                observed=observed,
            )
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed=observed,
            error=f"Expected pattern '{expected_pattern}' not found in command output (exit 0)",
        )

    env = _get_sanitized_environment(Path(resolved_cwd))
    popen_kwargs: Dict[str, Any] = {
        "cwd": resolved_cwd,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }

    is_posix = sys.platform != "win32"
    if is_posix:
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        proc = subprocess.Popen(tokens, **popen_kwargs)
        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        total_counter = [0]
        lock = threading.Lock()
        stop_event = threading.Event()

        t_out = threading.Thread(
            target=_read_stream_bounded,
            args=(proc.stdout, stdout_chunks, total_counter, lock, stop_event, MAX_COMMAND_OUTPUT_BYTES),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_read_stream_bounded,
            args=(proc.stderr, stderr_chunks, total_counter, lock, stop_event, MAX_COMMAND_OUTPUT_BYTES),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        start_time = time.monotonic()
        timed_out = False
        cancelled = False
        truncated = False

        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                stop_event.set()
                _kill_process_tree(proc, is_posix)
                break

            if stop_event.is_set():
                truncated = True
                _kill_process_tree(proc, is_posix)
                break

            ret = proc.poll()
            if ret is not None:
                if total_counter[0] > MAX_COMMAND_OUTPUT_BYTES:
                    truncated = True
                break

            if (time.monotonic() - start_time) > eff_timeout:
                timed_out = True
                stop_event.set()
                _kill_process_tree(proc, is_posix)
                break

            time.sleep(0.01)

        # Ensure both reader threads are completely drained and terminated before processing chunks
        for t, s in ((t_out, proc.stdout), (t_err, proc.stderr)):
            t.join(timeout=1.0)
            if t.is_alive():
                try:
                    s.close()
                except Exception:
                    pass
                t.join(timeout=1.0)

        if cancelled:
            return VerificationResult(
                success=False,
                check="command_output",
                target=command_str,
                observed={"cancelled": True, "exit_code": None},
                error=f"Command verification cancelled: {command_str}",
            )

        if timed_out:
            return VerificationResult(
                success=False,
                check="command_output",
                target=command_str,
                observed={"timeout": True, "exit_code": None},
                error=f"Command timed out after {eff_timeout}s: {command_str}",
            )

        out = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        err = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        if truncated:
            return VerificationResult(
                success=False,
                check="command_output",
                target=command_str,
                observed={
                    "exit_code": proc.returncode,
                    "stdout": out[:1000],
                    "stderr": err[:1000],
                    "truncated": True,
                },
                error=f"Command output exceeded maximum allowed size ({MAX_COMMAND_OUTPUT_BYTES} bytes) and was truncated; verification cannot reliably succeed.",
            )

        matched = (expected_pattern in out) or (expected_pattern in err)
        if not matched:
            try:
                matched = bool(re.search(expected_pattern, out) or re.search(expected_pattern, err))
            except Exception:
                pass

        observed = {
            "exit_code": proc.returncode,
            "stdout": out[:1000],
            "stderr": err[:1000],
            "matched": matched,
            "truncated": False,
        }

        if matched:
            return VerificationResult(
                success=True,
                check="command_output",
                target=command_str,
                observed=observed,
            )
        combined = "\n".join(part.strip() for part in (out, err) if part and part.strip()).strip()
        snippet = _extract_command_failure_snippet(combined, max_chars=900)
        err_msg = f"Expected pattern '{expected_pattern}' not found in command output (exit {proc.returncode})"
        if snippet:
            err_msg = f"{err_msg} | output: {snippet}"
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed=observed,
            error=err_msg,
        )
    except Exception as exc:
        return VerificationResult(
            success=False,
            check="command_output",
            target=command_str,
            observed={"execution_error": True},
            error=f"Failed to execute command '{command_str}': {exc}",
        )


_FORBIDDEN_BROWSER_SELECTOR_RE = re.compile(
    r"(?:document\.cookie|localstorage|sessionstorage|indexeddb|chrome\.cookies|javascript:|window\.|eval\s*\(|new\s+function)",
    flags=re.IGNORECASE,
)

_BROWSER_STATE_BOOL_MAP: Dict[str, Tuple[str, bool]] = {
    "disabled": ("disabled", True),
    "enabled": ("disabled", False),
    "checked": ("checked", True),
    "unchecked": ("checked", False),
    "selected": ("selected", True),
    "unselected": ("selected", False),
    "visible": ("visible", True),
    "hidden": ("visible", False),
    "focused": ("focused", True),
    "unfocused": ("focused", False),
}


def _run_coro_sync(coro_factory: Callable[[], Awaitable[VerificationResult]]) -> VerificationResult:
    """Run an async verification coroutine synchronously without deadlocking active event loops."""
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return asyncio.run(coro_factory())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: asyncio.run(coro_factory()))
        return future.result(timeout=15.0)


def _normalize_http_url(url_str: str) -> Optional[Tuple[str, str, str, str, str]]:
    """Return normalized (scheme, netloc, path, query, fragment) for HTTP(S) URLs."""
    try:
        p = urllib.parse.urlparse(url_str.strip())
    except Exception:
        return None
    if p.scheme.lower() not in ("http", "https") or not p.netloc:
        return None
    path = p.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return (p.scheme.lower(), p.netloc.lower(), path, p.query, p.fragment)


_DESKTOP_STATE_BOOL_MAP: Dict[str, Tuple[str, bool]] = {
    "disabled": ("disabled", True),
    "enabled": ("disabled", False),
    "checked": ("checked", True),
    "unchecked": ("checked", False),
    "selected": ("selected", True),
    "unselected": ("selected", False),
    "visible": ("visible", True),
    "hidden": ("visible", False),
    "showing": ("showing", True),
    "focused": ("focused", True),
    "unfocused": ("focused", False),
    "editable": ("editable", True),
    "readonly": ("read_only", True),
    "read_only": ("read_only", True),
    "expanded": ("expanded", True),
    "collapsed": ("expanded", False),
    "active": ("active", True),
    "inactive": ("active", False),
    "indeterminate": ("indeterminate", True),
}


def _normalize_state_expectation(
    expected: Any,
    *,
    check_name: str,
    bool_map: Dict[str, Tuple[str, bool]],
    allowed_bool_keys: Set[str],
    string_keys: Tuple[str, ...] = (),
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Shared parser/validator for browser and desktop element/widget state expectations."""
    if expected is None:
        return None, f"Incomplete verification specification: '{check_name}' requires an expected state."
    if isinstance(expected, dict):
        raw_map = expected
    else:
        raw_s = str(expected).strip()
        if not raw_s:
            return None, f"Incomplete verification specification: '{check_name}' requires a non-empty expected state."
        if raw_s.startswith("{"):
            try:
                decoded = json.loads(raw_s)
            except Exception as exc:
                return None, f"Invalid JSON state specification for '{check_name}': {exc}"
            if not isinstance(decoded, dict) or not decoded:
                return None, f"JSON state specification for '{check_name}' must be a non-empty object."
            raw_map = decoded
        elif "=" in raw_s:
            k, v = [p.strip() for p in raw_s.split("=", 1)]
            k_low = k.lower()
            if k_low in string_keys or (k_low == "option" and "selected_option" in string_keys):
                canon_k = "selected_option" if k_low == "option" else k_low
                return {canon_k: v}, None
            if v.lower() in ("true", "1", "yes"):
                raw_map = {k_low: True}
            elif v.lower() in ("false", "0", "no"):
                raw_map = {k_low: False}
            else:
                return None, f"Unsupported boolean state value '{v}' for key '{k}'."
        else:
            tok_low = raw_s.lower()
            if tok_low in bool_map:
                field_name, field_bool = bool_map[tok_low]
                return {field_name: field_bool}, None
            return None, (
                f"Unsupported expected state '{raw_s}' for '{check_name}'. "
                f"Supported states: {', '.join(sorted(bool_map.keys()))}."
            )

    parsed: Dict[str, Any] = {}
    for k, v in raw_map.items():
        k_low = str(k).strip().lower()
        if k_low in allowed_bool_keys:
            if not isinstance(v, bool):
                return None, f"State key '{k_low}' expects a boolean value."
            parsed[k_low] = v
        elif k_low in string_keys:
            parsed[k_low] = str(v)
        else:
            return None, f"Unsupported state attribute '{k}' in '{check_name}'."
    return (parsed, None) if parsed else (None, f"Empty state specification for '{check_name}'.")


def _parse_browser_state_expectation(expected: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and parse expected browser element state into canonical dict assertions."""
    return _normalize_state_expectation(
        expected,
        check_name="browser_element_state",
        bool_map=_BROWSER_STATE_BOOL_MAP,
        allowed_bool_keys={"disabled", "checked", "selected", "visible", "focused"},
        string_keys=("selected_option",),
    )


def _parse_desktop_state_expectation(expected: Any, check_name: str = "desktop_widget_state") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate and parse expected desktop widget state into canonical dict assertions."""
    return _normalize_state_expectation(
        expected,
        check_name=check_name,
        bool_map=_DESKTOP_STATE_BOOL_MAP,
        allowed_bool_keys={
            "enabled", "disabled", "checked", "selected", "visible", "showing",
            "focused", "editable", "read_only", "expanded", "active", "indeterminate",
        },
        string_keys=(),
    )


def _extract_browser_context_args(
    target_val: str,
    kwargs: Dict[str, Any],
) -> Tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Extract and validate optional browser context filters (tab_id, expected_url, expected_origin, url_prefix)."""
    from zelyr.tools.browser_native_host import validate_browser_url

    ctx = kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {}
    tab_id_raw = kwargs.get("tab_id") if kwargs.get("tab_id") is not None else ctx.get("tab_id")
    tab_id: Optional[int] = None
    if tab_id_raw is not None and str(tab_id_raw).strip() != "":
        try:
            tab_id = int(tab_id_raw)
            if tab_id < 1:
                return None, None, None, None, "Browser verification 'tab_id' must be a positive integer."
        except (TypeError, ValueError):
            return None, None, None, None, f"Invalid browser verification 'tab_id': {tab_id_raw!r}."
    elif target_val.strip().lower().startswith("tab:"):
        raw_tid = target_val.strip()[4:].strip()
        try:
            tab_id = int(raw_tid)
            if tab_id < 1:
                return None, None, None, None, "Browser verification 'tab:<id>' requires a positive integer."
        except (TypeError, ValueError):
            return None, None, None, None, f"Invalid tab reference '{target_val}'."

    exp_url = kwargs.get("expected_url") or kwargs.get("url") or ctx.get("expected_url") or ctx.get("url")
    exp_origin = kwargs.get("expected_origin") or kwargs.get("origin") or ctx.get("expected_origin") or ctx.get("origin")
    url_prefix = kwargs.get("url_prefix") or ctx.get("url_prefix")

    for label, val in (("url", exp_url), ("origin", exp_origin), ("url_prefix", url_prefix)):
        if val is not None and str(val).strip():
            ok_u, u_err = validate_browser_url(str(val).strip())
            if not ok_u:
                return None, None, None, None, f"Invalid browser context '{label}': {u_err}"

    return (
        tab_id,
        str(exp_url).strip() if exp_url else None,
        str(exp_origin).strip() if exp_origin else None,
        str(url_prefix).strip() if url_prefix else None,
        None,
    )


def _validate_browser_params(
    norm_check: str,
    target_val: str,
    expected: Optional[Any],
    kwargs: Dict[str, Any],
) -> Union[Tuple[None, VerificationResult], Tuple[Dict[str, Any], None]]:
    """Pure parameter/selector/URL validation for browser verification checks (no IPC side effects)."""
    from zelyr.tools.browser_native_host import (
        MAX_SELECTOR_LEN,
        MAX_URL_LEN,
        validate_browser_url,
    )

    canon_check = "browser_url" if norm_check == "browser_navigation" else norm_check
    ctx = kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {}
    criterion_obj = {"check": canon_check, "target": target_val, "expected": expected}

    def _invalid(msg: str, obs: Optional[Dict[str, Any]] = None) -> VerificationResult:
        return VerificationResult(
            success=False,
            check=canon_check,
            target=target_val,
            observed={"invalid_spec": True, **(obs or {})},
            error=msg,
            result=VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )

    if "\x00" in target_val or (isinstance(expected, str) and "\x00" in expected):
        return None, _invalid("Invalid browser verification specification: null byte is not permitted.")

    tab_id, exp_url, exp_origin, url_prefix, ctx_err = _extract_browser_context_args(target_val, kwargs)
    if ctx_err:
        return None, _invalid(ctx_err)

    match_mode = str(kwargs.get("match") or ctx.get("match") or "").strip().lower()
    selector: Optional[str] = None
    parsed_state: Optional[Dict[str, Any]] = None
    eff_expected = expected

    if canon_check == "browser_url":
        if expected is not None and str(expected).strip() != "":
            eff_expected = str(expected).strip()
            if (
                target_val.lower().startswith(("http://", "https://"))
                and exp_url is None
                and url_prefix is None
            ):
                url_prefix = target_val
        elif target_val and target_val.lower() not in ("active", "current") and not target_val.lower().startswith("tab:"):
            eff_expected = target_val
        else:
            return None, _invalid("Incomplete verification specification: 'browser_url' requires a target URL or expected URL.")

        exp_s = str(eff_expected).strip()
        if len(exp_s) > MAX_URL_LEN:
            return None, _invalid(f"Expected URL exceeds {MAX_URL_LEN} char limit.")
        if not exp_s.startswith("/"):
            ok_u, u_err = validate_browser_url(exp_s)
            if not ok_u:
                return None, _invalid(f"Invalid browser URL expectation: {u_err}")
        mode = "url"
    elif canon_check == "browser_origin":
        if expected is not None and str(expected).strip() != "":
            eff_expected = str(expected).strip()
        elif target_val and target_val.lower() not in ("active", "current") and not target_val.lower().startswith("tab:"):
            eff_expected = target_val
        else:
            return None, _invalid("Incomplete verification specification: 'browser_origin' requires an expected origin.")
        ok_o, o_err = validate_browser_url(str(eff_expected).strip())
        if not ok_o:
            return None, _invalid(f"Invalid browser origin expectation: {o_err}")
        mode = "origin"
    elif canon_check == "browser_title":
        if not target_val:
            return None, _invalid("Incomplete verification specification: 'browser_title' requires a target context or 'active'.")
        if expected is None or str(expected).strip() == "":
            return None, _invalid("Incomplete verification specification: 'browser_title' requires a non-empty expected title.")
        if target_val.lower().startswith(("http://", "https://")) and exp_url is None and url_prefix is None:
            url_prefix = target_val
        eff_expected = str(expected).strip()
        mode = "title"
    elif canon_check in (
        "browser_element_present",
        "browser_element_absent",
        "browser_visible_text",
        "browser_element_state",
        "browser_field_value",
    ):
        selector = str(kwargs.get("selector") or ctx.get("selector") or target_val).strip()
        if not selector:
            return None, _invalid(f"Incomplete verification specification: '{canon_check}' requires a non-empty CSS selector.")
        if len(selector) > MAX_SELECTOR_LEN:
            return None, _invalid(f"Selector exceeds {MAX_SELECTOR_LEN} char limit.")
        if _FORBIDDEN_BROWSER_SELECTOR_RE.search(selector):
            return None, _invalid("Blocked selector referencing cookies, browser storage, or executable script.")

        if canon_check == "browser_element_present":
            exp_low = "" if expected is None else str(expected).strip().lower()
            if exp_low not in ("", "visible", "present", "exists"):
                return None, _invalid(
                    f"Incompatible expectation '{expected}' for 'browser_element_present' (allowed: omitted or 'visible')."
                )
            mode = "element_present"
        elif canon_check == "browser_element_absent":
            if expected is not None and str(expected).strip() != "":
                return None, _invalid("Incompatible specification: 'browser_element_absent' does not accept an expected value.")
            mode = "element_absent"
        elif canon_check == "browser_visible_text":
            if expected is None or str(expected).strip() == "":
                return None, _invalid("Incomplete verification specification: 'browser_visible_text' requires expected text.")
            eff_expected = str(expected)
            mode = "visible_text"
        elif canon_check == "browser_element_state":
            parsed_state, st_err = _parse_browser_state_expectation(expected)
            if st_err:
                return None, _invalid(st_err)
            eff_expected = parsed_state
            mode = "element_state"
        else:
            if expected is None:
                return None, _invalid("Incomplete verification specification: 'browser_field_value' requires an expected value.")
            eff_expected = str(expected)
            mode = "field_value"
    else:
        return None, _invalid(f"Unknown browser verification check '{norm_check}'.")

    try:
        timeout = float(kwargs.get("timeout", 5.0))
    except (TypeError, ValueError):
        return None, _invalid(f"Invalid timeout for '{canon_check}': {kwargs.get('timeout')!r}.")

    criterion_obj["expected"] = eff_expected
    return {
        "canon_check": canon_check,
        "mode": mode,
        "selector": selector,
        "parsed_state": parsed_state,
        "eff_expected": eff_expected,
        "tab_id": tab_id,
        "exp_url": exp_url,
        "exp_origin": exp_origin,
        "url_prefix": url_prefix,
        "match_mode": match_mode,
        "timeout": timeout,
        "ctx": ctx,
        "criterion_obj": criterion_obj,
    }, None


async def verify_browser_outcome_async(
    check: str,
    target: str,
    expected: Optional[Any] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Verify read-only browser DOM/navigation state via Chromium Native Messaging."""
    from zelyr.tools.browser_native_host import NativeBrowserClient, validate_browser_url

    norm_raw = (check or "").strip().lower()
    target_val = "" if target is None else str(target).strip()
    validated, invalid_res = _validate_browser_params(norm_raw, target_val, expected, kwargs)
    if invalid_res is not None:
        return invalid_res
    assert validated is not None

    canon_check = validated["canon_check"]
    mode = validated["mode"]
    selector = validated["selector"]
    parsed_state = validated["parsed_state"]
    eff_expected = validated["eff_expected"]
    tab_id = validated["tab_id"]
    exp_url = validated["exp_url"]
    exp_origin = validated["exp_origin"]
    url_prefix = validated["url_prefix"]
    match_mode = validated["match_mode"]
    timeout = validated["timeout"]
    ctx = validated["ctx"]
    criterion_obj = validated["criterion_obj"]

    client = kwargs.get("browser_client")
    if client is None:
        sock_override = kwargs.get("browser_socket_path")
        env_sock = os.environ.get("ZELYR_BROWSER_NATIVE_SOCKET", "").strip()
        if not sock_override and not env_sock:
            return VerificationResult(
                success=False,
                check=canon_check,
                target=target_val,
                observed={"unverified": True, "evidence_collected": False, "error_type": "EXTENSION_DISCONNECTED"},
                error="Browser verification failed closed: Chromium Native Messaging socket is not configured (ZELYR_BROWSER_NATIVE_SOCKET unset).",
                result=VERDICT_UNVERIFIED,
                evidence_source="browser_native_messaging",
                expected=eff_expected,
                evidence_collected=False,
                criterion=criterion_obj,
            )
        client = NativeBrowserClient(socket_path=sock_override or env_sock)
        if not client.is_socket_present():
            return VerificationResult(
                success=False,
                check=canon_check,
                target=target_val,
                observed={"unverified": True, "evidence_collected": False, "error_type": "EXTENSION_DISCONNECTED"},
                error=f"Browser extension disconnected: Native Messaging socket not found ({client.socket_path}).",
                result=VERDICT_UNVERIFIED,
                evidence_source="browser_native_messaging",
                expected=eff_expected,
                evidence_collected=False,
                criterion=criterion_obj,
            )

    cmd_params: Dict[str, Any] = {"mode": mode}
    if selector is not None:
        cmd_params["selector"] = selector
    if tab_id is not None:
        cmd_params["tab_id"] = tab_id
    if exp_url:
        cmd_params["expected_url"] = exp_url
    if exp_origin:
        cmd_params["expected_origin"] = exp_origin
    if url_prefix:
        cmd_params["url_prefix"] = url_prefix

    native_res = await client.send_command("verify_dom_state", timeout=timeout, **cmd_params)
    resp = native_res.output if isinstance(native_res.output, dict) else {}

    if not native_res.success:
        if resp.get("invalid_selector"):
            return VerificationResult(
                success=False,
                check=canon_check,
                target=target_val,
                observed={"invalid_spec": True, "url": resp.get("url"), "selector": selector},
                error=native_res.error or f"Invalid CSS selector '{selector}'.",
                result=VERDICT_FAIL,
                evidence_source="browser_native_messaging",
                expected=eff_expected,
                evidence_collected=False,
                criterion=criterion_obj,
            )
        if resp.get("context_mismatch"):
            return VerificationResult(
                success=False,
                check=canon_check,
                target=target_val,
                observed={
                    "context_mismatch": True,
                    "tab_id": resp.get("tab_id"),
                    "url": resp.get("url"),
                    "origin": resp.get("origin"),
                    "title": resp.get("title"),
                },
                error=native_res.error or "Active browser tab context does not match requested URL/origin.",
                result=VERDICT_FAIL,
                evidence_source="browser_native_messaging",
                expected=eff_expected,
                evidence_collected=True,
                criterion=criterion_obj,
            )
        return VerificationResult(
            success=False,
            check=canon_check,
            target=target_val,
            observed={
                "unverified": True,
                "evidence_collected": False,
                "stale": bool(resp.get("stale", False)),
                "error_type": resp.get("error_type") or "EXTENSION_UNAVAILABLE",
                "url": resp.get("url"),
            },
            error=native_res.error or "Unable to collect browser verification evidence.",
            result=VERDICT_UNVERIFIED,
            evidence_source="browser_native_messaging",
            expected=eff_expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )

    obs_url = str(resp.get("url") or "")
    if obs_url and obs_url.lower() != "about:blank":
        ok_obs_url, obs_url_err = validate_browser_url(obs_url)
        if not ok_obs_url:
            return VerificationResult(
                success=False,
                check=canon_check,
                target=target_val,
                observed={"safety_blocked": True, "url": obs_url, "evidence_collected": False},
                error=f"Observed browser page URL rejected by security policy: {obs_url_err}",
                result=VERDICT_FAIL,
                evidence_source="browser_native_messaging",
                expected=eff_expected,
                evidence_collected=False,
                criterion=criterion_obj,
            )

    obs_origin = str(resp.get("origin") or "")
    if not obs_origin and obs_url.lower().startswith(("http://", "https://")):
        pu = urllib.parse.urlparse(obs_url)
        obs_origin = f"{pu.scheme.lower()}://{pu.netloc.lower()}"
    obs_title = str(resp.get("title") or "")
    tab_used = resp.get("tab_id")

    if canon_check == "browser_url":
        exp_s = str(eff_expected).strip()
        if exp_s.startswith("/"):
            p_obs = _normalize_http_url(obs_url)
            exp_path = "/" if exp_s == "/" else exp_s.rstrip("/")
            matched = bool(p_obs and p_obs[2] == exp_path)
        elif match_mode == "contains":
            matched = exp_s in obs_url
        elif match_mode == "prefix":
            matched = obs_url.startswith(exp_s)
        else:
            matched = (_normalize_http_url(obs_url) is not None and _normalize_http_url(obs_url) == _normalize_http_url(exp_s))
        obs_payload = {"tab_id": tab_used, "url": obs_url, "origin": obs_origin, "title": obs_title}
        return VerificationResult(
            success=matched,
            check=canon_check,
            target=target_val,
            observed=obs_payload,
            error=None if matched else f"Expected browser URL '{exp_s}', observed '{obs_url}'.",
            result=VERDICT_PASS if matched else VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=exp_s,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    if canon_check == "browser_origin":
        p_exp = urllib.parse.urlparse(str(eff_expected).strip())
        exp_orig = f"{p_exp.scheme.lower()}://{p_exp.netloc.lower()}"
        matched = obs_origin.lower() == exp_orig
        obs_payload = {"tab_id": tab_used, "url": obs_url, "origin": obs_origin}
        return VerificationResult(
            success=matched,
            check=canon_check,
            target=target_val,
            observed=obs_payload,
            error=None if matched else f"Expected browser origin '{exp_orig}', observed '{obs_origin}'.",
            result=VERDICT_PASS if matched else VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=exp_orig,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    if canon_check == "browser_title":
        exp_s = str(eff_expected).strip()
        matched = (obs_title == exp_s) if match_mode == "exact" else (exp_s in obs_title)
        obs_payload = {"tab_id": tab_used, "url": obs_url, "title": obs_title}
        return VerificationResult(
            success=matched,
            check=canon_check,
            target=target_val,
            observed=obs_payload,
            error=None if matched else f"Expected title containing '{exp_s}', observed '{obs_title}'.",
            result=VERDICT_PASS if matched else VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=exp_s,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    match_count = int(resp.get("match_count") or 0)
    visible_count = int(resp.get("visible_count") or 0)
    matches = resp.get("matches") if isinstance(resp.get("matches"), list) else []

    if canon_check == "browser_element_present":
        require_visible = str(expected or "").strip().lower() == "visible" or bool(kwargs.get("visible_only") or ctx.get("visible_only"))
        passed = (visible_count >= 1) if require_visible else (match_count >= 1)
        obs_payload = {
            "tab_id": tab_used,
            "url": obs_url,
            "selector": selector,
            "present": match_count >= 1,
            "match_count": match_count,
            "visible_count": visible_count,
        }
        err_msg = None if passed else (
            f"Element '{selector}' not visible (match_count={match_count}, visible_count={visible_count})."
            if require_visible
            else f"Element '{selector}' not found on page '{obs_url}'."
        )
        return VerificationResult(
            success=passed,
            check=canon_check,
            target=target_val,
            observed=obs_payload,
            error=err_msg,
            result=VERDICT_PASS if passed else VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=eff_expected,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    if canon_check == "browser_element_absent":
        passed = match_count == 0
        obs_payload = {
            "tab_id": tab_used,
            "url": obs_url,
            "selector": selector,
            "absent": passed,
            "match_count": match_count,
        }
        return VerificationResult(
            success=passed,
            check=canon_check,
            target=target_val,
            observed=obs_payload,
            error=None if passed else f"Element '{selector}' is unexpectedly present (match_count={match_count}).",
            result=VERDICT_PASS if passed else VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=None,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    if match_count == 0 or not matches:
        return VerificationResult(
            success=False,
            check=canon_check,
            target=target_val,
            observed={
                "tab_id": tab_used,
                "url": obs_url,
                "selector": selector,
                "match_count": 0,
                "visible_count": 0,
            },
            error=f"DOM element matching '{selector}' was not found on '{obs_url}'.",
            result=VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=eff_expected,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    first = matches[0] if isinstance(matches[0], dict) else {}

    if canon_check == "browser_visible_text":
        if any(isinstance(m, dict) and m.get("sensitive_redacted") for m in matches):
            return VerificationResult(
                success=False,
                check=canon_check,
                target=target_val,
                observed={"safety_blocked": True, "selector": selector, "url": obs_url},
                error=f"Refused to verify text content on sensitive credential field '{selector}'.",
                result=VERDICT_FAIL,
                evidence_source="browser_native_messaging",
                expected=eff_expected,
                evidence_collected=False,
                criterion=criterion_obj,
            )
        exp_txt = str(eff_expected)
        visible_texts = [
            str(m.get("visible_text") or "")
            for m in matches
            if isinstance(m, dict) and m.get("visible")
        ]
        if match_mode == "exact":
            matched = any(t == exp_txt for t in visible_texts)
        else:
            matched = any(exp_txt in t for t in visible_texts)
        obs_text = visible_texts[0] if visible_texts else ""
        obs_payload = {
            "tab_id": tab_used,
            "url": obs_url,
            "selector": selector,
            "visible": bool(first.get("visible")) if not visible_texts else True,
            "visible_text": obs_text,
            "match_count": match_count,
            "visible_count": visible_count,
        }
        err_msg = None
        if not matched:
            if visible_count == 0:
                err_msg = f"Element '{selector}' matched DOM but is hidden (visible_count=0)."
            else:
                err_msg = f"Expected visible text '{exp_txt}' on '{selector}', observed '{obs_text}'."
        return VerificationResult(
            success=matched,
            check=canon_check,
            target=target_val,
            observed=obs_payload,
            error=err_msg,
            result=VERDICT_PASS if matched else VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=exp_txt,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    if canon_check == "browser_field_value":
        if first.get("sensitive_redacted"):
            return VerificationResult(
                success=False,
                check=canon_check,
                target=target_val,
                observed={"safety_blocked": True, "selector": selector, "url": obs_url},
                error=f"Refused to inspect value of sensitive password/credential field '{selector}'.",
                result=VERDICT_FAIL,
                evidence_source="browser_native_messaging",
                expected=eff_expected,
                evidence_collected=False,
                criterion=criterion_obj,
            )
        obs_val = first.get("value")
        exp_val = str(eff_expected)
        if obs_val is None:
            matched = False
            err_msg = f"Element '{selector}' (tag={first.get('tag')}) does not expose a form value."
        elif match_mode == "contains":
            matched = exp_val in str(obs_val)
            err_msg = None if matched else f"Expected field '{selector}' value containing '{exp_val}', observed '{obs_val}'."
        else:
            matched = str(obs_val) == exp_val
            err_msg = None if matched else f"Expected field '{selector}' value '{exp_val}', observed '{obs_val}'."
        return VerificationResult(
            success=matched,
            check=canon_check,
            target=target_val,
            observed={"tab_id": tab_used, "url": obs_url, "selector": selector, "value": obs_val, "tag": first.get("tag")},
            error=err_msg,
            result=VERDICT_PASS if matched else VERDICT_FAIL,
            evidence_source="browser_native_messaging",
            expected=exp_val,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    # browser_element_state
    assert parsed_state is not None
    state_mismatches: List[str] = []
    obs_state: Dict[str, Any] = {
        "visible": first.get("visible"),
        "disabled": first.get("disabled"),
        "checked": first.get("checked"),
        "selected": first.get("selected"),
        "focused": first.get("focused"),
        "value": first.get("value") if not first.get("sensitive_redacted") else None,
        "selected_options": first.get("selected_options"),
    }
    for attr_name, exp_attr_val in parsed_state.items():
        if attr_name == "selected_option":
            sel_opts = first.get("selected_options") or []
            opt_vals = [str(o.get("value", "")) for o in sel_opts if isinstance(o, dict)]
            opt_texts = [str(o.get("text", "")) for o in sel_opts if isinstance(o, dict)]
            if str(exp_attr_val) not in opt_vals and str(exp_attr_val) not in opt_texts and str(first.get("value") or "") != str(exp_attr_val):
                state_mismatches.append(f"selected_option={exp_attr_val!r} (observed options={sel_opts})")
        else:
            actual_val = first.get(attr_name)
            if actual_val is None or bool(actual_val) != bool(exp_attr_val):
                state_mismatches.append(f"{attr_name}={exp_attr_val} (observed {actual_val})")

    passed = not state_mismatches
    return VerificationResult(
        success=passed,
        check=canon_check,
        target=target_val,
        observed={"tab_id": tab_used, "url": obs_url, "selector": selector, "state": obs_state},
        error=None if passed else f"Element '{selector}' state mismatch: {'; '.join(state_mismatches)}.",
        result=VERDICT_PASS if passed else VERDICT_FAIL,
        evidence_source="browser_native_messaging",
        expected=parsed_state,
        evidence_collected=True,
        criterion=criterion_obj,
    )


def _validate_desktop_params(
    norm_check: str,
    target_val: str,
    expected: Optional[Any],
    kwargs: Dict[str, Any],
) -> Union[Tuple[None, VerificationResult], Tuple[Dict[str, Any], None]]:
    """Pure parameter/selector/target validation for desktop verification checks."""
    from zelyr.engine.desktop_accessibility import (
        MAX_RETURNED_TEXT_LEN,
        parse_desktop_selector,
        split_window_and_widget_target,
    )

    ctx = kwargs.get("context") if isinstance(kwargs.get("context"), dict) else {}
    ev_src = "hyprland_ipc" if norm_check in DESKTOP_COMPOSITOR_CHECKS else "desktop_accessibility"
    criterion_obj: Dict[str, Any] = {"check": norm_check, "target": target_val, "expected": expected}

    def _invalid(msg: str) -> VerificationResult:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={"invalid_spec": True},
            error=msg,
            result=VERDICT_FAIL,
            evidence_source=ev_src,
            expected=expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )

    if "\x00" in target_val or (isinstance(expected, str) and "\x00" in expected):
        return None, _invalid("Invalid desktop verification specification: null byte is not permitted.")

    if norm_check in DESKTOP_COMPOSITOR_CHECKS:
        if not target_val:
            return None, _invalid(
                f"Incomplete verification specification: '{norm_check}' requires a non-empty target window identifier."
            )
        if norm_check in ("desktop_window_exists", "desktop_active_window"):
            if expected is not None and str(expected).strip() != "":
                return None, _invalid(
                    f"Incompatible specification: '{norm_check}' does not accept expected text (use 'desktop_window_title' for title checks)."
                )
        elif expected is None or str(expected).strip() == "":
            return None, _invalid("Incomplete verification specification: 'desktop_window_title' requires an expected title.")
        return {
            "norm_check": norm_check,
            "window_target": target_val,
            "selector": None,
            "parsed_state": None,
            "eff_expected": expected,
            "criterion_obj": criterion_obj,
            "match_mode": str(kwargs.get("match") or ctx.get("match") or "").strip().lower(),
        }, None

    if norm_check not in DESKTOP_ACCESSIBILITY_CHECKS:
        return None, _invalid(f"Unknown desktop verification check '{norm_check}'.")

    explicit_window = kwargs.get("window") or ctx.get("window")
    explicit_selector = kwargs.get("selector") or ctx.get("selector")
    window_target, raw_selector = split_window_and_widget_target(
        target_val,
        explicit_window=str(explicit_window) if explicit_window is not None else None,
        explicit_selector=str(explicit_selector) if explicit_selector is not None else None,
    )
    if not window_target or window_target == "*" or len(window_target) > 256:
        return None, _invalid(
            f"Invalid desktop target window '{window_target}' for '{norm_check}': must be a specific non-empty window identifier."
        )
    if len(window_target) < 2 and not window_target.lower().startswith(("0x", "address:")):
        return None, _invalid(f"Ambiguous window target '{window_target}' is too short for deterministic identification.")

    sel_obj, sel_err = parse_desktop_selector(raw_selector)
    if sel_err or sel_obj is None:
        return None, _invalid(sel_err or f"Invalid widget selector for '{norm_check}'.")

    criterion_obj = {
        "check": norm_check,
        "target": target_val,
        "window": window_target,
        "selector": sel_obj.raw,
        "expected": expected,
    }

    parsed_state: Optional[Dict[str, Any]] = None
    eff_expected: Any = expected
    if norm_check in ("desktop_widget_state", "desktop_ui_state"):
        parsed_state, st_err = _parse_desktop_state_expectation(expected, check_name=norm_check)
        if st_err:
            return None, _invalid(st_err)
        eff_expected = parsed_state
    else:
        if expected is None or str(expected).strip() == "":
            return None, _invalid(f"Incomplete verification specification: '{norm_check}' requires expected visible text.")
        exp_str = str(expected)
        if len(exp_str) > MAX_RETURNED_TEXT_LEN:
            return None, _invalid(f"Expected text exceeds {MAX_RETURNED_TEXT_LEN} char limit.")
        eff_expected = exp_str

    criterion_obj["expected"] = eff_expected
    return {
        "norm_check": norm_check,
        "window_target": window_target,
        "selector": sel_obj,
        "parsed_state": parsed_state,
        "eff_expected": eff_expected,
        "criterion_obj": criterion_obj,
        "match_mode": str(kwargs.get("match") or ctx.get("match") or "").strip().lower(),
    }, None


async def _verify_desktop_accessibility_async(
    validated: Dict[str, Any],
    target_val: str,
    kwargs: Dict[str, Any],
) -> VerificationResult:
    """Execute narrowly scoped read-only desktop semantic verification via AT-SPI2."""
    from zelyr.engine.desktop_accessibility import (
        DEFAULT_DBUS_TIMEOUT,
        is_widget_checkable,
        query_desktop_widget_via_accessibility,
    )

    norm_check = validated["norm_check"]
    window_target = validated["window_target"]
    sel_obj = validated["selector"]
    parsed_state = validated["parsed_state"]
    eff_expected = validated["eff_expected"]
    criterion_obj = validated["criterion_obj"]
    match_mode = validated["match_mode"]

    legacy_reader = kwargs.get("desktop_accessibility_reader")
    if callable(legacy_reader):
        try:
            acc_res = legacy_reader(
                target=target_val,
                window=window_target,
                selector=sel_obj.raw,
                expected=eff_expected,
                check=norm_check,
            )
            if asyncio.iscoroutine(acc_res):
                acc_res = await acc_res
            if isinstance(acc_res, VerificationResult):
                return acc_res
        except Exception as acc_exc:
            return VerificationResult(
                success=False,
                check=norm_check,
                target=target_val,
                observed={"unverified": True, "evidence_collected": False},
                error=f"Desktop accessibility query failed: {acc_exc}",
                result=VERDICT_UNVERIFIED,
                evidence_source="desktop_accessibility",
                expected=eff_expected,
                evidence_collected=False,
                criterion=criterion_obj,
            )

    try:
        timeout = float(kwargs.get("timeout", DEFAULT_DBUS_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_DBUS_TIMEOUT

    qres = await query_desktop_widget_via_accessibility(
        window_target=window_target,
        selector=sel_obj,
        backend=kwargs.get("desktop_accessibility_backend"),
        hyprland_client=kwargs.get("hyprland_client"),
        hyprland_socket_path=kwargs.get("hyprland_socket_path"),
        timeout=timeout,
    )

    if qres.unsupported:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={
                "unsupported": True,
                "accessibility_backend": qres.backend,
                "evidence_collected": False,
                "focus_ipc_not_semantic": True,
                "reason": qres.error or "AT-SPI2 accessibility backend unavailable.",
            },
            error=qres.error or f"Unsupported verification criterion '{norm_check}': AT-SPI2 backend unavailable.",
            result=VERDICT_UNSUPPORTED,
            evidence_source="desktop_accessibility",
            expected=eff_expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )

    if qres.ambiguous or qres.inaccessible:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={
                "unverified": True,
                "accessibility_backend": qres.backend,
                "evidence_collected": False,
                "window_found": qres.window_found,
                "window_accessible": qres.window_accessible,
                "widget_found": qres.widget_found,
                "widget_accessible": qres.widget_accessible,
                "window_title": qres.window_title,
                "window_pid": qres.window_pid,
                "ambiguous": qres.ambiguous,
                "inaccessible": qres.inaccessible,
                "truncated": qres.truncated,
                "inspected_descendants": qres.inspected_descendants,
            },
            error=qres.error or "Target desktop widget could not be unambiguously inspected.",
            result=VERDICT_UNVERIFIED,
            evidence_source="desktop_accessibility",
            expected=eff_expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )

    if not qres.window_found:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={
                "window_found": False,
                "accessibility_backend": qres.backend,
                "window_target": window_target,
                "selector": sel_obj.raw,
            },
            error=qres.error or f"Target desktop window '{window_target}' was not found.",
            result=VERDICT_FAIL,
            evidence_source="desktop_accessibility",
            expected=eff_expected,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    if not qres.widget_found or qres.widget is None:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={
                "window_found": True,
                "window_accessible": True,
                "widget_found": False,
                "widget_accessible": qres.widget_accessible,
                "accessibility_backend": qres.backend,
                "window_title": qres.window_title,
                "window_pid": qres.window_pid,
                "selector": sel_obj.raw,
                "inspected_descendants": qres.inspected_descendants,
            },
            error=qres.error or f"Widget matching '{sel_obj.raw}' was not found inside '{qres.window_title}'.",
            result=VERDICT_FAIL,
            evidence_source="desktop_accessibility",
            expected=eff_expected,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    w = qres.widget
    if w.sensitive_redacted:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={
                "safety_blocked": True,
                "accessibility_backend": qres.backend,
                "window_title": w.window_title,
                "window_pid": w.window_pid,
                "selector": sel_obj.raw,
                "role": w.role,
            },
            error=f"Refused to verify or expose sensitive/password desktop widget matching '{sel_obj.raw}'.",
            result=VERDICT_FAIL,
            evidence_source="desktop_accessibility",
            expected=eff_expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )

    base_widget_obs = {
        "accessibility_backend": qres.backend,
        "window_found": True,
        "window_accessible": True,
        "widget_found": True,
        "widget_accessible": True,
        "window_title": w.window_title,
        "window_pid": w.window_pid,
        "selector": sel_obj.raw,
        "inspected_descendants": qres.inspected_descendants,
        "widget": {
            "role": w.role,
            "accessible_id": w.accessible_id,
            "name": w.name,
            "visible": bool(w.states.get("visible") and w.states.get("showing")),
        },
    }

    if norm_check in ("desktop_widget_state", "desktop_ui_state"):
        assert parsed_state is not None
        mismatches: List[str] = []
        reported_states = {
            k: bool(w.states.get(k))
            for k in (
                "enabled", "disabled", "checked", "selected", "focused",
                "visible", "showing", "editable", "read_only", "expanded",
                "active", "indeterminate",
            )
        }
        base_widget_obs["widget"]["states"] = reported_states

        for attr_name, exp_bool in parsed_state.items():
            if attr_name == "checked" and not is_widget_checkable(w):
                mismatches.append(f"widget role '{w.role}' is not checkable")
                continue
            if attr_name == "visible":
                actual = bool(w.states.get("visible") and w.states.get("showing"))
            else:
                actual = bool(w.states.get(attr_name))
            if actual != bool(exp_bool):
                mismatches.append(f"{attr_name}={exp_bool} (observed {actual})")

        passed = not mismatches
        return VerificationResult(
            success=passed,
            check=norm_check,
            target=target_val,
            observed=base_widget_obs,
            error=None if passed else f"Desktop widget '{sel_obj.raw}' state mismatch: {'; '.join(mismatches)}.",
            result=VERDICT_PASS if passed else VERDICT_FAIL,
            evidence_source="desktop_accessibility",
            expected=parsed_state,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    # desktop_visible_text / desktop_text / desktop_status
    is_visible = bool(w.states.get("visible") and w.states.get("showing"))
    obs_text = w.visible_text
    base_widget_obs["widget"]["visible_text"] = obs_text
    if not is_visible:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed=base_widget_obs,
            error=f"Desktop widget '{sel_obj.raw}' inside '{w.window_title}' is hidden/not showing.",
            result=VERDICT_FAIL,
            evidence_source="desktop_accessibility",
            expected=eff_expected,
            evidence_collected=True,
            criterion=criterion_obj,
        )

    exp_s = str(eff_expected)
    passed = (obs_text == exp_s) if match_mode == "exact" else (exp_s in obs_text)
    return VerificationResult(
        success=passed,
        check=norm_check,
        target=target_val,
        observed=base_widget_obs,
        error=(
            None
            if passed
            else f"Expected widget '{sel_obj.raw}' text containing '{exp_s}', observed '{obs_text}'."
        ),
        result=VERDICT_PASS if passed else VERDICT_FAIL,
        evidence_source="desktop_accessibility",
        expected=exp_s,
        evidence_collected=True,
        criterion=criterion_obj,
    )


async def verify_desktop_outcome_async(
    check: str,
    target: str,
    expected: Optional[Any] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Verify read-only Linux desktop compositor or AT-SPI2 accessibility UI state."""
    from zelyr.engine.hyprland_ipc import HyprlandIPCClient, HyprlandUnavailableError

    norm_check = (check or "").strip().lower()
    target_val = "" if target is None else str(target).strip()
    validated, invalid_res = _validate_desktop_params(norm_check, target_val, expected, kwargs)
    if invalid_res is not None:
        return invalid_res
    assert validated is not None

    if norm_check in DESKTOP_ACCESSIBILITY_CHECKS:
        return await _verify_desktop_accessibility_async(validated, target_val, kwargs)

    match_mode = validated["match_mode"]
    criterion_obj = validated["criterion_obj"]

    hypr = kwargs.get("hyprland_client")
    if hypr is None:
        hypr = HyprlandIPCClient(socket_path=kwargs.get("hyprland_socket_path"))

    can_conn = getattr(hypr, "can_connect", None)
    if callable(can_conn) and not can_conn():
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={"unverified": True, "evidence_collected": False, "error_type": "COMPOSITOR_UNAVAILABLE"},
            error="Hyprland compositor IPC is unavailable; desktop window state cannot be verified.",
            result=VERDICT_UNVERIFIED,
            evidence_source="hyprland_ipc",
            expected=expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )

    try:
        if norm_check == "desktop_active_window":
            active_win = await hypr.get_active_window()
            if not isinstance(active_win, dict) or not active_win:
                return VerificationResult(
                    success=False,
                    check=norm_check,
                    target=target_val,
                    observed={
                        "active_matches_target": False,
                        "focus_only": True,
                        "semantic_outcome_verified": False,
                    },
                    error=f"No focused window is active; expected '{target_val}'.",
                    result=VERDICT_FAIL,
                    evidence_source="hyprland_ipc",
                    expected=target_val,
                    evidence_collected=True,
                    criterion=criterion_obj,
                )
            matched = HyprlandIPCClient._match_client([active_win], target_val)
            if matched is None:
                # Privacy guard: do not dump arbitrary unrelated private user window titles.
                return VerificationResult(
                    success=False,
                    check=norm_check,
                    target=target_val,
                    observed={
                        "active_matches_target": False,
                        "focus_only": True,
                        "semantic_outcome_verified": False,
                    },
                    error=f"Focused compositor window does not match target '{target_val}'.",
                    result=VERDICT_FAIL,
                    evidence_source="hyprland_ipc",
                    expected=target_val,
                    evidence_collected=True,
                    criterion=criterion_obj,
                )
            ws_obj = matched.get("workspace") if isinstance(matched.get("workspace"), dict) else {}
            return VerificationResult(
                success=True,
                check=norm_check,
                target=target_val,
                observed={
                    "active_matches_target": True,
                    "focus_only": True,
                    "semantic_outcome_verified": False,
                    "window": {
                        "address": matched.get("address"),
                        "title": matched.get("title"),
                        "class": matched.get("class") or matched.get("initialClass"),
                        "workspace_id": ws_obj.get("id"),
                    },
                },
                error=None,
                result=VERDICT_PASS,
                evidence_source="hyprland_ipc",
                expected=target_val,
                evidence_collected=True,
                criterion=criterion_obj,
            )

        clients = await hypr.get_clients()
        if not isinstance(clients, list):
            raise HyprlandUnavailableError("Invalid compositor client response.")
        matched = HyprlandIPCClient._match_client(clients, target_val)
        # Privacy guard: inspect only the target-matching client; never leak unrelated user windows.
        win_info = (
            {
                "address": matched.get("address"),
                "title": matched.get("title"),
                "class": matched.get("class") or matched.get("initialClass"),
                "workspace_id": (matched.get("workspace") or {}).get("id") if isinstance(matched.get("workspace"), dict) else None,
                "mapped": matched.get("mapped"),
            }
            if matched is not None
            else None
        )
        if norm_check == "desktop_window_exists":
            passed = matched is not None
            return VerificationResult(
                success=passed,
                check=norm_check,
                target=target_val,
                observed={"exists": passed, "window": win_info},
                error=None if passed else f"No open desktop window matching '{target_val}'.",
                result=VERDICT_PASS if passed else VERDICT_FAIL,
                evidence_source="hyprland_ipc",
                expected=True,
                evidence_collected=True,
                criterion=criterion_obj,
            )

        # desktop_window_title
        exp_title = str(expected).strip()
        if matched is None:
            return VerificationResult(
                success=False,
                check=norm_check,
                target=target_val,
                observed={"exists": False, "window": None},
                error=f"Target window '{target_val}' was not found to verify title '{exp_title}'.",
                result=VERDICT_FAIL,
                evidence_source="hyprland_ipc",
                expected=exp_title,
                evidence_collected=True,
                criterion=criterion_obj,
            )
        obs_title = str(matched.get("title") or "")
        passed = (obs_title == exp_title) if match_mode == "exact" else (exp_title in obs_title)
        return VerificationResult(
            success=passed,
            check=norm_check,
            target=target_val,
            observed={"exists": True, "title": obs_title, "window": win_info},
            error=None if passed else f"Expected window title containing '{exp_title}', observed '{obs_title}'.",
            result=VERDICT_PASS if passed else VERDICT_FAIL,
            evidence_source="hyprland_ipc",
            expected=exp_title,
            evidence_collected=True,
            criterion=criterion_obj,
        )
    except (HyprlandUnavailableError, OSError, asyncio.TimeoutError) as exc:
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={"unverified": True, "evidence_collected": False, "error_type": "COMPOSITOR_UNAVAILABLE"},
            error=f"Hyprland compositor IPC query failed: {exc}",
            result=VERDICT_UNVERIFIED,
            evidence_source="hyprland_ipc",
            expected=expected,
            evidence_collected=False,
            criterion=criterion_obj,
        )


def verify_browser_outcome(
    check: str,
    target: str,
    expected: Optional[Any] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Synchronous wrapper around verify_browser_outcome_async."""
    return _run_coro_sync(lambda: verify_browser_outcome_async(check, target, expected, **kwargs))


def verify_desktop_outcome(
    check: str,
    target: str,
    expected: Optional[Any] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Synchronous wrapper around verify_desktop_outcome_async."""
    return _run_coro_sync(lambda: verify_desktop_outcome_async(check, target, expected, **kwargs))


def _with_expected_meta(vr: VerificationResult, expected: Any) -> VerificationResult:
    """Populate expected metadata on sync VerificationResult instances when caller supplied expected."""
    if expected is None or vr.expected is not None:
        return vr
    return VerificationResult(
        success=vr.success,
        check=vr.check,
        target=vr.target,
        observed=vr.observed,
        error=vr.error,
        result=vr.result,
        evidence_source=vr.evidence_source,
        expected=expected,
        evidence_collected=vr.evidence_collected,
        criterion={**vr.criterion, "expected": expected},
    )


def _dispatch_sync_check(
    norm_check: str,
    raw_check: str,
    target_val: str,
    expected: Optional[Any],
    kwargs: Dict[str, Any],
) -> Optional[VerificationResult]:
    """Dispatch synchronous filesystem, process, port, and allowlisted command checks."""
    cwd = kwargs.get("cwd")
    allow_external = bool(kwargs.get("allow_external", False))

    if norm_check in ("file_exists", "exists"):
        if expected is not None and str(expected).strip() != "":
            return VerificationResult(
                success=False,
                check="file_exists",
                target=target_val,
                observed={"expected_content_provided": expected, "invalid_spec": True},
                error=(
                    "Incompatible verification specification: 'file_exists' only verifies file existence "
                    "and does not accept expected content. Use 'file_contains:target:expected' instead."
                ),
            )
        return verify_file_exists(target_val, cwd=cwd, allow_external=allow_external)
    if norm_check in ("file_contains", "contains"):
        if expected is None or str(expected) == "":
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_val,
                observed={"pattern": expected, "invalid_spec": True},
                error="Incomplete verification specification: 'file_contains' requires expected pattern or content.",
            )
        regex = bool(kwargs.get("regex", False))
        return _with_expected_meta(
            verify_file_contains(target_val, expected, regex=regex, cwd=cwd, allow_external=allow_external),
            expected,
        )
    if norm_check in ("file_checksum", "checksum", "sha256"):
        return _with_expected_meta(
            verify_file_checksum(target_val, expected or "", cwd=cwd, allow_external=allow_external),
            expected,
        )
    if norm_check in ("file_json", "json"):
        schema = kwargs.get("schema")
        if schema is None and expected is not None and str(expected).strip() != "":
            exp_str = str(expected).strip()
            if exp_str.startswith("{"):
                try:
                    schema = json.loads(exp_str)
                except Exception as exc:
                    return VerificationResult(
                        success=False,
                        check="file_json",
                        target=target_val,
                        observed={"valid": False, "invalid_spec": True},
                        error=f"Invalid schema JSON provided for 'file_json': {exc}",
                    )
            else:
                schema = {"required_keys": [k.strip() for k in exp_str.split(",") if k.strip()]}
        return _with_expected_meta(
            verify_file_json(target_val, schema=schema, cwd=cwd, allow_external=allow_external),
            expected,
        )
    if norm_check in ("process_running", "process", "proc"):
        if expected is not None and str(expected).strip() != "":
            return VerificationResult(
                success=False,
                check="process_running",
                target=target_val,
                observed={"expected_content_provided": expected, "invalid_spec": True},
                error="Incompatible verification specification: 'process_running' only verifies process existence and does not accept expected content.",
            )
        return verify_process_running(target_val)
    if norm_check in ("port_listening", "port", "listen"):
        target_spec = target_val
        if expected and str(expected).strip():
            target_clean = target_val.strip()
            exp_clean = str(expected).strip()
            if ":" not in target_clean and not target_clean.isdigit() and exp_clean.isdigit():
                target_spec = f"{target_clean}:{exp_clean}"
            elif target_clean.isdigit() and not exp_clean.isdigit():
                target_spec = f"{exp_clean}:{target_clean}"
        host = kwargs.get("host", "127.0.0.1")
        try:
            timeout = float(kwargs.get("timeout", 0.5))
        except (TypeError, ValueError):
            return VerificationResult(
                success=False,
                check="port_listening",
                target=target_val,
                observed={"invalid_spec": True},
                error=f"Invalid timeout for 'port_listening': {kwargs.get('timeout')!r}.",
            )
        return _with_expected_meta(
            verify_port_listening(target_spec, host=host, timeout=timeout),
            expected,
        )
    if norm_check in ("command_output", "command", "cmd"):
        try:
            timeout = float(kwargs.get("timeout", 5.0))
        except (TypeError, ValueError):
            return VerificationResult(
                success=False,
                check="command_output",
                target=target_val,
                observed={"invalid_spec": True},
                error=f"Invalid timeout for 'command_output': {kwargs.get('timeout')!r}.",
            )
        cancel_event = kwargs.get("cancel_event")
        return _with_expected_meta(
            verify_command_output(target_val, expected or "", timeout=timeout, cwd=cwd, cancel_event=cancel_event),
            expected,
        )
    return None


def _validate_base_dispatch_args(check: str, target: str) -> Optional[VerificationResult]:
    if not isinstance(check, str) or not check.strip():
        return VerificationResult(
            success=False,
            check=str(check or ""),
            target="" if target is None else str(target),
            observed={"invalid_spec": True},
            error="Incomplete verification specification: 'check' must be a non-empty check name.",
        )
    if target is None or not isinstance(target, (str, Path)):
        return VerificationResult(
            success=False,
            check=check.strip().lower(),
            target="" if target is None else str(target),
            observed={"invalid_spec": True},
            error="Invalid verification specification: 'target' must be a string or Path.",
        )
    return None


async def verify_outcome_async(
    check: str,
    target: str,
    expected: Optional[Any] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Async outcome verification dispatcher supporting filesystem, host, browser, and desktop checks."""
    invalid_base = _validate_base_dispatch_args(check, target)
    if invalid_base is not None:
        return invalid_base

    norm_check = check.strip().lower()
    target_val = str(target)

    if norm_check in BROWSER_VERIFICATION_CHECKS:
        return await verify_browser_outcome_async(norm_check, target_val, expected=expected, **kwargs)
    if norm_check in DESKTOP_VERIFICATION_CHECKS:
        return await verify_desktop_outcome_async(norm_check, target_val, expected=expected, **kwargs)

    sync_res = _dispatch_sync_check(norm_check, check, target_val, expected, kwargs)
    if sync_res is not None:
        return sync_res

    return VerificationResult(
        success=False,
        check=check,
        target=target_val,
        observed={"invalid_spec": True},
        error=(
            f"Unknown verification check '{check}'. Supported checks: "
            f"{', '.join(sorted(ALL_KNOWN_VERIFICATION_CHECKS))}."
        ),
    )


def verify_outcome(
    check: str,
    target: str,
    expected: Optional[Any] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Dispatch outcome verification based on check name."""
    invalid_base = _validate_base_dispatch_args(check, target)
    if invalid_base is not None:
        return invalid_base

    norm_check = check.strip().lower()
    target_val = str(target)

    if norm_check in BROWSER_VERIFICATION_CHECKS:
        return verify_browser_outcome(norm_check, target_val, expected=expected, **kwargs)
    if norm_check in DESKTOP_VERIFICATION_CHECKS:
        return verify_desktop_outcome(norm_check, target_val, expected=expected, **kwargs)

    sync_res = _dispatch_sync_check(norm_check, check, target_val, expected, kwargs)
    if sync_res is not None:
        return sync_res

    return VerificationResult(
        success=False,
        check=check,
        target=target_val,
        observed={"invalid_spec": True},
        error=(
            f"Unknown verification check '{check}'. Supported checks: "
            f"{', '.join(sorted(ALL_KNOWN_VERIFICATION_CHECKS))}."
        ),
    )


def validate_criterion_spec(
    criterion: Dict[str, Any],
    **kwargs: Any,
) -> Optional[VerificationResult]:
    """Validate a single parsed criterion statically before task execution (no external I/O on valid specs)."""
    if not isinstance(criterion, dict):
        return VerificationResult(
            success=False,
            check="",
            target="",
            observed={"invalid_spec": True},
            error="Verification criterion must be an object.",
        )
    check = criterion.get("check", "")
    target = criterion.get("target", "")
    expected = criterion.get("expected")
    invalid_base = _validate_base_dispatch_args(check, target)
    if invalid_base is not None:
        return invalid_base

    norm_check = str(check).strip().lower()
    target_val = str(target).strip()

    if norm_check not in ALL_KNOWN_VERIFICATION_CHECKS:
        return VerificationResult(
            success=False,
            check=str(check),
            target=target_val,
            observed={"invalid_spec": True},
            error=(
                f"Unknown verification check '{check}'. Supported checks: "
                f"{', '.join(sorted(ALL_KNOWN_VERIFICATION_CHECKS))}."
            ),
        )

    if "\x00" in target_val or (isinstance(expected, str) and "\x00" in expected):
        return VerificationResult(
            success=False,
            check=norm_check,
            target=target_val,
            observed={"invalid_spec": True},
            error="Invalid verification specification: null byte is not permitted.",
        )

    if norm_check in BROWSER_VERIFICATION_CHECKS:
        merged_kw = dict(kwargs)
        if isinstance(criterion.get("context"), dict):
            merged_kw.setdefault("context", criterion["context"])
        _, invalid_res = _validate_browser_params(norm_check, target_val, expected, merged_kw)
        return invalid_res

    if norm_check in DESKTOP_VERIFICATION_CHECKS:
        merged_kw = dict(kwargs)
        if isinstance(criterion.get("context"), dict):
            merged_kw.setdefault("context", criterion["context"])
        _, invalid_res = _validate_desktop_params(norm_check, target_val, expected, merged_kw)
        return invalid_res

    if norm_check in ("file_exists", "exists"):
        if not target_val:
            return VerificationResult(
                success=False,
                check="file_exists",
                target="",
                observed={"exists": False, "invalid_spec": True},
                error="Incomplete verification specification: 'file_exists' requires a non-empty target path.",
            )
        if expected is not None and str(expected).strip() != "":
            return VerificationResult(
                success=False,
                check="file_exists",
                target=target_val,
                observed={"expected_content_provided": expected, "invalid_spec": True},
                error="Incompatible verification specification: 'file_exists' does not accept expected content.",
            )
        return None
    if norm_check in ("file_contains", "contains"):
        if not target_val:
            return VerificationResult(
                success=False,
                check="file_contains",
                target="",
                observed={"matched": False, "invalid_spec": True},
                error="Incomplete verification specification: 'file_contains' requires a non-empty target path.",
            )
        if expected is None or str(expected) == "":
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_val,
                observed={"pattern": expected, "invalid_spec": True},
                error="Incomplete verification specification: 'file_contains' requires expected pattern or content.",
            )
        return None
    if norm_check in ("file_checksum", "checksum", "sha256"):
        if not target_val or expected is None or not str(expected).strip():
            return VerificationResult(
                success=False,
                check="file_checksum",
                target=target_val,
                observed={"invalid_spec": True},
                error="Incomplete verification specification: 'file_checksum' requires target path and expected checksum.",
            )
        return None
    if norm_check in ("file_json", "json"):
        if not target_val:
            return VerificationResult(
                success=False,
                check="file_json",
                target="",
                observed={"valid": False, "invalid_spec": True},
                error="Incomplete verification specification: 'file_json' requires a non-empty target path.",
            )
        if expected is not None and str(expected).strip().startswith("{"):
            try:
                json.loads(str(expected).strip())
            except Exception as exc:
                return VerificationResult(
                    success=False,
                    check="file_json",
                    target=target_val,
                    observed={"valid": False, "invalid_spec": True},
                    error=f"Invalid schema JSON provided for 'file_json': {exc}",
                )
        return None
    if norm_check in ("process_running", "process", "proc"):
        if not target_val or (expected is not None and str(expected).strip() != ""):
            return VerificationResult(
                success=False,
                check="process_running",
                target=target_val,
                observed={"invalid_spec": True},
                error="Invalid specification for 'process_running': requires non-empty pattern and no expected value.",
            )
        return None
    if norm_check in ("port_listening", "port", "listen"):
        port_s = str(expected).strip() if (expected and ":" not in target_val and not target_val.isdigit()) else target_val.split(":")[-1].strip()
        try:
            pnum = int(port_s)
            if not (1 <= pnum <= 65535):
                raise ValueError("out of range")
        except Exception:
            return VerificationResult(
                success=False,
                check="port_listening",
                target=target_val,
                observed={"invalid_spec": True},
                error=f"Invalid port specification for 'port_listening': '{target_val}'.",
            )
        return None
    if norm_check in ("command_output", "command", "cmd"):
        if not target_val or expected is None or str(expected) == "":
            return VerificationResult(
                success=False,
                check="command_output",
                target=target_val,
                observed={"invalid_spec": True},
                error="Incomplete verification specification: 'command_output' requires command and expected output.",
            )
        return None
    return None


def validate_verification_spec(
    spec: Any,
    *,
    parser_fn: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[VerificationResult]]:
    """Parse and statically validate a --verify / acceptance_criteria spec before execution."""
    if spec is None:
        return [], "Empty verification specification provided.", None
    if isinstance(spec, str):
        stripped = spec.strip()
        if not stripped:
            return [], "Empty verification specification provided.", None
        if stripped.startswith(("[", "{")):
            try:
                decoded = json.loads(stripped)
            except Exception as exc:
                return [], f"Malformed JSON verification specification: {exc}", None
            if isinstance(decoded, list) and not decoded:
                return [], "Empty verification specification list provided.", None
            if not isinstance(decoded, (list, dict)):
                return [], "JSON verification specification must be an object or array of objects.", None

    if parser_fn is None:
        from zelyr.engine.execution_ledger import parse_acceptance_criteria as parser_fn

    parsed = parser_fn(spec)
    if not parsed:
        return [], f"Unable to parse verification specification: '{spec}'", None

    for item in parsed:
        err_res = validate_criterion_spec(item)
        if err_res is not None:
            return parsed, f"{err_res.check} on {err_res.target}: FAIL ({err_res.error})", err_res
    return parsed, None, None


def summarize_verification_scope(
    criteria: Optional[List[Dict[str, Any]]],
    v_results: Optional[List[Tuple[str, str, VerificationResult]]] = None,
) -> Dict[str, Any]:
    """Summarize verified surfaces, verdict counts, and evidence attribution cleanly."""
    surfaces: List[str] = []
    checks_to_inspect: List[str] = []
    if v_results:
        checks_to_inspect.extend(str(chk or "").strip().lower() for chk, _, _ in v_results)
    elif criteria:
        checks_to_inspect.extend(str(c.get("check") or "").strip().lower() for c in criteria if isinstance(c, dict))

    for chk in checks_to_inspect:
        if chk in BROWSER_VERIFICATION_CHECKS and "browser" not in surfaces:
            surfaces.append("browser")
        elif chk in DESKTOP_COMPOSITOR_CHECKS and "desktop_compositor" not in surfaces:
            surfaces.append("desktop_compositor")
        elif chk in DESKTOP_ACCESSIBILITY_CHECKS and "desktop_accessibility" not in surfaces:
            surfaces.append("desktop_accessibility")
        elif chk in ("file_exists", "exists", "file_contains", "contains", "file_checksum", "checksum", "sha256", "file_json", "json"):
            if "filesystem" not in surfaces:
                surfaces.append("filesystem")
        elif chk in ("process_running", "process", "proc", "port_listening", "port", "listen"):
            if "host_runtime" not in surfaces:
                surfaces.append("host_runtime")
        elif chk in ("command_output", "command", "cmd"):
            if "command" not in surfaces:
                surfaces.append("command")

    counts = {VERDICT_PASS: 0, VERDICT_FAIL: 0, VERDICT_UNVERIFIED: 0, VERDICT_UNSUPPORTED: 0}
    if v_results:
        for _, _, vr in v_results:
            v = getattr(vr, "result", VERDICT_PASS if getattr(vr, "success", False) else VERDICT_FAIL)
            counts[v] = counts.get(v, 0) + 1

    if not v_results:
        overall = VERDICT_UNVERIFIED
    elif counts[VERDICT_UNSUPPORTED] > 0:
        overall = VERDICT_UNSUPPORTED
    elif counts[VERDICT_UNVERIFIED] > 0:
        overall = VERDICT_UNVERIFIED
    elif counts[VERDICT_FAIL] > 0:
        overall = VERDICT_FAIL
    else:
        overall = VERDICT_PASS

    return {
        "surfaces": surfaces,
        "overall_result": overall,
        "counts": counts,
    }


class OutcomeVerifier:
    """Independent outcome verifier engine for deterministic state assertions."""

    @staticmethod
    def verify_criterion(criterion: Dict[str, Any], **kwargs: Any) -> VerificationResult:
        """Verify a parsed criterion dictionary synchronously (preserving typed context)."""
        merged = dict(kwargs)
        if isinstance(criterion.get("context"), dict):
            merged.setdefault("context", criterion["context"])
        return verify_outcome(
            check=str(criterion.get("check", "")),
            target=str(criterion.get("target", "")),
            expected=criterion.get("expected"),
            **merged,
        )

    @classmethod
    async def verify_criterion_async(cls, criterion: Dict[str, Any], **kwargs: Any) -> VerificationResult:
        """Verify a parsed criterion dictionary asynchronously (preserving typed context and test mocks)."""
        merged = dict(kwargs)
        if isinstance(criterion.get("context"), dict):
            merged.setdefault("context", criterion["context"])
        chk = str(criterion.get("check", ""))
        tgt = str(criterion.get("target", ""))
        exp = criterion.get("expected")
        norm_check = chk.strip().lower()
        # Honor monkeypatched/mocked OutcomeVerifier.verify and route synchronous checks cleanly through cls.verify.
        if cls.verify is not verify_outcome or (
            norm_check not in BROWSER_VERIFICATION_CHECKS
            and norm_check not in DESKTOP_VERIFICATION_CHECKS
        ):
            return cls.verify(chk, tgt, expected=exp, **merged)
        return await verify_outcome_async(
            check=chk,
            target=tgt,
            expected=exp,
            **merged,
        )

    verify = staticmethod(verify_outcome)
    verify_async = staticmethod(verify_outcome_async)
    validate_spec = staticmethod(validate_verification_spec)
    validate_criterion = staticmethod(validate_criterion_spec)
    summarize_scope = staticmethod(summarize_verification_scope)
    verify_file_exists = staticmethod(verify_file_exists)
    verify_file_contains = staticmethod(verify_file_contains)
    verify_file_checksum = staticmethod(verify_file_checksum)
    verify_file_json = staticmethod(verify_file_json)
    verify_process_running = staticmethod(verify_process_running)
    verify_port_listening = staticmethod(verify_port_listening)
    verify_command_output = staticmethod(verify_command_output)
    verify_browser_outcome = staticmethod(verify_browser_outcome)
    verify_browser_outcome_async = staticmethod(verify_browser_outcome_async)
    verify_desktop_outcome = staticmethod(verify_desktop_outcome)
    verify_desktop_outcome_async = staticmethod(verify_desktop_outcome_async)
