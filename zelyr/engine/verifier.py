"""Independent Outcome Verifier Engine for deterministic state assertions.

Standard-library based state verification for files, contents, checksums,
JSON formatting, processes, listening ports, and command outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class VerificationResult:
    """Immutable result of an outcome verification check."""

    success: bool
    check: str
    target: str
    observed: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a standard dictionary payload."""
        return asdict(self)


def _resolve_target_path(target_str: str, cwd: Optional[Union[str, Path]] = None) -> Path:
    """Resolve file path target respecting optional cwd or Path.cwd()."""
    p = Path(os.path.expanduser(target_str))
    if p.is_absolute():
        return p.resolve()
    base = Path(cwd).resolve() if cwd else Path.cwd()
    return (base / p).resolve()


def verify_file_exists(
    path: Union[str, Path],
    cwd: Optional[Union[str, Path]] = None,
) -> VerificationResult:
    """Verify that a target file or path exists."""
    target_str = str(path)
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd)
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
) -> VerificationResult:
    """Verify that a target file contains the specified pattern or regex."""
    target_str = str(path)
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd)
        if not expanded.exists() or not expanded.is_file():
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_str,
                observed={"exists": False},
                error=f"File not found: {target_str}",
            )

        content = expanded.read_text(encoding="utf-8", errors="replace")
        if regex:
            match = re.search(pattern, content)
            if match:
                return VerificationResult(
                    success=True,
                    check="file_contains",
                    target=target_str,
                    observed={"matched": True, "snippet": match.group(0)[:120], "regex": True},
                )
        else:
            if pattern in content:
                return VerificationResult(
                    success=True,
                    check="file_contains",
                    target=target_str,
                    observed={"matched": True, "snippet": pattern[:120], "regex": False},
                )

        return VerificationResult(
            success=False,
            check="file_contains",
            target=target_str,
            observed={"matched": False},
            error=f"Pattern '{pattern}' not found in {target_str}",
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
) -> VerificationResult:
    """Verify that a target file matches the expected SHA-256 digest."""
    target_str = str(path)
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd)
        if not expanded.exists() or not expanded.is_file():
            return VerificationResult(
                success=False,
                check="file_checksum",
                target=target_str,
                observed={"exists": False},
                error=f"File not found: {target_str}",
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
) -> VerificationResult:
    """Verify that a target file contains valid JSON and optionally matches required keys."""
    target_str = str(path)
    try:
        expanded = _resolve_target_path(target_str, cwd=cwd)
        if not expanded.exists() or not expanded.is_file():
            return VerificationResult(
                success=False,
                check="file_json",
                target=target_str,
                observed={"exists": False},
                error=f"File not found: {target_str}",
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
                            if pattern in cmd:
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
                        if pid != curr_pid and pattern in args:
                            if pid not in pids:
                                pids.append(pid)
        except Exception:
            pass

    if pids:
        return VerificationResult(
            success=True,
            check="process_running",
            target=pattern,
            observed={"matched_pids": sorted(pids), "count": len(pids)},
        )
    return VerificationResult(
        success=False,
        check="process_running",
        target=pattern,
        observed={"matched_pids": [], "count": 0},
        error=f"No process matching '{pattern}' found",
    )


def verify_port_listening(
    port: Union[int, str],
    host: str = "127.0.0.1",
    timeout: float = 0.5,
) -> VerificationResult:
    """Verify that a TCP port is open and accepting connections."""
    target_str = str(port)
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
                observed={"listening": False},
                error=f"Invalid port: {target_str}",
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


def verify_command_output(
    command: str,
    expected_pattern: str,
    timeout: float = 5.0,
    cwd: Optional[Union[str, Path]] = None,
) -> VerificationResult:
    """Verify that executing a shell command produces output matching expected_pattern."""
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
        out = res.stdout or ""
        err = res.stderr or ""
        matched = (expected_pattern in out) or (expected_pattern in err)
        if not matched:
            try:
                matched = bool(re.search(expected_pattern, out) or re.search(expected_pattern, err))
            except Exception:
                pass

        observed = {
            "exit_code": res.returncode,
            "stdout": out[:1000],
            "stderr": err[:1000],
            "matched": matched,
        }
        if matched:
            return VerificationResult(
                success=True,
                check="command_output",
                target=command,
                observed=observed,
            )
        return VerificationResult(
            success=False,
            check="command_output",
            target=command,
            observed=observed,
            error=f"Expected pattern '{expected_pattern}' not found in command output (exit {res.returncode})",
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            success=False,
            check="command_output",
            target=command,
            observed={"timeout": True},
            error=f"Command timed out after {timeout}s: {command}",
        )
    except Exception as exc:
        return VerificationResult(
            success=False,
            check="command_output",
            target=command,
            observed={},
            error=f"Failed to execute command '{command}': {exc}",
        )


def verify_outcome(
    check: str,
    target: str,
    expected: Optional[str] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Dispatch outcome verification based on check name."""
    norm_check = (check or "").strip().lower()
    cwd = kwargs.get("cwd")

    if norm_check in ("file_exists", "exists"):
        return verify_file_exists(target, cwd=cwd)
    elif norm_check in ("file_contains", "contains"):
        regex = kwargs.get("regex", False)
        return verify_file_contains(target, expected or "", regex=regex, cwd=cwd)
    elif norm_check in ("file_checksum", "checksum", "sha256"):
        return verify_file_checksum(target, expected or "", cwd=cwd)
    elif norm_check in ("file_json", "json"):
        schema = kwargs.get("schema")
        return verify_file_json(target, schema=schema, cwd=cwd)
    elif norm_check in ("process_running", "process", "proc"):
        return verify_process_running(target)
    elif norm_check in ("port_listening", "port", "listen"):
        host = kwargs.get("host", "127.0.0.1")
        timeout = float(kwargs.get("timeout", 0.5))
        return verify_port_listening(target, host=host, timeout=timeout)
    elif norm_check in ("command_output", "command", "cmd"):
        timeout = float(kwargs.get("timeout", 5.0))
        return verify_command_output(target, expected or "", timeout=timeout, cwd=cwd)
    else:
        return VerificationResult(
            success=False,
            check=check,
            target=target,
            observed=None,
            error=(
                f"Unknown verification check '{check}'. Supported checks: "
                "file_exists, file_contains, file_checksum, file_json, "
                "process_running, port_listening, command_output."
            ),
        )


class OutcomeVerifier:
    """Independent outcome verifier engine for deterministic state assertions."""

    @staticmethod
    def verify(
        check: str,
        target: str,
        expected: Optional[str] = None,
        **kwargs: Any,
    ) -> VerificationResult:
        """Verify state outcome for the given check type, target, and expectation."""
        return verify_outcome(check, target, expected, **kwargs)

    verify_file_exists = staticmethod(verify_file_exists)
    verify_file_contains = staticmethod(verify_file_contains)
    verify_file_checksum = staticmethod(verify_file_checksum)
    verify_file_json = staticmethod(verify_file_json)
    verify_process_running = staticmethod(verify_process_running)
    verify_port_listening = staticmethod(verify_port_listening)
    verify_command_output = staticmethod(verify_command_output)
