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
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union


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


@dataclass(frozen=True)
class VerificationResult:
    """Immutable result of an outcome verification check."""

    success: bool
    check: str
    target: str
    observed: Any
    error: Optional[str] = None

    @property
    def status(self) -> str:
        """Return structured outcome status category preserving 5-field dict compatibility."""
        if self.success:
            return "verified"
        obs = self.observed if isinstance(self.observed, dict) else {}
        if obs.get("safety_blocked") or obs.get("workspace_escape") or obs.get("interpreter_rejected"):
            return "rejected"
        if obs.get("timeout"):
            return "timeout"
        if obs.get("cancelled"):
            return "cancelled"
        if obs.get("invalid_spec"):
            return "invalid_spec"
        if obs.get("execution_error") or obs.get("truncated"):
            return "execution_error"
        return "failed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a standard dictionary payload."""
        return asdict(self)


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


def verify_outcome(
    check: str,
    target: str,
    expected: Optional[str] = None,
    **kwargs: Any,
) -> VerificationResult:
    """Dispatch outcome verification based on check name."""
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

    norm_check = check.strip().lower()
    target_val = str(target)
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
    elif norm_check in ("file_contains", "contains"):
        if expected is None or str(expected) == "":
            return VerificationResult(
                success=False,
                check="file_contains",
                target=target_val,
                observed={"pattern": expected, "invalid_spec": True},
                error="Incomplete verification specification: 'file_contains' requires expected pattern or content.",
            )
        regex = bool(kwargs.get("regex", False))
        return verify_file_contains(target_val, expected, regex=regex, cwd=cwd, allow_external=allow_external)
    elif norm_check in ("file_checksum", "checksum", "sha256"):
        return verify_file_checksum(target_val, expected or "", cwd=cwd, allow_external=allow_external)
    elif norm_check in ("file_json", "json"):
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
        return verify_file_json(target_val, schema=schema, cwd=cwd, allow_external=allow_external)
    elif norm_check in ("process_running", "process", "proc"):
        if expected is not None and str(expected).strip() != "":
            return VerificationResult(
                success=False,
                check="process_running",
                target=target_val,
                observed={"expected_content_provided": expected, "invalid_spec": True},
                error="Incompatible verification specification: 'process_running' only verifies process existence and does not accept expected content.",
            )
        return verify_process_running(target_val)
    elif norm_check in ("port_listening", "port", "listen"):
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
        return verify_port_listening(target_spec, host=host, timeout=timeout)
    elif norm_check in ("command_output", "command", "cmd"):
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
        return verify_command_output(target_val, expected or "", timeout=timeout, cwd=cwd, cancel_event=cancel_event)
    else:
        return VerificationResult(
            success=False,
            check=check,
            target=target_val,
            observed={"invalid_spec": True},
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
