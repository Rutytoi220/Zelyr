"""Independent observable-state validators and assertions for Zelyr evaluations."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from zelyr.eval.types import ValidationResult


# --- Low-level assertion helpers returning Tuple[bool, str] ---

def assert_file_exists(workspace: Union[Path, Any], rel_path: str) -> Tuple[bool, str]:
    """Verify that a specific relative file exists inside the workspace directory."""
    ws = getattr(workspace, "workspace_dir", workspace)
    target = (ws / rel_path).resolve()
    if not str(target).startswith(str(ws.resolve())):
        return False, f"Path traversal error: '{rel_path}' escapes workspace"
    if target.is_file():
        return True, f"File '{rel_path}' exists on disk ({target.stat().st_size} bytes)."
    return False, f"File '{rel_path}' does not exist on disk."


def assert_file_content(
    workspace: Union[Path, Any],
    rel_path: str,
    expected_pattern: Union[str, List[str]],
    exact: bool = False,
) -> Tuple[bool, str]:
    """Verify that a workspace file exists and satisfies content matching rules."""
    ws = getattr(workspace, "workspace_dir", workspace)
    exists, reason = assert_file_exists(ws, rel_path)
    if not exists:
        return False, reason

    target = (ws / rel_path).resolve()
    try:
        content = target.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"Failed reading file '{rel_path}': {exc}"

    if isinstance(expected_pattern, (list, tuple)):
        for pat in expected_pattern:
            if pat not in content and not re.search(pat, content):
                return False, f"File '{rel_path}' does not contain expected pattern '{pat}'."
        return True, f"File '{rel_path}' contains all {len(expected_pattern)} expected patterns."

    if exact:
        if content == expected_pattern:
            return True, f"File '{rel_path}' exactly matches expected content."
        return False, f"File '{rel_path}' content does not match exactly."
    else:
        if expected_pattern in content or re.search(expected_pattern, content):
            return True, f"File '{rel_path}' contains expected pattern '{expected_pattern}'."
        return False, f"File '{rel_path}' does not contain expected pattern '{expected_pattern}'."


def assert_regex_match(
    workspace: Union[Path, Any],
    rel_path: str,
    pattern: str,
) -> Tuple[bool, str]:
    """Verify that a workspace file exists and matches a regular expression pattern."""
    ws = getattr(workspace, "workspace_dir", workspace)
    exists, reason = assert_file_exists(ws, rel_path)
    if not exists:
        return False, reason

    target = (ws / rel_path).resolve()
    try:
        content = target.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"Failed reading file '{rel_path}': {exc}"

    if re.search(pattern, content):
        return True, f"File '{rel_path}' matches regex pattern '{pattern}'."
    return False, f"File '{rel_path}' does not match regex pattern '{pattern}'."


def assert_memory_record(
    db_path: Path,
    category: str,
    key: str,
    expected_substr: str,
) -> Tuple[bool, str]:
    """Directly query SQLite database on disk to verify persistent memory row."""
    if not db_path.exists():
        return False, f"Memory database '{db_path}' does not exist."

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT content, is_active FROM memories WHERE category = ? AND key = ? AND is_active = 1",
            (category, key),
        )
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return False, f"No active memory record found for category='{category}', key='{key}'."

        matching = [r for r in rows if expected_substr in r["content"]]
        if matching:
            return True, f"Memory record verified for category='{category}', key='{key}' with '{expected_substr}'."
        return False, f"Found {len(rows)} memory record(s) for '{key}', but none contained substring '{expected_substr}'."
    except Exception as exc:
        return False, f"Database query failed: {exc}"


def assert_tool_called(
    tool_calls: Union[List[Dict[str, Any]], Any],
    tool_name: str,
    min_count: int = 1,
) -> Tuple[bool, str]:
    """Verify that a specific tool was dispatched at least min_count times."""
    calls = getattr(tool_calls, "tool_calls", tool_calls)
    if not isinstance(calls, list):
        calls = []
    count = sum(1 for c in calls if (c.get("name") or c.get("tool")) == tool_name)
    if count >= min_count:
        return True, f"Tool '{tool_name}' was called {count} time(s) (>= {min_count})."
    return False, f"Tool '{tool_name}' was called {count} time(s), expected at least {min_count}."


def assert_no_tool_errors(
    tool_calls_or_ctx: Union[List[Dict[str, Any]], Any],
) -> Tuple[bool, str]:
    """Verify that no unhandled tool exceptions or execution failure envelopes occurred."""
    if hasattr(tool_calls_or_ctx, "execution_error") and tool_calls_or_ctx.execution_error is not None:
        return False, f"Execution exception occurred: {tool_calls_or_ctx.execution_error}"

    calls = getattr(tool_calls_or_ctx, "tool_calls", tool_calls_or_ctx)
    if not isinstance(calls, list):
        calls = []

    for c in calls:
        res = c.get("result")
        name = c.get("name") or c.get("tool", "unknown")
        if isinstance(res, dict):
            if res.get("success") is False:
                err_msg = res.get("error") or res.get("remedy_hint") or "unknown error"
                return False, f"Tool '{name}' failed: {err_msg}"
        elif hasattr(res, "success") and not res.success:
            err_msg = getattr(res, "error", "unknown error")
            return False, f"Tool '{name}' failed: {err_msg}"

    return True, "No tool execution errors detected."


def assert_tool_sequence(
    tool_calls: List[Dict[str, Any]],
    expected_sequence: List[str],
) -> Tuple[bool, str]:
    """Verify that tool calls contain the expected sequence in order."""
    sequence = [c.get("name") or c.get("tool", "") for c in tool_calls]
    seq_idx = 0
    for name in sequence:
        if seq_idx < len(expected_sequence) and name == expected_sequence[seq_idx]:
            seq_idx += 1

    if seq_idx == len(expected_sequence):
        return True, f"Expected sequence {expected_sequence} observed in {sequence}."
    return False, f"Tool sequence {sequence} did not contain expected subsequence {expected_sequence}."


def assert_json_field(
    workspace: Path,
    rel_path: str,
    key_path: str,
    expected_val: Any,
) -> Tuple[bool, str]:
    """Parse JSON file on disk, traverse dot-separated key_path, and assert value equality."""
    exists, reason = assert_file_exists(workspace, rel_path)
    if not exists:
        return False, reason

    target = (workspace / rel_path).resolve()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed parsing JSON in '{rel_path}': {exc}"

    current = data
    keys = key_path.split(".")
    for part in keys:
        if isinstance(current, dict):
            if part not in current:
                return False, f"Key '{part}' in path '{key_path}' not found in '{rel_path}'."
            current = current[part]
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return False, f"Invalid list index '{part}' in path '{key_path}' for '{rel_path}'."
        else:
            return False, f"Cannot traverse key '{part}' in path '{key_path}' on non-collection in '{rel_path}'."

    if current == expected_val:
        return True, f"JSON field '{key_path}' in '{rel_path}' matches expected value {expected_val!r}."
    return False, f"JSON field '{key_path}' in '{rel_path}' is {current!r}, expected {expected_val!r}."


def assert_tool_updated(
    tools_dir: Path,
    tool_name: str,
    min_version_marker: str,
) -> Tuple[bool, str]:
    """Verify tool file exists on disk in tools_dir and contains expected updated code/markers."""
    tool_file = (tools_dir / f"{tool_name}.py").resolve()
    if not str(tool_file).startswith(str(tools_dir.resolve())):
        return False, f"Path traversal error for tool '{tool_name}' in '{tools_dir}'."
    if not tool_file.is_file():
        return False, f"Tool file '{tool_file.name}' does not exist in '{tools_dir}'."

    try:
        content = tool_file.read_text(encoding="utf-8")
    except Exception as exc:
        return False, f"Failed reading tool file '{tool_file.name}': {exc}"

    if min_version_marker in content:
        return True, f"Tool '{tool_name}' contains expected version marker '{min_version_marker}'."
    return False, f"Tool '{tool_name}' does not contain expected version marker '{min_version_marker}'."


# --- Higher-order validator adapters for backward compatibility ---

ValidatorFn = Callable[[Any], ValidationResult]


def validate_file_exists(relative_path: Union[str, Path]) -> ValidatorFn:
    rel = str(relative_path)
    def _val(context: Any) -> ValidationResult:
        workspace = getattr(context, "workspace_dir", Path("."))
        ok, reason = assert_file_exists(workspace, rel)
        return ValidationResult(success=ok, message=reason)
    return _val


def validate_file_content(
    relative_path: Union[str, Path],
    expected_substring: Optional[str] = None,
    exact_match: Optional[str] = None,
) -> ValidatorFn:
    rel = str(relative_path)
    def _val(context: Any) -> ValidationResult:
        workspace = getattr(context, "workspace_dir", Path("."))
        if exact_match is not None:
            ok, reason = assert_file_content(workspace, rel, exact_match, exact=True)
        else:
            ok, reason = assert_file_content(workspace, rel, expected_substring or "", exact=False)
        if ok:
            return ValidationResult(success=True, message=f"File '{rel}' content verified successfully.")
        return ValidationResult(success=False, message=reason)
    return _val


def validate_tool_called(tool_name: str, min_count: int = 1, exact_count: Optional[int] = None) -> ValidatorFn:
    def _val(context: Any) -> ValidationResult:
        calls = getattr(context, "tool_calls", [])
        if exact_count is not None:
            count = sum(1 for c in calls if (c.get("name") or c.get("tool")) == tool_name)
            ok = count == exact_count
            return ValidationResult(ok, f"Tool '{tool_name}' called {count} time(s), expected {exact_count}.")
        ok, reason = assert_tool_called(calls, tool_name, min_count=min_count)
        return ValidationResult(ok, reason)
    return _val


def validate_tool_sequence(expected_sequence: List[str], subsequence: bool = True) -> ValidatorFn:
    def _val(context: Any) -> ValidationResult:
        calls = getattr(context, "tool_calls", [])
        ok, reason = assert_tool_sequence(calls, expected_sequence)
        return ValidationResult(ok, reason)
    return _val


def validate_memory_record(category: Optional[str] = None, key: Optional[str] = None, content_substring: Optional[str] = None) -> ValidatorFn:
    def _val(context: Any) -> ValidationResult:
        db_path = getattr(context, "db_path", Path("."))
        ok, reason = assert_memory_record(db_path, category or "", key or "", content_substring or "")
        return ValidationResult(ok, reason)
    return _val


def combine_validators(*validators: ValidatorFn) -> ValidatorFn:
    def _combined(context: Any) -> ValidationResult:
        passed_messages = []
        for v in validators:
            res = v(context)
            if not res.success:
                return res
            if res.message:
                passed_messages.append(res.message)
        return ValidationResult(True, "; ".join(passed_messages) if passed_messages else "All validators passed.")
    return _combined


def validate_json_field(relative_path: Union[str, Path], key_path: str, expected_val: Any) -> ValidatorFn:
    rel = str(relative_path)
    def _val(context: Any) -> ValidationResult:
        workspace = getattr(context, "workspace_dir", Path("."))
        ok, reason = assert_json_field(workspace, rel, key_path, expected_val)
        return ValidationResult(ok, reason)
    return _val


def validate_tool_updated(tool_name: str, min_version_marker: str) -> ValidatorFn:
    def _val(context: Any) -> ValidationResult:
        tools_dir = getattr(context, "tools_directory", getattr(context, "tools_dir", Path(".")))
        ok, reason = assert_tool_updated(tools_dir, tool_name, min_version_marker)
        return ValidationResult(ok, reason)
    return _val

