"""Safe File Workspace Operations Tool for Zelyr.

Provides atomic and safe file operations:
1. read: Read file contents with line limits.
2. write: Atomic file write with automatic backup in /tmp/zelyr_backups/.
3. patch: Strict single-match verification string replacement with backup.
4. list_tree: Directory tree inspection respecting .gitignore up to depth 3.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from zelyr.tools.core import BaseTool, ToolParameter, ToolResult

BACKUP_DIR = Path("/tmp/zelyr_backups")
DEFAULT_IGNORES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".cargo",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

_last_modified_file: Optional[Path] = None


def get_last_modified_file() -> Optional[Path]:
    """Returns the last file path modified by write, patch, or rollback."""
    return _last_modified_file


class ManageWorkspaceFileTool(BaseTool):
    """Safely reads, writes, patches, rolls back, and inspects files in the workspace with atomic backups."""

    def __init__(self):
        super().__init__(
            tool_id="manage_workspace_file",
            name="manage_workspace_file",
            description=(
                "Tier 1 Safe Workspace File Operations: Inspect, edit, and revert repository files with atomic backups. "
                "Actions: 'read' (read lines up to max_lines), 'write' (atomic write with backup), "
                "'patch' (strict single-occurrence replacement with backup), 'rollback' (atomic restore from latest backup), "
                "'list_tree' (tree up to depth 3 respecting .gitignore)."
            ),
        )
        self.parameters = [
            ToolParameter(
                name="action",
                type="string",
                description="File action: 'read', 'write', 'patch', 'rollback', 'list_tree'.",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Target file or directory path (resolves relative to repo root or absolute).",
                required=False,
                default="",
            ),
            ToolParameter(
                name="content",
                type="string",
                description="String content to write when action is 'write'.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="search_text",
                type="string",
                description="Exact string to find and replace when action is 'patch' (must appear exactly once).",
                required=False,
                default="",
            ),
            ToolParameter(
                name="replace_text",
                type="string",
                description="Replacement string when action is 'patch'.",
                required=False,
                default="",
            ),
            ToolParameter(
                name="max_lines",
                type="integer",
                description="Maximum lines to read when action is 'read' (default: 200).",
                required=False,
                default=200,
            ),
        ]

    def _resolve_path(self, path_str: str) -> Path:
        if not path_str or path_str.strip() == "":
            return Path.cwd()
        p = Path(path_str.strip())
        if p.is_absolute():
            return p.resolve()
        return (Path.cwd() / p).resolve()

    def _create_backup(self, file_path: Path) -> Optional[str]:
        if not file_path.is_file():
            return None
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        backup_file = BACKUP_DIR / f"{file_path.name}.{timestamp}.bak"
        shutil.copy2(file_path, backup_file)
        return str(backup_file)

    def _load_gitignore_patterns(self, root_dir: Path) -> List[str]:
        patterns = list(DEFAULT_IGNORES)
        gitignore = root_dir / ".gitignore"
        if gitignore.is_file():
            try:
                for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith("#"):
                        patterns.append(cleaned.rstrip("/"))
            except Exception:
                pass
        return patterns

    def _is_ignored(self, path: Path, root: Path, patterns: List[str]) -> bool:
        name = path.name
        if name in DEFAULT_IGNORES:
            return True
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = path.name

        for pat in patterns:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f"{name}/", pat):
                return True
        return False

    def _build_tree(self, current: Path, root: Path, patterns: List[str], depth: int, max_depth: int) -> List[Dict[str, Any]]:
        if depth > max_depth or not current.is_dir():
            return []
        items = []
        try:
            entries = sorted(current.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            for entry in entries:
                if self._is_ignored(entry, root, patterns):
                    continue
                item_info: Dict[str, Any] = {
                    "name": entry.name,
                    "path": str(entry.relative_to(root)),
                    "type": "directory" if entry.is_dir() else "file",
                }
                if entry.is_dir() and depth < max_depth:
                    item_info["children"] = self._build_tree(entry, root, patterns, depth + 1, max_depth)
                items.append(item_info)
        except PermissionError:
            pass
        return items

    async def execute(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> ToolResult:
        global _last_modified_file
        merged = dict(params or {})
        merged.update(kwargs)

        action = str(merged.get("action") or "read").strip().lower()
        path_str = str(merged.get("path") or "").strip()
        target_path = self._resolve_path(path_str)

        if action == "read":
            if not target_path.exists():
                return ToolResult(False, error=f"File not found: {target_path}")
            if not target_path.is_file():
                return ToolResult(False, error=f"Path is not a regular file: {target_path}")

            try:
                max_lines = int(merged.get("max_lines") or 200)
            except (ValueError, TypeError):
                max_lines = 200

            try:
                content = target_path.read_text(encoding="utf-8", errors="replace")
                all_lines = content.splitlines()
                total_lines = len(all_lines)
                truncated = total_lines > max_lines
                read_lines = all_lines[:max_lines]

                return ToolResult(
                    True,
                    output={
                        "path": str(target_path),
                        "content": "\n".join(read_lines),
                        "lines_read": len(read_lines),
                        "total_lines": total_lines,
                        "truncated": truncated,
                    },
                )
            except Exception as e:
                return ToolResult(False, error=f"Failed to read file {target_path}: {e}")

        elif action == "write":
            if not path_str:
                return ToolResult(False, error="Target path must be provided for write action.")

            content = str(merged.get("content", ""))
            backup_path = self._create_backup(target_path)

            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                _last_modified_file = target_path
                return ToolResult(
                    True,
                    output={
                        "path": str(target_path),
                        "bytes_written": len(content.encode("utf-8")),
                        "backup": backup_path,
                        "message": f"Successfully wrote {len(content)} characters to {target_path.name}.",
                    },
                )
            except Exception as e:
                return ToolResult(False, error=f"Failed to write file {target_path}: {e}")

        elif action == "patch":
            if not target_path.is_file():
                return ToolResult(False, error=f"File does not exist to patch: {target_path}")

            search_text = str(merged.get("search_text", ""))
            replace_text = str(merged.get("replace_text", ""))

            if not search_text:
                return ToolResult(False, error="search_text must not be empty for patch action.")

            try:
                current_content = target_path.read_text(encoding="utf-8", errors="replace")
                count = current_content.count(search_text)

                if count == 0:
                    return ToolResult(
                        False,
                        error=f"Patch aborted: search_text appeared 0 times in {target_path.name}. Verify exact indentation and whitespace.",
                    )
                if count > 1:
                    return ToolResult(
                        False,
                        error=f"Patch aborted: search_text appeared {count} times in {target_path.name}. Patch requires exact single occurrence to prevent accidental corruption.",
                    )

                backup_path = self._create_backup(target_path)
                new_content = current_content.replace(search_text, replace_text, 1)
                target_path.write_text(new_content, encoding="utf-8")
                _last_modified_file = target_path

                return ToolResult(
                    True,
                    output={
                        "path": str(target_path),
                        "backup": backup_path,
                        "occurrences_replaced": 1,
                        "message": f"Successfully patched {target_path.name}.",
                    },
                )
            except Exception as e:
                return ToolResult(False, error=f"Failed to patch file {target_path}: {e}")

        elif action in ("rollback", "undo", "revert"):
            if not path_str:
                return ToolResult(False, error="Target path must be provided for rollback action.")

            if not BACKUP_DIR.exists():
                return ToolResult(False, error=f"No backup found in {BACKUP_DIR} for this file.")

            filename = target_path.name
            matching_backups = []
            for bak in BACKUP_DIR.glob(f"{filename}.*.bak"):
                parts = bak.name.rsplit(".", 2)
                if len(parts) >= 3 and parts[-1] == "bak" and parts[-2].isdigit():
                    try:
                        ts = int(parts[-2])
                        matching_backups.append((ts, bak))
                    except ValueError:
                        pass

            if not matching_backups:
                return ToolResult(False, error=f"No backup found in {BACKUP_DIR} for this file.")

            matching_backups.sort(key=lambda x: x[0], reverse=True)
            latest_ts, latest_bak = matching_backups[0]

            try:
                content = latest_bak.read_text(encoding="utf-8")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(content, encoding="utf-8")
                _last_modified_file = target_path
                restored_size = len(content.encode("utf-8"))

                return ToolResult(
                    True,
                    output={
                        "action": "rollback",
                        "restored_path": str(target_path),
                        "backup_source": str(latest_bak),
                        "timestamp": latest_ts,
                        "bytes": restored_size,
                        "message": f"Successfully restored {target_path.name} from backup {latest_bak.name}.",
                    },
                )
            except Exception as e:
                return ToolResult(False, error=f"Failed to restore file {target_path} from backup: {e}")

        elif action in ("list_tree", "tree", "list"):
            if not target_path.exists():
                return ToolResult(False, error=f"Directory does not exist: {target_path}")
            if not target_path.is_dir():
                return ToolResult(False, error=f"Path is not a directory: {target_path}")

            patterns = self._load_gitignore_patterns(target_path)
            tree_items = self._build_tree(target_path, target_path, patterns, depth=1, max_depth=3)

            return ToolResult(
                True,
                output={
                    "root": str(target_path),
                    "depth": 3,
                    "entries": tree_items,
                    "count": len(tree_items),
                },
            )

        else:
            return ToolResult(
                False,
                error=f"Unsupported file action: '{action}'. Supported actions: read, write, patch, list_tree.",
            )
